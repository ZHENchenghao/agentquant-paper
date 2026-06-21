# -*- coding: utf-8 -*-
"""ETF轮动 v2 · 剥离门禁+对比多组因子+Top3/5/10
==============================================
测试矩阵:
  门禁: 无 / DD_SMART轻(-15%/-8%) / DD_SMART重(-10%/-15%)
  因子: 纯动量 / 动量+低波 / 动量+量比 / 全因子
  持仓: Top3 / Top5 / Top10
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("ETF轮动 v2 · 多配置对比")
print("=" * 60)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# === 1. 加载 ===
etf_raw = con.execute("SELECT ts_code, trade_date, open, high, low, close, volume, name FROM etf_daily ORDER BY ts_code, trade_date").df()
etf_raw['trade_date'] = pd.to_datetime(etf_raw['trade_date'])

hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300 = hs300.set_index('trade_date')['close']

# 北向
nb = con.execute("SELECT trade_date, net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date'] = pd.to_datetime(nb['trade_date']); nb = nb.set_index('trade_date')['net_flow']
con.close()

# 过滤
etf_cnt = etf_raw.groupby('ts_code')['trade_date'].nunique()
valid_etfs = etf_cnt[etf_cnt > 750].index.tolist()
etf = etf_raw[etf_raw['ts_code'].isin(valid_etfs)].sort_values(['ts_code', 'trade_date'])
etf['ret'] = etf.groupby('ts_code')['close'].pct_change()

# === 2. 月度因子 ===
etf['month'] = etf['trade_date'].dt.to_period('M')
monthly = etf.groupby(['ts_code', 'month']).agg(
    close=('close', 'last'),
    name=('name', 'first'),
    ret_1m=('ret', lambda x: (1+x).prod()-1),
    ret_3m=('ret', lambda x: (1+x.tail(63)).prod()-1 if len(x)>=40 else np.nan),
    ret_6m=('ret', lambda x: (1+x.tail(126)).prod()-1 if len(x)>=100 else np.nan),
    vol_20d=('ret', lambda x: x.tail(20).std()*np.sqrt(252) if len(x)>=15 else np.nan),
    vol_ratio=('volume', lambda x: x.tail(5).mean()/x.tail(60).mean() if len(x)>=50 else 1.0),
    avg_volume=('volume', lambda x: x.tail(60).mean()),
    dd_1y=('close', lambda x: x.iloc[-1]/x.tail(252).max()-1 if len(x)>=200 else np.nan),
).reset_index()
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly = monthly[monthly['avg_volume'] > 20000000]
print(f"[1] 月度: {len(monthly)}行, {monthly['ts_code'].nunique()}只")

# === 3. 门禁计算 ===
hs300_m = hs300.resample('ME').last()
hs300_peak = hs300_m.rolling(504).max()  # 2年
hs300_dd = hs300_m / hs300_peak - 1

# 北向月度
nb_m = nb.resample('ME').sum()

def make_gate(name, exit_dd, reentry, use_nb=False):
    """构建门禁状态序列"""
    in_dd = False; dd_low = 1.0
    states = {}
    for d in hs300_dd.index:
        dd = hs300_dd[d]
        nb_signal = True
        if use_nb and d in nb_m.index:
            nb_signal = nb_m[d] > -50  # 北向月流出<50亿才触发

        if not in_dd and dd < exit_dd and nb_signal:
            in_dd = True
            dd_low = hs300_m[d] / hs300_peak[d]
        if in_dd:
            dd_low = min(dd_low, hs300_m[d] / hs300_peak[d])
            recovery = (hs300_m[d] / hs300_peak[d]) / dd_low - 1
            if recovery > reentry:
                in_dd = False
        states[d] = in_dd
    return states

GATES = {
    '无门禁': None,
    'DD轻(-15%/-8%)': make_gate('light', -0.15, 0.08),
    'DD重(-10%/-15%)': make_gate('heavy', -0.10, 0.15),
}

# === 4. 因子组合 ===
FACTOR_SETS = {
    '纯动量(1m)': {'cols': ['f_mom1'], 'ascending': [True]},
    '动量(1m+3m)': {'cols': ['f_mom1', 'f_mom3'], 'ascending': [True, True]},
    '动量+低波': {'cols': ['f_mom1', 'f_mom3', 'f_lowvol'], 'ascending': [True, True, True]},
    '动量+量比': {'cols': ['f_mom1', 'f_mom3', 'f_volratio'], 'ascending': [True, True, True]},
}

# 计算因子
monthly['f_mom1'] = monthly.groupby('month')['ret_1m'].rank(pct=True)
monthly['f_mom3'] = monthly.groupby('month')['ret_3m'].rank(pct=True)
monthly['f_lowvol'] = monthly.groupby('month')['vol_20d'].rank(pct=True, ascending=False)
monthly['f_volratio'] = monthly.groupby('month')['vol_ratio'].rank(pct=True)
monthly['fwd_ret'] = monthly.groupby('ts_code')['ret_1m'].shift(-1)
monthly = monthly.dropna(subset=['fwd_ret'])

# === 5. 全组合回测 ===
POSITIONS = [3, 5, 10]
START_YEAR = 2016; years = list(range(START_YEAR + 3, 2027))
COST = 0.003

print(f"\n[2] 测试 {len(GATES)}门禁 x {len(FACTOR_SETS)}因子 x {len(POSITIONS)}持仓 = {len(GATES)*len(FACTOR_SETS)*len(POSITIONS)}组合")
print(f"    WF: {years[0]}-{years[-1]}")

all_results = []

for gate_name, gate_states in GATES.items():
    for factor_name, factor_cfg in FACTOR_SETS.items():
        # 计算得分
        score = monthly[factor_cfg['cols']].mean(axis=1)
        monthly_tmp = monthly.copy()
        monthly_tmp['score'] = score

        for top_n in POSITIONS:
            rets = []
            for test_yr in years:
                for m in monthly_tmp[monthly_tmp['month'].dt.year == test_yr]['month'].unique():
                    if m > pd.Timestamp(f'{test_yr}-12-31'):
                        continue

                    this_m = monthly_tmp[monthly_tmp['month'] == m]
                    if len(this_m) < top_n:
                        continue

                    # 门禁
                    if gate_states is not None:
                        closest = min(gate_states.keys(), key=lambda d: abs((d - m).days))
                        if gate_states.get(closest, False):
                            rets.append(0.0)
                            continue

                    top = this_m.nlargest(top_n, 'score')
                    rets.append(top['fwd_ret'].mean() - COST)

            if not rets:
                continue
            rets = np.array(rets)
            cum = np.prod(1 + rets)
            n_years = len(rets) / 12
            ann = cum ** (1 / n_years) - 1
            vol = np.std(rets) * np.sqrt(12)
            sharpe = ann / vol if vol > 0 else 0
            mdd = 0; peak = 1.0
            for r in rets:
                peak = max(peak, 1+r)
                mdd = min(mdd, (1+r)/peak - 1)
            win = np.mean(rets > 0)

            all_results.append({
                '门禁': gate_name, '因子': factor_name, '持仓': top_n,
                '年化': ann, '波动': vol, 'Sharpe': sharpe, 'MDD': mdd,
                '胜率': win, '累积': cum-1, '月数': len(rets),
            })

# === 6. 排名输出 ===
res_df = pd.DataFrame(all_results).sort_values('年化', ascending=False)
print(f"\n{'='*80}")
print("排名(按年化)")
print(f"{'='*80}")
print(f"{'门禁':<16s} {'因子':<16s} {'持仓':>4s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>8s} {'胜率':>6s} {'累积':>8s}")
print("-" * 80)

for _, r in res_df.head(20).iterrows():
    print(f"{r['门禁']:<16s} {r['因子']:<16s} {r['持仓']:>4d} {r['年化']*100:>+7.1f}% {r['Sharpe']:>6.2f} {r['MDD']*100:>7.1f}% {r['胜率']*100:>5.0f}% {r['累积']*100:>+7.1f}%")

# === 7. 最佳策略详细分析 ===
best = res_df.iloc[0]
print(f"\n{'='*80}")
print(f"最佳: {best['门禁']} × {best['因子']} × Top{int(best['持仓'])}")
print(f"{'='*80}")
print(f"年化: {best['年化']*100:+.1f}% | Sharpe: {best['Sharpe']:.2f} | MDD: {best['MDD']*100:.1f}%")

# vs HS300
hs300_yearly = {}
for yr in range(2019, 2027):
    hs300_yr = hs300_m[hs300_m.index.year == yr]
    if len(hs300_yr) > 1:
        hs300_yearly[yr] = hs300_yr.iloc[-1] / hs300_yr.iloc[0] - 1

# 等权基准
monthly_all = monthly.dropna(subset=['fwd_ret'])
eq_by_month = monthly_all.groupby('month')['fwd_ret'].mean()
eq_annual = (1+eq_by_month).prod() ** (12/len(eq_by_month)) - 1
# 等权重算
eq_rets_full = []
for yr in years:
    for m in monthly_all[monthly_all['month'].dt.year == yr]['month'].unique():
        grp = monthly_all[monthly_all['month'] == m]
        if len(grp) > 3:
            eq_rets_full.append(grp['fwd_ret'].mean())
eq_ret_arr = np.array(eq_rets_full)
eq_cum = np.prod(1+eq_ret_arr)
eq_ann = eq_cum ** (12/len(eq_ret_arr)) - 1
print(f"基准(等权买入持有): 年化{eq_ann*100:+.1f}% 累积{eq_cum-1:+.1%} MDD{(np.min(np.minimum.accumulate(np.cumprod(1+eq_ret_arr))/np.maximum.accumulate(np.cumprod(1+eq_ret_arr)))-1)*100:.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
