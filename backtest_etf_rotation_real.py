# -*- coding: utf-8 -*-
"""行业ETF轮动回测 · 真实ETF数据
================================
用37只ETF(含行业+宽基)日线, Walk-Forward月度调仓
多因子打分: 动量+波动率+量比+回撤恢复 → Top3-5等权
DD_SMART v2门禁: 回撤<-12%→空仓, 恢复>+10%→重新入场
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("行业ETF轮动 · 真实ETF数据 WF回测")
print("=" * 60)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# === 1. 加载ETF日线 ===
etf_raw = con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, volume, name
    FROM etf_daily ORDER BY ts_code, trade_date
""").df()
etf_raw['trade_date'] = pd.to_datetime(etf_raw['trade_date'])

# 过滤: 至少3年历史, 剔除ETF联接
etf_cnt = etf_raw.groupby('ts_code')['trade_date'].nunique()
valid_etfs = etf_cnt[etf_cnt > 750].index.tolist()  # 3年≈750交易日
etf = etf_raw[etf_raw['ts_code'].isin(valid_etfs)].copy()
print(f"[1] ETF: {len(valid_etfs)}只有效 (>3年历史)")

# HS300基准
hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close']
con.close()

# === 2. 计算月度因子 ===
etf = etf.sort_values(['ts_code', 'trade_date'])
etf['month'] = etf['trade_date'].dt.to_period('M')

# 每日ETF收益率
etf['ret'] = etf.groupby('ts_code')['close'].pct_change()

# 月末数据
monthly = etf.groupby(['ts_code', 'month']).agg(
    close=('close', 'last'),
    name=('name', 'first'),
    ret_1m=('ret', lambda x: (1+x).prod()-1),  # 月收益
    ret_3m=('ret', lambda x: (1+x.tail(63)).prod()-1 if len(x)>=40 else np.nan),
    vol_20d=('ret', lambda x: x.tail(20).std()*np.sqrt(252) if len(x)>=15 else np.nan),
    vol_ratio=('volume', lambda x: x.tail(5).mean()/x.tail(60).mean() if len(x)>=50 else 1.0),
    dd_1y=('close', lambda x: x.iloc[-1]/x.tail(252).max()-1 if len(x)>=200 else np.nan),
    recovery=('close', lambda x: x.iloc[-1]/x.tail(252).min()-1 if len(x)>=200 else np.nan),
    avg_volume=('volume', lambda x: x.tail(60).mean()),
    n_days=('trade_date', 'nunique'),
).reset_index()

monthly['month'] = monthly['month'].dt.to_timestamp()

# 过滤流动性: 日均成交>2千万
monthly = monthly[monthly['avg_volume'] > 20000000]

# === 3. 多因子打分(每月横截面排名) ===
# 因子1: 1月动量(正)
monthly['f_mom1'] = monthly.groupby('month')['ret_1m'].rank(pct=True)

# 因子2: 3月动量(正)
monthly['f_mom3'] = monthly.groupby('month')['ret_3m'].rank(pct=True, ascending=True)  # 低3月动量可能超跌反弹

# 因子3: 低波(正) — 波动率越低越好
monthly['f_lowvol'] = monthly.groupby('month')['vol_20d'].rank(pct=True, ascending=False)

# 因子4: 量比(正) — 近期放量
monthly['f_volratio'] = monthly.groupby('month')['vol_ratio'].rank(pct=True)

# 因子5: 回撤恢复(正) — 从1年低点恢复中
monthly['f_recovery'] = monthly.groupby('month')['recovery'].rank(pct=True)

# 综合得分(等权)
score_cols = ['f_mom1', 'f_mom3', 'f_lowvol', 'f_volratio', 'f_recovery']
monthly['score'] = monthly[score_cols].mean(axis=1)

# 目标: 下月收益
monthly['fwd_ret'] = monthly.groupby('ts_code')['ret_1m'].shift(-1)

monthly = monthly.dropna(subset=['fwd_ret', 'score'])
print(f"[2] 月度信号: {len(monthly)}行, {monthly['ts_code'].nunique()}只ETF")

# === 4. Walk-Forward回测 ===
START_YEAR = 2016
END_YEAR = 2026
GATE_DD_EXIT = -0.12   # 大盘回撤超过12%→空仓
GATE_REENTRY = 0.10    # 从低点反弹10%→重新入场
COST_MONTHLY = 0.003   # 30bp/月(ETF佣金+滑点)

# 计算DD_SMART门禁状态
hs300_m = hs300.resample('ME').last()
hs300_peak = hs300_m.rolling(252).max()
hs300_dd = hs300_m / hs300_peak - 1
# 门禁状态
in_drawdown = False
dd_low = 1.0
gate_states = {}
for d in hs300_dd.index:
    dd = hs300_dd[d]
    if not in_drawdown and dd < GATE_DD_EXIT:
        in_drawdown = True
        dd_low = hs300_m[d] / hs300_peak[d]  # 触发时的低点比例
    if in_drawdown:
        dd_low = min(dd_low, hs300_m[d] / hs300_peak[d])
        recovery = (hs300_m[d] / hs300_peak[d]) / dd_low - 1
        if recovery > GATE_REENTRY:
            in_drawdown = False
    gate_states[d] = in_drawdown

years = list(range(START_YEAR + 3, END_YEAR + 1))  # 3年预热
print(f"\n[3] WF回测 {years[0]}-{years[-1]}")

results = []
for test_yr in years:
    train_end = pd.Timestamp(f'{test_yr}-01-01')
    test_end = pd.Timestamp(f'{test_yr}-12-31')

    for m in monthly[monthly['month'].dt.year == test_yr]['month'].unique():
        if m > test_end:
            continue

        this_month = monthly[monthly['month'] == m].copy()
        if len(this_month) < 5:
            continue

        # 门禁检查
        gate_date = m + pd.DateOffset(months=0)
        # 找到最近的月末日期
        closest_date = min(gate_states.keys(), key=lambda d: abs((d - gate_date).days))
        gate_on = gate_states.get(closest_date, False)

        if gate_on:
            results.append({
                'month': m,
                'ret': 0.0,  # 空仓
                'n_etfs': 0,
                'gate': 'OFF',
                'top_etfs': '',
            })
            continue

        # 选Top ETF
        top_n = 5
        top = this_month.nlargest(top_n, 'score')
        avg_ret = top['fwd_ret'].mean()
        etf_names = ', '.join(top['name'].head(3).tolist())

        results.append({
            'month': m,
            'ret': avg_ret - COST_MONTHLY,
            'n_etfs': len(top),
            'gate': 'ON',
            'top_etfs': etf_names,
        })

res_df = pd.DataFrame(results).sort_values('month')
print(f"[4] 交易: {len(res_df)}月, 空仓{(res_df['gate']=='OFF').sum()}月")

# === 5. 评估 ===
print(f"\n{'='*60}")
print("回测结果")
print(f"{'='*60}")

monthly_rets = res_df['ret'].values
cum = np.prod(1 + monthly_rets)
annual_ret = cum ** (12 / len(monthly_rets)) - 1
annual_vol = np.std(monthly_rets) * np.sqrt(12)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
mdd = 0
peak = 1.0
for r in monthly_rets:
    peak = max(peak, 1 + r)
    mdd = min(mdd, (1 + r) / peak - 1)

# 胜率
win_rate = np.mean(monthly_rets > 0)

# 年份拆解
print(f"年化收益: {annual_ret*100:+.1f}%")
print(f"年化波动: {annual_vol*100:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"最大回撤: {mdd*100:.1f}%")
print(f"月胜率: {win_rate*100:.0f}%")
print(f"累积收益: {cum-1:+.1%}")
print(f"交易月数: {len(monthly_rets)}")

# 年份分析
res_df['year'] = res_df['month'].dt.year
print(f"\n年度收益:")
for yr, grp in res_df.groupby('year'):
    yr_ret = np.prod(1 + grp['ret'].values) - 1
    gate_off = (grp['gate'] == 'OFF').sum()
    print(f"  {yr}: {yr_ret*100:+6.1f}%  (空仓{gate_off}月)")

# vs HS300
hs300_yearly = {}
for yr, grp in res_df.groupby('year'):
    hs300_yr = hs300_m[hs300_m.index.year == yr]
    if len(hs300_yr) > 1:
        hs300_yearly[yr] = hs300_yr.iloc[-1] / hs300_yr.iloc[0] - 1

print(f"\nvs 沪深300:")
for yr in sorted(hs300_yearly.keys()):
    strategy_yr = res_df[res_df['year'] == yr]['ret'].apply(lambda x: 1+x).prod() - 1
    print(f"  {yr}: 策略{strategy_yr*100:+5.1f}%  HS300{hs300_yearly[yr]*100:+5.1f}%")

# === 6. vs 等权买入持有 ===
eq_rets = []
for m in res_df['month']:
    this_m = monthly[monthly['month'] == m]
    if len(this_m) > 3:
        eq_rets.append(this_m['fwd_ret'].mean())

if eq_rets:
    eq_cum = np.prod(1 + np.array(eq_rets))
    eq_annual = eq_cum ** (12 / len(eq_rets)) - 1
    print(f"\nvs ETF等权: 策略{annual_ret*100:+.1f}% vs 等权{eq_annual*100:+.1f}%")

# === 7. 最新信号 ===
latest_m = monthly['month'].max()
latest = monthly[monthly['month'] == latest_m].nlargest(5, 'score')
print(f"\n最新信号({latest_m.date()}):")
for _, r in latest.iterrows():
    print(f"  {r['ts_code']} {r['name']:12s}  score={r['score']:.2f}  mom1={r['ret_1m']*100:+.1f}%  mom3={r['ret_3m']*100:+.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
