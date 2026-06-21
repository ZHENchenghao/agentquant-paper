# -*- coding: utf-8 -*-
"""ETF轮动 v3 · 双模切换 + 真实ETF
====================================
牛熊判断: HS300>MA200 且 MA200斜率>0 → BULL → 动量Top5
         否则 → BEAR → 低波Top5
回撤>12% → 强制空仓(FLOOR)
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("ETF轮动 v3 · 双模切换")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ETF数据
etf_all = con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, volume, name
    FROM etf_daily ORDER BY ts_code, trade_date
""").df()
etf_all['trade_date'] = pd.to_datetime(etf_all['trade_date'])

# HS300
hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300 = hs300.set_index('trade_date')['close']
con.close()

# 过滤: >3年历史
etf_cnt = etf_all.groupby('ts_code')['trade_date'].nunique()
valid = etf_cnt[etf_cnt > 750].index.tolist()
etf = etf_all[etf_all['ts_code'].isin(valid)].sort_values(['ts_code','trade_date'])
etf['ret'] = etf.groupby('ts_code')['close'].pct_change()
print(f"[1] ETF: {len(valid)}只")

# 月度因子
etf['month'] = etf['trade_date'].dt.to_period('M')
monthly = etf.groupby(['ts_code','month']).agg(
    close=('close','last'), name=('name','first'),
    ret_1m=('ret',lambda x:(1+x).prod()-1),
    ret_3m=('ret',lambda x:(1+x.tail(63)).prod()-1 if len(x)>=40 else np.nan),
    vol_1m=('ret',lambda x:x.tail(20).std()*np.sqrt(252) if len(x)>=15 else np.nan),
    avg_vol=('volume',lambda x:x.tail(60).mean()),
).reset_index()
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly = monthly[monthly['avg_vol'] > 20000000]  # 流动性
monthly['fwd_ret'] = monthly.groupby('ts_code')['ret_1m'].shift(-1)
monthly = monthly.dropna(subset=['fwd_ret','ret_1m'])

# 牛熊判断
hs300_m = hs300.resample('ME').last()
hs300_ma200 = hs300.rolling(200).mean().resample('ME').last()
hs300_ma200_slope = hs300_ma200.diff(3)  # 3月变化

# 门禁
hs300_h2y = hs300_m.rolling(24).max()
hs300_dd = hs300_m / hs300_h2y - 1

print(f"[2] 月度: {len(monthly)}行, {monthly['ts_code'].nunique()}只, {monthly['month'].nunique()}月")

# === 策略对比 ===
YEARS = sorted(set(d.year for d in monthly['month']))
TRAIN = 5; WF_START = YEARS[0] + TRAIN + 1
COST = 0.003

strategies = {
    'old_动量Top5': {'mode': 'mom', 'top': 5, 'gate': False},
    'old_等权': {'mode': 'eq', 'top': 0, 'gate': False},
    '双模(动量/低波)': {'mode': 'dual', 'top': 5, 'gate': False},
    '双模+门禁': {'mode': 'dual', 'top': 5, 'gate': True},
}

all_result = {}
for sname, cfg in strategies.items():
    long_r = []; eq_r = []
    in_dd = False; dd_low = 1.0

    for yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
        test = monthly[(monthly['month']>=ts)&(monthly['month']<=te)]

        for m, grp in test.groupby('month'):
            if len(grp) < 5: continue
            grp = grp.dropna(subset=['fwd_ret'])
            if len(grp) < 3: continue

            # 门禁
            gate_pos = 1.0
            if cfg['gate']:
                closest = min(hs300_dd.index, key=lambda d: abs((d-m).days))
                dd_val = hs300_dd.get(closest, 0)
                if pd.notna(dd_val):
                    if not in_dd and dd_val < -0.12:
                        in_dd = True; dd_low = hs300_m.get(closest, 1)
                    if in_dd:
                        dd_low = min(dd_low, hs300_m.get(closest, dd_low))
                        recovery = hs300_m.get(closest, 1)/dd_low - 1 if dd_low > 0 else 0
                        if recovery > 0.10:
                            in_dd = False
                        else:
                            gate_pos = 0.10  # FLOOR

            if gate_pos < 0.01:
                long_r.append(0.0); eq_r.append(grp['fwd_ret'].mean()); continue

            grp = grp.copy()
            n = max(1, min(5, len(grp)//4)) if cfg['top'] > 0 else len(grp)

            if cfg['mode'] == 'eq':
                top = grp
            elif cfg['mode'] == 'mom':
                grp['score'] = grp['ret_1m'].rank(pct=True)
                top = grp.nlargest(n, 'score')
            elif cfg['mode'] == 'dual':
                # 判断牛熊
                closest_m = min(hs300_ma200.index, key=lambda d: abs((d-m).days))
                price = hs300_m.get(closest_m, 0)
                ma200 = hs300_ma200.get(closest_m, 0)
                slope = hs300_ma200_slope.get(closest_m, 0)

                is_bull = pd.notna(price) and pd.notna(ma200) and price > ma200 and pd.notna(slope) and slope > 0

                if is_bull:
                    grp['score'] = grp['ret_1m'].rank(pct=True)  # 动量
                else:
                    grp['score'] = grp['vol_1m'].rank(pct=True, ascending=False)  # 低波

                top = grp.nlargest(n, 'score')
            else:
                top = grp

            long_r.append((top['fwd_ret'].mean() - COST) * gate_pos)
            eq_r.append(grp['fwd_ret'].mean())

    if long_r:
        la = np.array(long_r); n = len(la)
        lac = np.prod(1+la); la_ann = lac**(12/n)-1
        c = np.cumprod(1+la); mdd = np.min(c/np.maximum.accumulate(c)-1)
        vol = np.std(la)*np.sqrt(12); sh = la_ann/vol if vol>0 else 0
        ea = np.array(eq_r); eq_ann = np.prod(1+ea)**(12/n)-1
        all_result[sname] = {'ann':la_ann,'sh':sh,'mdd':mdd,'cum':lac-1,'eq_ann':eq_ann,'n':n}

print(f"\n[3] 策略对比 (WF {WF_START}-{YEARS[-1]}):")
print(f"{'策略':<20s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>7s} {'vs等权':>8s} {'累积':>8s}")
print('-'*65)
for sname, r in sorted(all_result.items(), key=lambda x: x[1]['ann'], reverse=True):
    vs_eq = r['ann'] - r['eq_ann']
    print(f"{sname:<20s} {r['ann']*100:+7.1f}% {r['sh']:+6.2f} {r['mdd']*100:+6.1f}% {vs_eq*100:+7.1f}% {r['cum']*100:+7.1f}%")

# 分年
best_name = max(all_result, key=lambda x: all_result[x]['ann'])
print(f"\n[4] {best_name} 分年:")
for yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
    test = monthly[(monthly['month']>=ts)&(monthly['month']<=te)]
    yr_ret = []; yr_eq = []
    in_dd_local = False; dd_low_local = 1.0
    for m, grp in test.groupby('month'):
        if len(grp)<5: continue
        grp = grp.dropna(subset=['fwd_ret'])
        if len(grp)<3: continue
        # simplified dual mode
        closest_m = min(hs300_ma200.index, key=lambda d: abs((d-m).days))
        price = hs300_m.get(closest_m, 0)
        ma200 = hs300_ma200.get(closest_m, 0)
        slope = hs300_ma200_slope.get(closest_m, 0)
        is_bull = pd.notna(price) and pd.notna(ma200) and price>ma200 and pd.notna(slope) and slope>0
        grp = grp.copy()
        grp['score'] = grp['ret_1m'].rank(pct=True) if is_bull else grp['vol_1m'].rank(pct=True, ascending=False)
        n = max(1, min(5, len(grp)//4))
        top = grp.nlargest(n,'score')
        yr_ret.append(top['fwd_ret'].mean()-COST)
        yr_eq.append(grp['fwd_ret'].mean())
    if yr_ret:
        yr_cum = np.prod(1+np.array(yr_ret))-1
        eq_cum = np.prod(1+np.array(yr_eq))-1
        mkt = '牛' if is_bull else '熊'
        print(f"  {yr}: 策略{yr_cum*100:+6.1f}%  等权{eq_cum*100:+6.1f}%  {mkt}市{len(yr_ret)}月")

print(f"\n耗时: {time.time()-t0:.0f}s")
