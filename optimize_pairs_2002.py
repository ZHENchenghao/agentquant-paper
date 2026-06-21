# -*- coding: utf-8 -*-
"""小众配对优化 · 2002-2026 全周期"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from itertools import combinations
t0 = time.time()

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

kline = con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(close*total_share/10000,close*vol/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
con.close()

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m
del kline; gc.collect()

FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS = list(combinations(FEATS, 2))
YEARS = sorted(set(d.year for d in monthly_dates)); TRAIN = 5; FIRST = YEARS[0]+TRAIN
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095; TOP_N = 30; COST = 0.0033

print("=" * 70)
print(f"小众配对优化 · 2002-2026全周期 (WF {FIRST}-{YEARS[-1]})")
print("=" * 70)

# === 每对单独 ===
pair_perf = {}
for fa, fb in ALL_PAIRS:
    results = []
    for yr in range(FIRST, YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 50: continue
            for f in [fa, fb]:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score'] = day[fa+'_r'] * day[fb+'_r']
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue
            top = day.nlargest(TOP_N, 'score')
            if len(top) >= 10: results.append(top['fwd_ret'].mean()-COST)

    if results:
        arr = np.array(results); n = len(arr)
        cum = np.prod(1+arr); ann = cum**(12/n)-1
        c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
        vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
        win = np.mean(arr>0)*100
        pair_perf[(fa,fb)] = {'ann':ann,'sh':sh,'mdd':mdd,'win':win,'n':n}

print(f"\n{'排名':<4s} {'因子对':<28s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>7s} {'胜率':>6s}")
print('-'*65)
for rank, ((fa,fb), r) in enumerate(sorted(pair_perf.items(), key=lambda x: x[1]['ann'], reverse=True)):
    print(f'{rank+1:<4d} {fa[:8]:>8s} x {fb:<8s}  {r["ann"]*100:+7.1f}% {r["sh"]:+6.2f} {r["mdd"]*100:+6.1f}% {r["win"]:>5.0f}%')

# === 多对组合 ===
print(f"\n多对组合:")
sorted_pairs = sorted(pair_perf.items(), key=lambda x: x[1]['ann'], reverse=True)
for n_pairs in [1, 2, 3, 4, 5, 6, 8, 15]:
    top_pairs = [p for p, _ in sorted_pairs[:n_pairs]]
    results = []
    for yr in range(FIRST, YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 50: continue
            all_f = list(set([x for p in top_pairs for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0; valid_n = 0
            for fa, fb in top_pairs:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r'] * day[fb+'_r']; valid_n += 1
            if valid_n == 0: continue
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue
            top = day.nlargest(TOP_N, 'score')
            if len(top) >= 10: results.append(top['fwd_ret'].mean()-COST)

    if results:
        arr = np.array(results); n = len(arr)
        cum = np.prod(1+arr); ann = cum**(12/n)-1
        c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
        vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
        vs_best = ann - pair_perf[sorted_pairs[0][0]]['ann']
        flag = ' <--' if n_pairs == 4 else ''
        print(f"  Top{n_pairs:>2d}对: 年化{ann*100:+7.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+6.1f}% vs_best{vs_best*100:+6.1f}%{flag}")

print(f"\n耗时: {time.time()-t0:.0f}s")
