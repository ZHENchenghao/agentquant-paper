# -*- coding: utf-8 -*-
"""小众因子配对优化 · 15对全测 + 权重方案
==========================================
问题: 当前4对等权, 是否有更优组合?
测试: ①Top1/Top2/.../Top8对 ②IR加权 vs 等权 ③剔除弱因子
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from itertools import combinations
t0 = time.time()

print("=" * 70)
print("小众因子配对优化")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 因子
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

# K线
kline = con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(close*total_share/10000,close*vol/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300['high_2y'] = hs300['close'].rolling(504).max()
con.close()

FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS = list(combinations(FEATS, 2))
print(f"[1] 6因子 → {len(ALL_PAIRS)}个交互对")

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 预计算
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m
del kline; import gc; gc.collect()

hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0: r = row.iloc[0]; hs300_m[d] = {'close':r['close'],'high_2y':r['high_2y']}

YEARS = sorted(set(d.year for d in monthly_dates)); TRAIN=5; FIRST=YEARS[0]+TRAIN
MCAP_FLOOR=0.20; LIMIT_UP=0.095; TOP_N=30; COST=0.0033

# === 实验1: 每对单独回测(近10年) ===
print(f"\n[2] 单对回测 (WF 2015-2026):")
print(f"{'交互对':<28s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>7s}")

pair_perf = {}
for fa, fb in ALL_PAIRS:
    results = []
    for yr in range(max(2015, FIRST), YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<100: continue
            for f in [fa,fb]:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score'] = day[fa+'_r'] * day[fb+'_r']
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]; day = day[day['fwd_ret'].notna()]
            if len(day)<80: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=15: results.append(top['fwd_ret'].mean()-COST)

    if results:
        arr = np.array(results); n=len(arr)
        cum=np.prod(1+arr); ann=cum**(12/n)-1
        c=np.cumprod(1+arr); mdd=np.min(c/np.maximum.accumulate(c)-1)
        vol=np.std(arr)*np.sqrt(12); sh=ann/vol if vol>0 else 0
        pair_perf[(fa,fb)] = {'ann':ann,'sh':sh,'mdd':mdd,'n':n}

for (fa,fb), r in sorted(pair_perf.items(), key=lambda x: x[1]['ann'], reverse=True):
    print(f"{fa[:6]:>6s}×{fb[:6]:<6s}       {r['ann']*100:+7.1f}% {r['sh']:+6.2f} {r['mdd']*100:+6.1f}%")

# === 实验2: Top N对等权组合 ===
print(f"\n[3] 多对组合 (按单对年化排序, Top N):")
sorted_pairs = sorted(pair_perf.items(), key=lambda x: x[1]['ann'], reverse=True)

for n_pairs in [1, 2, 3, 4, 6, 8, 15]:
    top_pairs = [p for p, _ in sorted_pairs[:n_pairs]]
    results = []
    for yr in range(max(2015, FIRST), YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<100: continue
            all_f = list(set([x for p in top_pairs for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0
            valid_pairs = 0
            for fa,fb in top_pairs:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r'] * day[fb+'_r']
                    valid_pairs += 1
            if valid_pairs==0: continue
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]; day = day[day['fwd_ret'].notna()]
            if len(day)<80: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=15: results.append(top['fwd_ret'].mean()-COST)

    if results:
        arr=np.array(results); n=len(arr)
        cum=np.prod(1+arr); ann=cum**(12/n)-1
        c=np.cumprod(1+arr); mdd=np.min(c/np.maximum.accumulate(c)-1)
        vol=np.std(arr)*np.sqrt(12); sh=ann/vol if vol>0 else 0
        print(f"  Top{n_pairs:>2d}对: 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.1f}%")

# === 实验3: 表现稳定性(分年夏普标准差) ===
print(f"\n[4] 配对稳定性 (夏普年化标准差, 越小越稳):")
for (fa,fb), r in sorted(pair_perf.items(), key=lambda x: x[1]['ann'], reverse=True)[:8]:
    # 分年夏普
    yr_sharpe = []
    for yr in range(max(2015, FIRST), YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        yr_r = []
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<100: continue
            for f in [fa,fb]:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score'] = day[fa+'_r'] * day[fb+'_r']
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]; day = day[day['fwd_ret'].notna()]
            if len(day)<80: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=15: yr_r.append(top['fwd_ret'].mean()-COST)
        if len(yr_r)>=6:
            arr=np.array(yr_r); a=np.mean(arr)*12; v=np.std(arr)*np.sqrt(12)
            yr_sharpe.append(a/v if v>0 else 0)
    if yr_sharpe:
        avg_sh = np.mean(yr_sharpe); std_sh = np.std(yr_sharpe)
        neg_yrs = sum(1 for s in yr_sharpe if s<0)
        print(f"  {fa[:6]:>6s}×{fb[:6]:<6s}  均Sh={avg_sh:+.2f}  std={std_sh:.2f}  负年={neg_yrs}/{len(yr_sharpe)}")

# 显示当前 vs 最优
print(f"\n[5] 当前4对 vs 最优组合:")
print(f"  当前: price×turnover, price×sr5, amihud×max, turnover×sr5")
for n in [4, 6, 8]:
    top_pairs = [p for p, _ in sorted_pairs[:n]]
    pair_names = ', '.join([f'{a[:4]}×{b[:4]}' for a,b in top_pairs])
    print(f"  最优{n}对: {pair_names}")

print(f"\n耗时: {time.time()-t0:.0f}s")
