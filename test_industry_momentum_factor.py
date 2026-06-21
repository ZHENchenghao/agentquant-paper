# -*- coding: utf-8 -*-
"""小众+行业动量因子 · 快速验证
新因子: ind_mom = 股票所属行业的近1月收益排名
测试: ①单因子IC ②加入配对 ③vs基准
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

# 行业映射
ind_map = con.execute("SELECT ts_code, ind_name FROM stock_industry").df()
ind_map = ind_map.rename(columns={'ind_name': 'industry'})

# K线+行业
kline = con.execute("""SELECT k.ts_code, k.trade_date, k.open, k.close,
    COALESCE(k.close*k.total_share/10000, k.close*k.vol/1000000) AS mcap,
    k.close/LAG(k.close) OVER(PARTITION BY k.ts_code ORDER BY k.trade_date)-1 AS ret_1d
FROM kline_daily k WHERE k.trade_date>='2005-01-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
kline = kline.merge(ind_map, on='ts_code', how='left')

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300['high_2y'] = hs300['close'].rolling(504).max()
con.close()

# 行业月收益
kline['month'] = kline['trade_date'].dt.to_period('M')
ind_monthly = kline.groupby(['industry', 'month'])['close'].last().reset_index()
ind_monthly['month'] = ind_monthly['month'].dt.to_timestamp()
ind_monthly['ind_ret_1m'] = ind_monthly.groupby('industry')['close'].pct_change()

# 合并到因子表
fn['month'] = fn['trade_date'].dt.to_period('M')
fn['month'] = fn['month'].dt.to_timestamp()
fn = fn.merge(ind_map, on='ts_code', how='left')
fn = fn.merge(ind_monthly[['industry', 'month', 'ind_ret_1m']], on=['industry', 'month'], how='left')

# 行业动量排名(每月横截面)
fn['ind_mom'] = fn.groupby('month')['ind_ret_1m'].rank(pct=True)
fn = fn.dropna(subset=['ind_mom'])
print(f'[1] 因子+行业动量: {len(fn)}行, 行业动量有效{fn["ind_mom"].notna().sum()}行')

# 回测框架
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

hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0: r = row.iloc[0]; hs300_m[d] = {'close':r['close'],'high_2y':r['high_2y']}

YEARS = sorted(set(d.year for d in monthly_dates)); TRAIN=3; FIRST=YEARS[0]+TRAIN
MCAP_FLOOR=0.20; LIMIT_UP=0.095; TOP_N=30; COST=0.0033
EXIT_THRESH=-0.12; REENTRY_THRESH=0.10; FLOOR=0.10

def gate(cur_date, state):
    if cur_date not in hs300_m: return 1.0, state
    info = hs300_m[cur_date]; close = info['close']; h2y = info['high_2y']
    if pd.isna(h2y): return 1.0, state
    if state['in']:
        dd = close/h2y-1
        if dd < EXIT_THRESH-0.05: return FLOOR, {'in':False}
        elif dd < EXIT_THRESH: return FLOOR*2, {'in':False}
        else: return 1.0, state
    else:
        recovery = close/min(h2y*0.5, close) if h2y>0 else 0
        if recovery > REENTRY_THRESH: return 0.7, {'in':True}
        elif recovery > REENTRY_THRESH*0.7: return FLOOR*2, state
        else: return FLOOR, state

FEATS_OLD = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
FEATS_NEW = FEATS_OLD + ['ind_mom']

# === 实验: ind_mom作为独立因子加入pair池 ===
from itertools import combinations
NEW_PAIRS = list(combinations(FEATS_NEW, 2))  # 21对

print(f'\n[2] 对比: 15对(old) vs 21对(+ind_mom)')

# 每个pair单独回测(2008-2026, 快速版只测近10年)
FAST_YEARS = range(max(2015, FIRST), YEARS[-1]+1)

pair_perf_new = {}
for fa, fb in NEW_PAIRS:
    results = []
    for yr in FAST_YEARS:
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<50: continue
            for f in [fa,fb]:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score'] = day[fa+'_r'] * day[fb+'_r']
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day)<50: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=10: results.append(top['fwd_ret'].mean()-COST)
    if results:
        arr=np.array(results); n=len(arr)
        cum=np.prod(1+arr); ann=cum**(12/n)-1
        c=np.cumprod(1+arr); mdd=np.min(c/np.maximum.accumulate(c)-1)
        vol=np.std(arr)*np.sqrt(12); sh=ann/vol if vol>0 else 0
        pair_perf_new[(fa,fb)] = {'ann':ann,'sh':sh,'mdd':mdd}

# 输出含ind_mom的新pair
print('\n含行业动量的新pair (按年化排序):')
print('排名   因子对                              年化   Sharpe     MDD')
for rank, ((fa,fb), r) in enumerate(sorted(pair_perf_new.items(), key=lambda x: x[1]['ann'], reverse=True)):
    if 'ind_mom' in (fa, fb):
        ann_val = r['ann']*100; sh_val = r['sh']; mdd_val = r['mdd']*100
        print(f'{rank+1:<4d} {fa[:10]:>10s} x {fb:<10s} {ann_val:+7.1f}% {sh_val:+6.2f} {mdd_val:+6.1f}% <-- NEW')

# === 策略对比: 最优4对 old vs 最优4对 new (WF) ===
print(f'\n[3] 完整WF策略对比 (TRAIN={TRAIN}年):')
sorted_old = sorted([(p,r) for p,r in pair_perf_new.items() if 'ind_mom' not in p], key=lambda x: x[1]['ann'], reverse=True)
sorted_all = sorted(pair_perf_new.items(), key=lambda x: x[1]['ann'], reverse=True)

for label, pairs_ranked in [('old最优4对', sorted_old), ('new最优4对', sorted_all)]:
    top4 = [p for p,_ in pairs_ranked[:4]]
    results = []; state = {'in':True}
    for yr in range(FIRST, YEARS[-1]+1):
        test_mds = [d for d in monthly_dates if d.year==yr]
        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = gate(rd, state)
            if pos<0.01: results.append(0.0); continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<50: continue
            all_f = list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0
            for fa,fb in top4:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r'] * day[fb+'_r']
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]; day = day[day['fwd_ret'].notna()]
            if len(day)<50: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=10: results.append((top['fwd_ret'].mean()-COST)*pos)

    r_all = np.array(results)
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    label_str = label
    print(f'  {label_str}: 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.1f}%')

print(f'\n耗时: {time.time()-t0:.0f}s')
