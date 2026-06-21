# -*- coding: utf-8 -*-
"""
导出DD_SMART逐月选股明细 (2007-2026)
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
FEATS = ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'), ('amihud','gap'), ('amihud','sr5'), ('amihud','vp_corr'),
    ('max_rev','gap'), ('max_rev','sr5'), ('max_rev','vp_corr'),
    ('gap','sr5'), ('gap','vp_corr'), ('sr5','vp_corr')
]

print("加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close, close/LAG(close) OVER(ORDER BY trade_date)-1 AS ret
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

print("构建价格映射...")
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[d] = {'close':r['close'], 'ma50':r['ma50'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[d] = {'close':r['close'], 'ma50':r['ma50'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y']}

def get_position_dd_smart(cur_date, state):
    if cur_date not in hs300_m: return 1.0, state
    info = hs300_m[cur_date]
    close = info['close']; ma50 = info['ma50']
    high_2y = info['high_2y']; low_1y = info['low_1y']
    if pd.isna(high_2y) or pd.isna(ma50): return 1.0, state

    if state['in_market']:
        dd_2y = close/high_2y - 1
        if dd_2y < -0.20: return 0.2, {'in_market':False,'exit_date':cur_date}
        elif dd_2y < -0.15: return 0.4, {'in_market':False,'exit_date':cur_date}
        else: return 1.0, state
    else:
        recovery = close/low_1y - 1 if pd.notna(low_1y) and low_1y>0 else 0
        above_ma50 = close > ma50
        if recovery > 0.15 and above_ma50: return 0.7, {'in_market':True,'exit_date':None}
        elif recovery > 0.10: return 0.4, state
        elif recovery > 0.05 and above_ma50: return 0.3, state
        else: return 0.15, state

YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

# 训练选对
print("训练选对...")
fold_pairs = {}
for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_mds = [d for d in monthly_dates if test_yr-TRAIN_YEARS <= d.year < test_yr]
    if len(train_mds) < 24: continue
    pair_ir = {}
    for (fa,fb) in ALL_PAIRS:
        spreads = []
        for rd in train_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue
            for f in [fa,fb]:
                if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)
            if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns: continue
            day['score'] = day[f'{fa}_r']*day[f'{fb}_r']
            day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
            vd = day.dropna(subset=['score','fwd_ret'])
            if len(vd) < 50: continue
            nq = int(len(vd)*0.2)
            spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
        if len(spreads)>=12:
            mu=np.mean(spreads); std=np.std(spreads)
            pair_ir[(fa,fb)]=mu/std if std>0 else 0
    sorted_pairs = sorted(pair_ir.items(), key=lambda x:x[1], reverse=True)
    fold_pairs[test_yr] = [p for p,ir in sorted_pairs[:4]]

# DD_SMART OOS + 记录选股
print("DD_SMART OOS + 记录选股...")
all_results = []; stock_picks = []
state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    if test_yr not in fold_pairs: continue
    top4 = fold_pairs[test_yr]
    test_mds = [d for d in monthly_dates if d.year==test_yr]
    if len(test_mds) < 3: continue

    for rd in test_mds:
        if rd not in rd_map: continue
        pos, state = get_position_dd_smart(rd, state)

        if pos < 0.01:
            all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos})
            continue

        day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
        valid = set(px.index); day = day[day['ts_code'].isin(valid)]
        if len(day) < 100: continue

        all_f = list(set([x for p in top4 for x in p]))
        for f in all_f:
            if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

        day['score'] = 0; ok = True
        for fa,fb in top4:
            if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                day['score'] += day[f'{fa}_r']*day[f'{fb}_r']
            else: ok = False
        if not ok: continue

        px_match = px.loc[day['ts_code'].values]
        day['mcap'] = px_match['mcap'].values
        day['ret_1d'] = px_match['ret_1d'].values
        day['fwd_ret'] = px_match['fwd_ret'].values
        day['mcap_r'] = day['mcap'].rank(pct=True)
        day = day[day['mcap_r']>=MCAP_FLOOR]
        day = day[day['ret_1d']<LIMIT_UP]
        day = day[day['fwd_ret'].notna()]
        if len(day) < 50: continue

        top = day.nlargest(TOP_N,'score')
        if len(top) < 5: continue

        month_ret = (top['fwd_ret'].mean() - COST) * pos
        all_results.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                           'yr':rd.year,'pos':pos,'mcap_med':top['mcap'].median()})

        # 记录每只选股
        for _, row in top.iterrows():
            stock_picks.append({
                'date': str(rd)[:7],
                'yr': rd.year,
                'ts_code': row['ts_code'],
                'score': row['score'],
                'fwd_ret': row['fwd_ret'],
                'position': pos,
                'mcap': row['mcap'],
                'amihud': row.get('amihud', np.nan),
                'max_rev': row.get('max_rev', np.nan),
                'gap': row.get('gap', np.nan),
                'sr5': row.get('sr5', np.nan),
                'vp_corr': row.get('vp_corr', np.nan),
                'pairs': '|'.join([f'{a[:4]}x{b[:4]}' for a,b in top4])
            })

# 汇总
r_all = np.array([x['ret'] for x in all_results])
ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
sh = ann/vol if vol>0 else 0
cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
win = (r_all>0).mean()*100

print(f"\nDD_SMART: 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% Win{win:.0f}%")

# 导出
df_picks = pd.DataFrame(stock_picks)
df_monthly = pd.DataFrame(all_results)

# 按月汇总
monthly_summary = df_picks.groupby('date').agg(
    n_stocks=('ts_code','count'),
    avg_fwd_ret=('fwd_ret','mean'),
    med_mcap=('mcap','median'),
    top3_stocks=('ts_code', lambda x: ' '.join(x.head(3))),
    avg_position=('position','first'),
    pairs=('pairs','first')
).reset_index()

OUT_PICKS = 'D:/AgentQuant/our/cache/dd_smart_picks.csv'
OUT_MONTHLY = 'D:/AgentQuant/our/cache/dd_smart_monthly.csv'

df_picks.to_csv(OUT_PICKS, index=False)
monthly_summary.to_csv(OUT_MONTHLY, index=False)

print(f"\n导出完成:")
print(f"  选股明细: {OUT_PICKS} ({len(df_picks)}条)")
print(f"  月度汇总: {OUT_MONTHLY} ({len(monthly_summary)}行)")

# 关键年选股样本
print(f"\n--- 关键年选股样本 ---")
for yr in [2008, 2009, 2015, 2018, 2021, 2024]:
    yr_picks = df_picks[df_picks['yr']==yr]
    if len(yr_picks)==0: continue
    print(f"\n{yr}年 (共{len(yr_picks)}只次):")
    # 该年出现最频繁的股票
    top_stocks = yr_picks['ts_code'].value_counts().head(10)
    for code, cnt in top_stocks.items():
        avg_ret = yr_picks[yr_picks['ts_code']==code]['fwd_ret'].mean()
        print(f"  {code}: {cnt}次选中, 平均收益{avg_ret*100:+.1f}%")

# 最差月度
print(f"\n--- 最差10个月 (纯策略收益, 未乘仓位) ---")
df_monthly_raw = df_monthly.copy()
df_monthly_raw['raw_ret'] = [x['ret']/(x['pos'] if x['pos']>0 else 1) for _,x in df_monthly_raw.iterrows()]
worst = df_monthly_raw.nsmallest(10, 'raw_ret')
for _, row in worst.iterrows():
    print(f"  {row['date']}: raw={row['raw_ret']*100:+.1f}% net={row['ret']*100:+.1f}% pos={row['pos']*100:.0f}%")

# 最佳月度
print(f"\n--- 最佳10个月 ---")
best = df_monthly_raw.nlargest(10, 'raw_ret')
for _, row in best.iterrows():
    print(f"  {row['date']}: raw={row['raw_ret']*100:+.1f}% net={row['ret']*100:+.1f}% pos={row['pos']*100:.0f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
