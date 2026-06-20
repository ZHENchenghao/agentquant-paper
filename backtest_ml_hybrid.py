# -*- coding: utf-8 -*-
"""
ML混合版 · LightGBM学习4对权重 · Walk-Forward
==============================================
思路: 不替代小众, 在小众框架内微调
  - 固定4对交互: Amihud×Turnover, Amihud×MaxRev, Amihud×SR5, Turnover×SR5
  - 每月计算4对得分(截面排名乘法)
  - LightGBM输入: 4对得分 → 输出: 未来收益预测
  - DD_SMART v2 门禁保护

优势: 维度4(非6非12), 输入是已验证的交互对, ML只做权重微调
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
PURGE_MONTHS = 1
# 固定4对
PAIRS = [('amihud','turnover_rev'),('amihud','max_rev'),('amihud','sr5'),('turnover_rev','sr5')]

print("="*60)
print("ML混合版 · 4对权重学习 · Walk-Forward")
print("="*60)

# 加载
print("[1] 加载...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""SELECT ts_code, trade_date, open, close,
    COALESCE(close*total_share/10000, GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2002-01-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2002-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean(); hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# 月度
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 价格映射
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m
del kline; gc.collect()

# HS300信号
hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0: r=row.iloc[0]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby=hs300[hs300['trade_date']<=d]
        if len(nearby)>0: r=nearby.iloc[-1]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart_v2(rd, state):
    if rd not in hs300_m: return 1.0, state
    info=hs300_m[rd]; c=info['close']; ma50=info['ma50']; h2y=info['high_2y']; l1y=info['low_1y']
    if pd.isna(h2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y=c/h2y-1
        if dd_2y<EXIT_THRESH-0.05: return FLOOR,{'in_market':False,'exit_date':rd}
        elif dd_2y<EXIT_THRESH: return FLOOR*2,{'in_market':False,'exit_date':rd}
        else: return 1.0, state
    else:
        rec=c/l1y-1 if pd.notna(l1y) and l1y>0 else 0; above=c>ma50
        if rec>REENTRY_THRESH and above: return 0.7,{'in_market':True,'exit_date':None}
        elif rec>REENTRY_THRESH*0.7: return FLOOR*2,state
        elif rec>0.05 and above: return FLOOR,state
        else: return FLOOR,state

# ============ 构建月度数据集(4对得分) ============
print("[2] 构建4对得分+目标...")
PAIR_NAMES = ['%s_x_%s'%(a[:4],b[:4]) for a,b in PAIRS]

monthly_rows = []
for rd in monthly_dates:
    if rd not in rd_map: continue
    day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
    valid = set(px.index); day = day[day['ts_code'].isin(valid)]
    if len(day) < 200: continue

    # 因子排名
    all_f = list(set([x for p in PAIRS for x in p]))
    for f in all_f:
        if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)

    # 4对得分
    for (fa,fb), pn in zip(PAIRS, PAIR_NAMES):
        if fa+'_r' in day.columns and fb+'_r' in day.columns:
            day[pn] = day[fa+'_r'] * day[fb+'_r']

    px_match = px.loc[day['ts_code'].values]
    day['mcap'] = px_match['mcap'].values; day['ret_1d'] = px_match['ret_1d'].values
    day['fwd_ret'] = px_match['fwd_ret'].values; day['trade_date'] = rd

    day = day.dropna(subset=PAIR_NAMES+['fwd_ret','mcap'])
    if len(day) < 200: continue
    monthly_rows.append(day[['ts_code','trade_date','mcap','ret_1d','fwd_ret']+PAIR_NAMES])

full_df = pd.concat(monthly_rows)
print("月度数据: %d行 %d月"%(len(full_df), full_df['trade_date'].nunique()))

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1

print("\n[3] Walk-Forward...")
all_results = []; state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_end = pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=PURGE_MONTHS+1)
    train_start = pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01'%test_yr); test_end = pd.Timestamp('%d-12-31'%test_yr)

    tr = full_df[(full_df['trade_date']>=train_start)&(full_df['trade_date']<train_end)].copy()
    te = full_df[(full_df['trade_date']>=test_start)&(full_df['trade_date']<=test_end)].copy()
    if len(tr) < 5000 or len(te) < 500: continue

    # LightGBM: 4对得分 → fwd_ret
    X_tr = tr[PAIR_NAMES].fillna(0).values.astype(float)
    y_tr = tr['fwd_ret'].fillna(0).values
    X_te = te[PAIR_NAMES].fillna(0).values.astype(float)

    if len(X_tr) > 100000:
        idx = np.random.choice(len(X_tr), 100000, replace=False); X_tr, y_tr = X_tr[idx], y_tr[idx]

    split = int(len(X_tr)*0.8)
    model = LGBMRegressor(n_estimators=100, num_leaves=7, max_depth=3, learning_rate=0.03,
        subsample=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_samples=200, verbose=-1, n_jobs=-1)
    model.fit(X_tr[:split], y_tr[:split], eval_set=[(X_tr[split:], y_tr[split:])])

    te['pred'] = model.predict(X_te)

    # 选股
    te['mcap_r'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[(te['mcap_r']>=MCAP_FLOOR)&(te['ret_1d'].fillna(0)<LIMIT_UP)]

    fold_rets = []
    for rd in sorted(te_f['trade_date'].unique()):
        pos, state = dd_smart_v2(rd, state)
        if pos < 0.01: fold_rets.append(0.0); all_results.append({'date':str(rd)[:7],'ret':0.0,'yr':rd.year}); continue
        day_te = te_f[te_f['trade_date']==rd]
        if len(day_te) < 50: continue
        top = day_te.nlargest(TOP_N, 'pred')
        if len(top) < 5: continue
        mr = (top['fwd_ret'].mean()-COST)*pos
        fold_rets.append(mr); all_results.append({'date':str(rd)[:7],'ret':mr,'yr':rd.year,'n':len(top)})

    if fold_rets:
        r=np.array(fold_rets); a=np.mean(r)*12; v=np.std(r)*np.sqrt(12)
        s=a/v if v>0 else 0; mdd=np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%%"%(test_yr,a*100,s,mdd*100))

# 总结
r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
mdd=np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1)
win=(r_all>0).mean()*100; calmar=ann/abs(mdd) if mdd!=0 else 0; total=np.prod(1+r_all)-1

print("\n"+"="*60)
print("ML混合版 · 终验")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | Calmar: %+.2f"%(ann*100,sh,mdd*100,calmar))
print("胜率: %.0f%% | 累计: %+.0f%%"%(win,total*100))

crash=[2008,2011,2017,2018,2022]; bull=[2007,2009,2015,2019,2021,2025]
cr=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bl=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1
print("5熊: %+.1f%% | 6牛: %+.1f%%"%(cr*100,bl*100))

print("\n年      收益   Sharpe    MDD")
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array(dr); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        print("%d %+7.1f%% %+6.2f %+6.1f%%"%(yr,(np.prod(1+rr)-1)*100,s,m*100))

print("\n=== 终极对比 ===")
versions = {
    'ML混合(4对权重)': (ann*100, sh, mdd*100, total*100),
    '小众乘法(等权)': (14.8, 0.70, -31.9, 1078),
    'ML全因子(6因子)': (7.3, 0.45, -52.1, 206),
}
print("%s %8s %8s %8s %8s"%('版本','年化','Sharpe','MDD','累计'))
for name,(a,s,m,t) in versions.items():
    print("%s %+7.1f%% %+7.2f %+7.1f%% %+7.0f%%"%(name,a,s,m,t))

print("\n耗时: %.0fs"%(time.time()-t0))
