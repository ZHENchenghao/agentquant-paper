# -*- coding: utf-8 -*-
"""
ML选股 v2 · 行为因子+LightGBM · 严格Walk-Forward
=================================================
改进:
  1. 因子: 小众6因子(非12 TA因子, 已FM验证)
  2. 目标: 月频超额收益(个股-沪深300)
  3. 双目标: 回归+排名(LightGBM原始设计)
  4. OLS中性化: mcap+行业, 仅在训练集拟合
  5. Purge Walk-Forward: 5年训→1年测, +1月隔离
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
PURGE_MONTHS = 1
FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']

print("="*60); print("ML v2 · 行为因子+LightGBM · Walk-Forward"); print("="*60)

# ============ 加载 ============
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
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
# DD_SMART v2 params
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10

industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()
con.close()

# 月度调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print("月度: %d" % len(monthly_dates))

# ============ 合并月度数据 ============
print("[2] 合并因子+价格+目标...")
# 构建月度持有的前向收益
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# HS300月度收益+DD_SMART信号
hs300_m = {}
for rd in monthly_dates:
    row = hs300[hs300['trade_date']==rd]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[rd] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=rd]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[rd] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart_v2(rd, state):
    if rd not in hs300_m: return 1.0, state
    info = hs300_m[rd]; c=info['close']; ma50=info['ma50']
    h2y=info['high_2y']; l1y=info['low_1y']
    if pd.isna(h2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y = c/h2y-1
        if dd_2y < EXIT_THRESH-0.05: return FLOOR, {'in_market':False,'exit_date':rd}
        elif dd_2y < EXIT_THRESH: return FLOOR*2, {'in_market':False,'exit_date':rd}
        else: return 1.0, state
    else:
        rec = c/l1y-1 if pd.notna(l1y) and l1y>0 else 0
        above = c > ma50
        if rec > REENTRY_THRESH and above: return 0.7, {'in_market':True,'exit_date':None}
        elif rec > REENTRY_THRESH*0.7: return FLOOR*2, state
        elif rec > 0.05 and above: return FLOOR, state
        else: return FLOOR, state

# 逐月建数据集
monthly_rows = []
for rd in monthly_dates:
    if rd not in rd_map: continue
    day = fn[fn['trade_date']==rd].copy()
    px = rd_map[rd]
    valid = set(px.index); day = day[day['ts_code'].isin(valid)]
    if len(day) < 200: continue

    # 合并: 因子 + 价格 + 行业
    day = day.merge(px[['mcap','ret_1d','fwd_ret']], left_on='ts_code', right_index=True, how='inner')
    day = day.dropna(subset=FEATS+['fwd_ret','mcap'])
    if len(day) < 200: continue

    # 超额收益
    if rd in hs300_m and rd in rd_map:
        next_rd = monthly_dates[monthly_dates.index(rd)+1] if monthly_dates.index(rd)+1 < len(monthly_dates) else None
        if next_rd and next_rd in hs300_m:
            hs300_fwd = hs300_m[next_rd]['close'] / hs300_m[rd]['close'] - 1
            day['excess_ret'] = day['fwd_ret'] - hs300_fwd

    day['trade_date'] = rd
    monthly_rows.append(day[['ts_code','trade_date']+FEATS+['mcap','ret_1d','fwd_ret','excess_ret']])

full_df = pd.concat(monthly_rows).merge(industry, on='ts_code', how='left')
full_df['ind_name'] = full_df['ind_name'].fillna('Other')
print("合并: %d行 %d个月" % (len(full_df), full_df['trade_date'].nunique()))

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1

print("\n[3] Walk-Forward (%d-%d)..." % (FIRST_TEST_YR, YEARS[-1]))

all_results = []
state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_end = pd.Timestamp('%d-01-01' % test_yr) - pd.DateOffset(months=PURGE_MONTHS+1)
    train_start = pd.Timestamp('%d-01-01' % (test_yr - TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01' % test_yr)
    test_end = pd.Timestamp('%d-12-31' % test_yr)

    tr = full_df[(full_df['trade_date'] >= train_start) & (full_df['trade_date'] < train_end)].copy()
    te = full_df[(full_df['trade_date'] >= test_start) & (full_df['trade_date'] <= test_end)].copy()
    if len(tr) < 5000 or len(te) < 500: continue

    # OLS中性化 (mcap + 行业, 仅训练集)
    all_inds = sorted(set(tr['ind_name'].unique()) | set(te['ind_name'].unique()))
    ind_map = {ind: i for i, ind in enumerate(all_inds)}
    tr_dum = np.zeros((len(tr), len(all_inds)))
    te_dum = np.zeros((len(te), len(all_inds)))
    for i, ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i, ind_map[ind]] = 1
    for i, ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i, ind_map[ind]] = 1

    X_ols_tr = np.column_stack([np.log(tr['mcap'].clip(lower=1e6)).values, tr_dum])
    X_ols_te = np.column_stack([np.log(te['mcap'].clip(lower=1e6)).values, te_dum])
    ols = LinearRegression(fit_intercept=False)

    neu_feats = []
    for f in FEATS:
        if f not in tr.columns: continue
        y_tr = tr[f].fillna(0).values; y_te = te[f].fillna(0).values
        ols.fit(X_ols_tr, y_tr)
        tr[f+'_n'] = y_tr - ols.predict(X_ols_tr)
        te[f+'_n'] = y_te - ols.predict(X_ols_te)
        mu, std = tr[f+'_n'].mean(), tr[f+'_n'].std()
        if std > 0: tr[f+'_n'] = (tr[f+'_n']-mu)/std; te[f+'_n'] = (te[f+'_n']-mu)/std
        neu_feats.append(f+'_n')

    # LightGBM (双目标)
    X_tr = tr[neu_feats].fillna(0).values.astype(float)
    y1 = tr['excess_ret'].fillna(0).values  # 超额收益
    y2 = tr.groupby('trade_date')['excess_ret'].rank(pct=True).fillna(0.5).values  # 截面排名
    X_te = te[neu_feats].fillna(0).values.astype(float)

    if len(X_tr) > 100000:
        idx = np.random.choice(len(X_tr), 100000, replace=False)
        X_tr, y1, y2 = X_tr[idx], y1[idx], y2[idx]

    split = int(len(X_tr)*0.8)
    m1 = LGBMRegressor(n_estimators=200,num_leaves=15,max_depth=4,learning_rate=0.02,
        subsample=0.7,reg_alpha=0.5,reg_lambda=0.5,min_child_samples=100,verbose=-1,n_jobs=-1)
    m1.fit(X_tr[:split], y1[:split], eval_set=[(X_tr[split:], y1[split:])],
           callbacks=[lambda cb: cb.early_stopping(20, verbose=False)] if hasattr(LGBMRegressor,'early_stopping') else None)

    m2 = LGBMRegressor(n_estimators=200,num_leaves=15,max_depth=4,learning_rate=0.02,
        subsample=0.7,reg_alpha=0.5,reg_lambda=0.5,min_child_samples=100,verbose=-1,n_jobs=-1)
    m2.fit(X_tr[:split], y2[:split], eval_set=[(X_tr[split:], y2[split:])])

    # 预测: 回归+排名模型平均
    te['pred'] = m1.predict(X_te) * 0.5 + m2.predict(X_te) * 0.5

    # 选股
    te['mcap_r'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[(te['mcap_r'] >= MCAP_FLOOR) & (te['ret_1d'].fillna(0) < LIMIT_UP)]

    fold_rets = []
    for rd in sorted(te_f['trade_date'].unique()):
        pos, state = dd_smart_v2(rd, state)
        if pos < 0.01:
            fold_rets.append(0.0); all_results.append({'date':str(rd)[:7],'ret':0.0,'yr':rd.year,'n':0})
            continue

        day_te = te_f[te_f['trade_date'] == rd]
        if len(day_te) < 50: continue
        top = day_te.nlargest(TOP_N, 'pred')
        if len(top) < 5: continue
        month_ret = (top['fwd_ret'].mean() - COST) * pos
        fold_rets.append(month_ret)
        all_results.append({'date':str(rd)[:7],'ret':month_ret,'yr':rd.year,'n':len(top),'pos':pos})

    if fold_rets:
        r=np.array(fold_rets); a=np.mean(r)*12; v=np.std(r)*np.sqrt(12)
        s=a/v if v>0 else 0; mdd=np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%%" % (test_yr,a*100,s,mdd*100))

# 总结
r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
mdd=np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1)
win=(r_all>0).mean()*100; calmar=ann/abs(mdd) if mdd!=0 else 0

print("\n"+"="*60)
print("ML v2 · 终验 (行为因子+超收+OLS+双模型+DD_SMART v2)")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | Calmar: %+.2f"%(ann*100,sh,mdd*100,calmar))
print("胜率: %.0f%% | 累计: %+.0f%% | 月数: %d"%(win,(np.prod(1+r_all)-1)*100,len(r_all)))

# 同台对比
crash=[2008,2011,2017,2018,2022]; bull=[2007,2009,2015,2019,2021,2025]
cr_ml=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bl_ml=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1
print("5熊: %+.1f%% | 6牛: %+.1f%%"%(cr_ml*100,bl_ml*100))

print("\n年      收益   Sharpe    MDD")
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array(dr); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        print("%d %+7.1f%% %+6.2f %+6.1f%%"%(yr,(np.prod(1+rr)-1)*100,s,m*100))

print("\n=== 与小众战法对比 ===")
print("%s %12s %12s" % ("指标","ML v2","小众 v2"))
print("年化: %+10.1f%% %+10.1f%%"%(ann*100,14.8))
print("Sharpe: %+10.2f %+10.2f"%(sh,0.70))
print("MDD: %+10.1f%% %+10.1f%%"%(mdd*100,31.9))
print("累计: %+10.0f%% %+10.0f%%"%((np.prod(1+r_all)-1)*100,1078))

print("\n耗时: %.0fs"%(time.time()-t0))
