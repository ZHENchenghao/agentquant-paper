# -*- coding: utf-8 -*-
"""
ML动量版 · 趋势跟随专项 · Walk-Forward
=======================================
定位: 不与小众争反转, 专做趋势延续
因子: 动量类(非反转类) — 收益率/均线/突破/相对强度
门禁: DD_SMART v2 (同小众标准)
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10; PURGE = 1

print("="*60)
print("ML动量版 · 趋势跟随 · Walk-Forward")
print("="*60)

# ============ 加载 ============
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""SELECT ts_code, trade_date, open, high, low, close, vol,
    COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap
FROM kline_daily WHERE trade_date>='2000-01-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2000-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean(); hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# 月度
dates = sorted(kline['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print("月度: %d" % len(monthly_dates))

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

# ============ 动量因子计算 ============
print("[1] 计算动量因子...")

# 先算每日基础指标
kline = kline.sort_values(['ts_code','trade_date'])
for code in kline['ts_code'].unique()[:1]:  # 验证逻辑用
    pass

# 简化: 用pandas groupby批量算
kline['ret_1d'] = kline.groupby('ts_code')['close'].pct_change()
kline['ret_5d'] = kline.groupby('ts_code')['close'].pct_change(5)
kline['ret_20d'] = kline.groupby('ts_code')['close'].pct_change(20)
kline['ret_60d'] = kline.groupby('ts_code')['close'].pct_change(60)
kline['ma20'] = kline.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
kline['ma60'] = kline.groupby('ts_code')['close'].transform(lambda x: x.rolling(60).mean())
kline['ma120'] = kline.groupby('ts_code')['close'].transform(lambda x: x.rolling(120).mean())
kline['high_20'] = kline.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).max())
kline['low_20'] = kline.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).min())
kline['vol_ma20'] = kline.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())

# 动量因子(截面排名用, 算到月末再rank)
MOM_FEATS = [
    'ret_1d','ret_5d','ret_20d','ret_60d',           # 多周期动量
    'div_ma20','div_ma60','div_ma120',                # 均线偏离
    'ma_align',                                       # 均线多头排列
    'vol_ratio',                                      # 放量
    'near_high',                                      # 接近高点
    'streak_up',                                      # 连涨天数
]
kline['div_ma20'] = kline['close']/kline['ma20']-1
kline['div_ma60'] = kline['close']/kline['ma60']-1
kline['div_ma120'] = kline['close']/kline['ma120']-1
kline['ma_align'] = ((kline['close']>kline['ma20']).astype(int) +
                      (kline['ma20']>kline['ma60']).astype(int) +
                      (kline['ma60']>kline['ma120']).astype(int))/3
kline['vol_ratio'] = kline['vol']/kline['vol_ma20']
kline['near_high'] = kline['close']/kline['high_20']
kline['streak_up'] = kline.groupby('ts_code')['ret_1d'].transform(
    lambda x: (x>0).rolling(5).sum())

print("因子计算完成")

# ============ 月度数据集 ============
print("[2] 构建月度数据+目标...")

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap']+MOM_FEATS].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m
del kline; gc.collect()

monthly_rows = []
for rd in monthly_dates:
    if rd not in rd_map: continue
    px = rd_map[rd]
    px = px.dropna(subset=MOM_FEATS+['fwd_ret','mcap'])
    if len(px) < 200: continue
    px['trade_date'] = rd
    monthly_rows.append(px.reset_index())

full_df = pd.concat(monthly_rows)
print("月度: %d行 %d月"%(len(full_df), full_df['trade_date'].nunique()))

# 超额收益(用HS300作为基准)
for rd in full_df['trade_date'].unique():
    if rd not in hs300_m: continue
    next_mds = [d for d in monthly_dates if d > rd]
    if not next_mds or next_mds[0] not in hs300_m: continue
    nxt = next_mds[0]
    hs300_fwd = hs300_m[nxt]['close']/hs300_m[rd]['close']-1
    mask = full_df['trade_date']==rd
    full_df.loc[mask,'excess_ret'] = full_df.loc[mask,'fwd_ret'] - hs300_fwd

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1

print("\n[3] Walk-Forward...")
all_results = []; state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_end = pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=PURGE+1)
    train_start = pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01'%test_yr); test_end = pd.Timestamp('%d-12-31'%test_yr)

    tr = full_df[(full_df['trade_date']>=train_start)&(full_df['trade_date']<train_end)].copy()
    te = full_df[(full_df['trade_date']>=test_start)&(full_df['trade_date']<=test_end)].copy()
    if len(tr) < 5000 or len(te) < 500: continue

    # OLS中性化(mcap, 仅在训练集拟合)
    X_ols_tr = np.column_stack([np.log(tr['mcap'].clip(lower=1e6)).values])
    X_ols_te = np.column_stack([np.log(te['mcap'].clip(lower=1e6)).values])
    ols = LinearRegression()
    neu_feats = []
    for f in MOM_FEATS:
        if f not in tr.columns: continue
        y_tr = tr[f].fillna(0).values; y_te = te[f].fillna(0).values
        ols.fit(X_ols_tr, y_tr)
        tr[f+'_n'] = y_tr - ols.predict(X_ols_tr); te[f+'_n'] = y_te - ols.predict(X_ols_te)
        mu, std = tr[f+'_n'].mean(), tr[f+'_n'].std()
        if std > 0: tr[f+'_n'] = (tr[f+'_n']-mu)/std; te[f+'_n'] = (te[f+'_n']-mu)/std
        neu_feats.append(f+'_n')

    X_tr = tr[neu_feats].fillna(0).values.astype(float)
    y_tr = tr['excess_ret'].fillna(0).values
    X_te = te[neu_feats].fillna(0).values.astype(float)

    if len(X_tr) > 80000:
        idx = np.random.choice(len(X_tr), 80000, replace=False); X_tr, y_tr = X_tr[idx], y_tr[idx]

    split = int(len(X_tr)*0.8)
    model = LGBMRegressor(n_estimators=200, num_leaves=15, max_depth=4, learning_rate=0.02,
        subsample=0.7, reg_alpha=1.0, reg_lambda=1.0, min_child_samples=100, verbose=-1, n_jobs=-1)
    model.fit(X_tr[:split], y_tr[:split], eval_set=[(X_tr[split:], y_tr[split:])])
    best_iter = model.best_iteration_ if model.best_iteration_ else 100

    model2 = LGBMRegressor(n_estimators=best_iter, num_leaves=15, max_depth=4, learning_rate=0.02,
        subsample=0.7, reg_alpha=1.0, reg_lambda=1.0, min_child_samples=100, verbose=-1, n_jobs=-1)
    model2.fit(X_tr, y_tr)
    te['pred'] = model2.predict(X_te)

    # 选股: 市值过滤+涨停过滤
    te['mcap_r'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[(te['mcap_r']>=MCAP_FLOOR)&(te['ret_1d'].fillna(0)<LIMIT_UP)]

    # 只选趋势向上的(动量>0)
    te_f = te_f[te_f['ret_20d'].fillna(-1) > -0.05]

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
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%% iter=%d"%(test_yr,a*100,s,mdd*100,best_iter))

# 总结
r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
mdd=np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1); total=np.prod(1+r_all)-1
win=(r_all>0).mean()*100

print("\n"+"="*60)
print("ML动量版 · 终验")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | 累计: %+.0f%%"%(ann*100,sh,mdd*100,total*100))
print("胜率: %.0f%% | 月数: %d"%(win,len(r_all)))

# 分级行情表现
bull_yrs = [2007,2009,2014,2015,2019,2021,2025]
bear_yrs = [2008,2011,2017,2018,2022]
bl_r = [x['ret'] for x in all_results if x['yr'] in bull_yrs]
br_r = [x['ret'] for x in all_results if x['yr'] in bear_yrs]
print("牛市年: %+.1f%% | 熊市年: %+.1f%%"%(
    (np.prod(1+np.array(bl_r))-1)*100 if bl_r else 0,
    (np.prod(1+np.array(br_r))-1)*100 if br_r else 0))

print("\n年      收益   Sharpe    MDD")
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array(dr); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        print("%d %+7.1f%% %+6.2f %+6.1f%%"%(yr,(np.prod(1+rr)-1)*100,s,m*100))

print("\n=== 三策略定位 ===")
print("ETF轮动: MA200趋势 | ML动量: 追涨 | 小众: 捡漏")
print("牛市:  ETF+ML吃肉 | 熊市: 小众兜底")

print("\n耗时: %.0fs"%(time.time()-t0))
