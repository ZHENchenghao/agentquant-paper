# -*- coding: utf-8 -*-
"""
宏观预测 v2 · 特征工程+多模型集成
================================
升级: 40+特征 + 交互项 + 极端标记 + 多模型投票
目标: 3月前向HS300涨跌 → 推高63%准确率
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
t0=time.time()
HORIZON=3; TRAIN_YEARS=5

print("="*60); print("宏观预测 v2 · 多模型集成 · %dm前向"%HORIZON); print("="*60)

# ===== 加载 =====
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
hs300=con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date']); hs300=hs300.set_index('trade_date')['close']
macro=con.execute("SELECT trade_date,vix,usdcny,m1_growth,m2_growth,spx,nasdaq,gold FROM macro_indicators ORDER BY trade_date").df()
macro['trade_date']=pd.to_datetime(macro['trade_date']); macro=macro.set_index('trade_date')
nb=con.execute("SELECT trade_date,net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date']=pd.to_datetime(nb['trade_date']); nb=nb.set_index('trade_date')['net_flow']
mg=con.execute("SELECT trade_date,margin_balance FROM margin_trading ORDER BY trade_date").df()
mg['trade_date']=pd.to_datetime(mg['trade_date']); mg=mg.set_index('trade_date')['margin_balance']
con.close()

fred=pd.read_csv('D:/AgentQuant/our/cache/macro_fred.csv',parse_dates=['DATE']).set_index('DATE')
fred2=pd.read_csv('D:/AgentQuant/our/cache/macro_fred_v2.csv',parse_dates=['DATE']).set_index('DATE')
fred_all=fred.join(fred2,how='outer',lsuffix='',rsuffix='_v2')
# Remove duplicate cols from v2 (keep original)
for c in fred_all.columns:
    if c.endswith('_v2'):
        orig=c[:-3]
        if orig in fred_all.columns:
            fred_all[orig]=fred_all[orig].fillna(fred_all[c])
            fred_all= fred_all.drop(columns=[c])

dates=sorted(set(hs300.index)|set(macro.index)|set(fred_all.index))
md=[];
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

def align(s,mdates):
    s=s.sort_index().dropna(); r={}
    for d in mdates:
        v=s[s.index<=d]
        if len(v)>0: r[d]=v.iloc[-1]
    return pd.Series(r)

# ===== 特征工程 =====
print("[1] 特征工程...")
F={}

# 基础宏观(水平值)
base_cols=['WTI','VIX_fred','US10Y','DXY','T10Y2Y','T10Y3M','BAA10Y','NFCI','T5YIE','T10YIE']
for col in base_cols:
    if col in fred_all.columns: F[col]=align(fred_all[col],md)

# DuckDB
for col in ['vix','usdcny','m1_growth','m2_growth','spx','nasdaq','gold']:
    if col in macro.columns: F[col.upper()]=align(macro[col],md)
F['M1M2']=align(macro['m1_growth']-macro['m2_growth'],md)
F['Margin']=align(mg,md); F['Margin_chg']=align(mg.pct_change(3),md)
F['North']=align(nb.rolling(60).sum(),md)

# 市场面
F['HS_mom']=align(hs300.pct_change(20),md)
F['HS_vol']=align(hs300.pct_change().rolling(20).std()*np.sqrt(252),md)
F['HS_dd']=align(hs300/hs300.rolling(504).max()-1,md)
F['HS_rec']=align(hs300/hs300.rolling(252).min()-1,md)

# 衍生特征: 1/3/6月变化
for col in list(F.keys()):
    s=F[col]
    if len(s.dropna())>50:
        F[col+'_d1']=s.diff(1); F[col+'_d3']=s.diff(3); F[col+'_d6']=s.diff(6)
        # z-score (滚动12月)
        roll_mean=s.rolling(12).mean(); roll_std=s.rolling(12).std().replace(0,1)
        F[col+'_z']=(s-roll_mean)/roll_std
        # 极端值标记
        F[col+'_hi']=(F[col+'_z']>1.5).astype(float)
        F[col+'_lo']=(F[col+'_z']<-1.5).astype(float)

# 关键交互: 利差×美元, 利率×VIX
for pair in [('US10Y_d3','DXY_d3'),('T10Y2Y_d3','VIX_fred_d3'),('T10Y3M','BAA10Y')]:
    a,b=pair
    if a in F and b in F: F[a+'_X_'+b]=F[a].fillna(0)*F[b].fillna(0)

# 趋势标记: 连续N月同向
for s_name in ['US10Y_d3','DXY_d3','M1M2_d3','HS_mom_d3']:
    if s_name in F: F[s_name+'_3mtrend']=F[s_name].rolling(3).mean()

ft=pd.DataFrame(F).dropna(how='all')
# 按有效行过滤
ft=ft.loc[:,ft.isna().mean()<0.5]  # 删除缺失>50%的列
ft=ft.dropna()
print("特征: %d个 | 有效月: %d"%(len(ft.columns),len(ft)))

# ===== 目标 =====
target_name='fwd'
ft[target_name]=np.nan
for i in range(len(ft)-HORIZON):
    today=ft.index[i]; future=ft.index[i+HORIZON]
    h_f=hs300[hs300.index<=future].iloc[-1]; h_t=hs300[hs300.index<=today].iloc[-1]
    ft.loc[today,target_name]=1 if h_f/h_t>1 else 0
ft=ft.dropna(subset=[target_name])
print("带目标: %d月 | 涨:%.0f%% 跌:%.0f%%"%(len(ft),ft[target_name].mean()*100,(1-ft[target_name].mean())*100))

# ===== Walk-Forward =====
print("\n[2] Walk-Forward (多模型)...")
YEARS=sorted(set(d.year for d in ft.index)); FY=YEARS[0]+TRAIN_YEARS+1
feat_cols=[c for c in ft.columns if not c.startswith('fwd') and c!=target_name]
print("训练特征: %d | 测试年: %d-%d"%(len(feat_cols),FY,YEARS[-1]))

results={model_name:[] for model_name in ['LGBM','RF','LR','ENSEMBLE','BASELINE']}

for test_yr in range(FY,YEARS[-1]+1):
    t_end=pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=2)
    t_start=pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    ts=pd.Timestamp('%d-01-01'%test_yr); te=pd.Timestamp('%d-12-31'%test_yr)

    tr=ft[(ft.index>=t_start)&(ft.index<t_end)]
    te_df=ft[(ft.index>=ts)&(ft.index<=te)]
    if len(tr)<12 or len(te_df)<3: continue

    X_tr=tr[feat_cols].fillna(0).values; y_tr=tr[target_name].values
    X_te=te_df[feat_cols].fillna(0).values; y_te=te_df[target_name].values
    if len(np.unique(y_tr))<2: continue

    # LGBM
    lgbm=LGBMClassifier(n_estimators=100,num_leaves=7,max_depth=3,learning_rate=0.05,
        subsample=0.8,reg_alpha=1.0,reg_lambda=1.0,min_child_samples=5,verbose=-1,random_state=42)
    lgbm.fit(X_tr,y_tr); p_lgbm=lgbm.predict_proba(X_te)[:,1]

    # RF
    rf=RandomForestClassifier(n_estimators=100,max_depth=5,min_samples_leaf=5,random_state=42,n_jobs=-1)
    rf.fit(X_tr,y_tr); p_rf=rf.predict_proba(X_te)[:,1]

    # LR (正则化)
    lr=LogisticRegression(C=0.1,penalty='l2',solver='liblinear',random_state=42)
    lr.fit(X_tr,y_tr); p_lr=lr.predict_proba(X_te)[:,1]

    # Ensemble: 等权投票
    p_ens=(p_lgbm+p_rf+p_lr)/3

    for name,preds in [('LGBM',p_lgbm),('RF',p_rf),('LR',p_lr),('ENSEMBLE',p_ens)]:
        results[name].extend(list(zip(preds,y_te)))

    results['BASELINE'].extend([(0, y) for y in y_te])  # 总是预测涨

# ===== 评估 =====
print("\n"+"="*60)
print("多模型对比 · %dm前向预测" % HORIZON)
print("="*60)
print("%-12s %6s %8s %8s %8s %8s" % ('模型','月数','准确率','vs基线','高分准确','低分准确'))
print("-"*60)
for name in ['BASELINE','LR','RF','LGBM','ENSEMBLE']:
    data=results[name]; n=len(data)
    if n<10: continue
    pr=np.array([x[0] for x in data]); ac=np.array([x[1] for x in data])
    acc=np.mean((pr>0.5)==ac); base=np.mean(ac)
    top_m=pr>=np.percentile(pr,67); bot_m=pr<=np.percentile(pr,33)
    top_acc=np.mean(ac[top_m]); bot_acc=np.mean(ac[bot_m])
    vs_base=acc-base
    print("%-12s %6d %7.1f%% %+7.1f%% %7.1f%% %7.1f%%"%(name,n,acc*100,vs_base*100,top_acc*100,bot_acc*100))

# 最佳模型特征重要性
best_name=max([n for n in ['LGBM','RF','LR'] if len(results[n])>10],key=lambda n:np.mean((np.array([x[0] for x in results[n]])>0.5)==np.array([x[1] for x in results[n]])))
print("\n最佳模型: %s 特征重要性Top10:" % best_name)
if best_name=='LGBM':
    fi=dict(zip(feat_cols,lgbm.feature_importances_))
    for f,v in sorted(fi.items(),key=lambda x:x[1],reverse=True)[:10]:
        print("  %-30s %.4f" % (f[:30], v))

print("\n耗时: %.0fs"%(time.time()-t0))
