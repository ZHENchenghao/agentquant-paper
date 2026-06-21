# -*- coding: utf-8 -*-
"""
宏观→市场方向预测 · Walk-Forward
===============================
目标: 预测HS300未来1/3/6个月涨跌方向
输入: 全套宏观变量(FRED+DuckDB)
方法: LightGBM + Walk-Forward(5年训1年测)
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMClassifier
t0=time.time()

print("="*60)
print("宏观→市场方向预测")
print("="*60)

# ===== 加载全部数据 =====
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
hs300=con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date']); hs300=hs300.set_index('trade_date')['close']

macro=con.execute("""SELECT trade_date,vix,usdcny,m1_growth,m2_growth,spx,nasdaq,gold,wti
    FROM macro_indicators ORDER BY trade_date""").df()
macro['trade_date']=pd.to_datetime(macro['trade_date']); macro=macro.set_index('trade_date')

nb=con.execute("SELECT trade_date,net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date']=pd.to_datetime(nb['trade_date']); nb=nb.set_index('trade_date')['net_flow']

mg=con.execute("SELECT trade_date,margin_balance FROM margin_trading ORDER BY trade_date").df()
mg['trade_date']=pd.to_datetime(mg['trade_date']); mg=mg.set_index('trade_date')['margin_balance']
con.close()

fred=pd.read_csv('D:/AgentQuant/our/cache/macro_fred.csv',parse_dates=['DATE']).set_index('DATE')

# ===== 构建月度特征集 =====
dates=sorted(set(hs300.index)|set(macro.index)|set(fred.index))
md=[];
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

def align(s,mdates):
    s=s.sort_index().dropna(); r={}
    for d in mdates:
        v=s[s.index<=d]
        if len(v)>0: r[d]=v.iloc[-1]
    return pd.Series(r)

# 构建特征: 水平值+1/3/6月变化+滚动z-score
features={}
# 基础宏观
for col in ['WTI','VIX_fred','US10Y','DXY','T10Y2Y']:
    if col in fred.columns:
        s=align(fred[col],md); features[col]=s
        features[col+'_1m_chg']=s.diff(1); features[col+'_3m_chg']=s.diff(3)
# DuckDB宏观
for col in ['vix','usdcny','m1_growth','m2_growth']:
    if col in macro.columns:
        s=align(macro[col],md); features[col.upper()]=s
features['M1M2']=align(macro['m1_growth']-macro['m2_growth'],md)
# 资金面
features['Margin']=align(mg,md); features['Margin_3m']=align(mg.pct_change(3),md)
features['North']=align(nb.rolling(60).sum(),md)
# 市场面
features['HS300_mom']=align(hs300.pct_change(20),md)
features['HS300_vol']=align(hs300.pct_change().rolling(20).std()*np.sqrt(252),md)
features['HS300_dd']=align(hs300/hs300.rolling(504).max()-1,md)
features['HS300_rec']=align(hs300/hs300.rolling(252).min()-1,md)

ft=pd.DataFrame(features).dropna()
print("特征: %d个, 有效月: %d (%s~%s)"%(len(ft.columns),len(ft),ft.index[0].date(),ft.index[-1].date()))

# ===== 目标: 未来N月方向 =====
for horizon in [1,3,6]:
    target_name='fwd_%dm'%horizon
    ft[target_name]=np.nan
    for i in range(len(ft)-horizon):
        today=ft.index[i]; future=ft.index[i+horizon]
        h_f=hs300[hs300.index<=future].iloc[-1]; h_t=hs300[hs300.index<=today].iloc[-1]
        ft.loc[today,target_name]=1 if h_f/h_t>1 else 0  # 1=涨,0=跌

    # Walk-Forward
    TRAIN_YEARS=3; YEARS=sorted(set(d.year for d in ft.index))
    FY=YEARS[0]+TRAIN_YEARS+1
    all_preds=[]; all_actuals=[]

    for test_yr in range(FY,YEARS[-1]+1):
        t_end=pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=2)
        t_start=pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
        ts=pd.Timestamp('%d-01-01'%test_yr); te=pd.Timestamp('%d-12-31'%test_yr)

        tr=ft[(ft.index>=t_start)&(ft.index<t_end)]
        te_df=ft[(ft.index>=ts)&(ft.index<=te)]
        tr_clean=tr.dropna(subset=[target_name])
        te_clean=te_df.dropna(subset=[target_name])
        if len(tr_clean)<6 or len(te_clean)<3: continue

        feat_cols=[c for c in ft.columns if not c.startswith('fwd_')]
        X_tr=tr_clean[feat_cols].fillna(0).values
        y_tr=tr_clean[target_name].values
        X_te=te_clean[feat_cols].fillna(0).values
        y_te=te_clean[target_name].values

        if X_tr.shape[1]==0 or len(np.unique(y_tr))<2: continue

        model=LGBMClassifier(n_estimators=50,num_leaves=7,max_depth=3,
            learning_rate=0.05,subsample=0.8,reg_alpha=0.5,reg_lambda=0.5,
            min_child_samples=10,verbose=-1,random_state=42)
        model.fit(X_tr,y_tr)
        preds=model.predict_proba(X_te)[:,1]
        all_preds.extend(preds); all_actuals.extend(y_te)

    if len(all_actuals)>20:
        pr=np.array(all_preds); ac=np.array(all_actuals)
        # 准确率
        acc=np.mean((pr>0.5)==ac)
        # 按预测分桶
        top_msk=pr>=np.percentile(pr,67); bot_msk=pr<=np.percentile(pr,33)
        top_acc=np.mean(ac[top_msk]); bot_acc=np.mean(ac[bot_msk])
        top_ret=top_acc*2-1  # 近似(涨比例-跌比例)
        bot_ret=bot_acc*2-1

        # 简单baseline: 总是预测涨
        base_acc=np.mean(ac)
        print("%dm前向: %d月 acc=%.1f%% (baseline:%.1f%%) top=%.1f%% bot=%.1f%% spread=%+.1f%% %s"%(
            horizon,len(ac),acc*100,base_acc*100,top_acc*100,bot_acc*100,
            (top_acc-bot_acc)*100,'SIG' if acc>max(0.55,base_acc+0.03) else 'NO'))

        # 特征重要性
        if horizon==3 and test_yr==YEARS[-1]:
            fi=dict(zip(feat_cols,model.feature_importances_))
            top_fi=sorted(fi.items(),key=lambda x:x[1],reverse=True)[:8]
            print("  Top特征: %s"%', '.join(['%s(%.2f)'%(f[:8],v) for f,v in top_fi]))

print("\n耗时: %.0fs"%(time.time()-t0))
