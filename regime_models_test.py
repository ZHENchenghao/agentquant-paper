# -*- coding: utf-8 -*-
import sys,io;sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from lightgbm import LGBMRegressor
from scipy import stats

data=pd.read_parquet('cache/factors_all.parquet')
c=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

target=c.execute("""SELECT s.ts_code,s.trade_date,(s.fc/s.close-1)-(x.fc/x.close-1) excess_ret
FROM (SELECT ts_code,trade_date,close,LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) fc
      FROM kline_daily WHERE trade_date BETWEEN '2015-01-01' AND '2026-06-16') s
JOIN (SELECT trade_date,close,LEAD(close,20) OVER(ORDER BY trade_date) fc
      FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date
WHERE s.fc IS NOT NULL""").df()

vix=c.execute("""SELECT trade_date,vix,vix-LAG(vix,5) OVER w vel5,
       (margin_balance/NULLIF(LAG(margin_balance,20) OVER w,0)-1)*100 mg20,
       SUM(net_flow) OVER w2 north_20d,
       close/NULLIF(MAX(close) OVER w3,0)-1 dd
FROM macro_indicators m LEFT JOIN margin_trading mt USING(trade_date)
LEFT JOIN north_bound_flow nb USING(trade_date)
LEFT JOIN kline_daily k ON m.trade_date=k.trade_date AND k.ts_code='sh000300'
WHERE m.vix IS NOT NULL
WINDOW w AS (ORDER BY m.trade_date),w2 AS (ORDER BY m.trade_date ROWS 19 PRECEDING),w3 AS (ORDER BY k.trade_date ROWS 249 PRECEDING)""").df()
c.close()

def fp(row):
    v=row['vix'];vel5=row.get('vel5',0)or 0;mg=row.get('mg20',0)or 0
    nf=row.get('north_20d',0)or 0;dd=row.get('dd',0)or 0
    if pd.isna(v):return 2
    b=5 if v>35 else(4 if v>28 else(3 if v>22 else(2 if v>16 else(1 if v>12 else 0))))
    if vel5>5:b=min(5,b+1)
    elif vel5<-3 and b>0:b-=1
    if mg<-10 and b<5:b+=1
    if nf<-200 and b<5:b+=1
    if mg>5 and nf>100 and b>0:b-=1
    if dd<-0.25 and b<5:b+=1
    return min(5,max(0,int(b)))

vix['regime']=vix.apply(fp,axis=1)
merged=data.merge(target,on=['ts_code','trade_date'],how='inner')
merged=merged.merge(vix[['trade_date','regime']],on='trade_date',how='left')
merged['regime']=merged['regime'].fillna(2).astype(int)

feats=[c for c in data.columns if c not in ('ts_code','trade_date','close','factor_group','_k','report_date')]
windows=[(2015,2017,2018),(2016,2018,2019),(2017,2019,2020),(2018,2020,2021),(2019,2021,2022),(2020,2022,2023)]
rnames={0:'Calm',1:'Low',2:'Normal',3:'Alert',4:'Danger',5:'Crisis'}

# === SINGLE MODEL ===
print('Single Model vs 6-Regime Models')
print('='*65)
print('%-22s %7s %8s %8s %8s'%('Model','IC','Sharpe','MDD','Excess'))
print('-'*55)

single_ms=[]
for tr_s,tr_e,te_yr in windows:
    train=merged[(merged['trade_date']>=str(tr_s)+'-01-01')&(merged['trade_date']<=str(tr_e)+'-12-31')].dropna(subset=feats+['excess_ret'])
    test=merged[(merged['trade_date']>=str(te_yr)+'-01-01')&(merged['trade_date']<=str(te_yr)+'-12-31')].dropna(subset=feats+['excess_ret'])
    if len(train)<2000 or len(test)<500:continue
    X_tr=train[feats].fillna(train[feats].median());y_tr=train['excess_ret']
    X_te=test[feats].fillna(train[feats].median());y_te=test['excess_ret']
    m=LGBMRegressor(learning_rate=0.05,num_leaves=63,max_depth=10,subsample=0.8,n_estimators=200,verbose=-1,random_state=42,n_jobs=-1)
    m.fit(X_tr,y_tr);pred=m.predict(X_te)
    mask=~np.isnan(pred)&~np.isnan(y_te.values)
    ic,_=stats.spearmanr(pred[mask],y_te.values[mask]) if mask.sum()>30 else (0,1)
    test=test.copy();test['pred']=pred;test['ym']=pd.to_datetime(test['trade_date']).dt.to_period('M')
    mrets=[]
    for mo,g in test.groupby('ym'):
        if len(g)<30:continue
        top=g.nlargest(30,'pred');mrets.append(top['excess_ret'].mean())
    if len(mrets)<3:continue
    rets=np.array(mrets);sh=np.mean(rets)*12/(np.std(rets)*np.sqrt(12)) if np.std(rets)>0 else 0
    mdd=np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    single_ms.append({'ic':ic,'sh':sh,'mdd':mdd})

# === 6 REGIME MODELS ===
regime_ms=[]
for tr_s,tr_e,te_yr in windows:
    train=merged[(merged['trade_date']>=str(tr_s)+'-01-01')&(merged['trade_date']<=str(tr_e)+'-12-31')]
    test=merged[(merged['trade_date']>=str(te_yr)+'-01-01')&(merged['trade_date']<=str(te_yr)+'-12-31')]
    models={};meds={}
    for r in range(6):
        tr_r=train[train['regime']==r].dropna(subset=feats+['excess_ret'])
        if len(tr_r)<500:continue
        meds[r]=tr_r[feats].median()
        m=LGBMRegressor(learning_rate=0.05,num_leaves=31,max_depth=6,subsample=0.8,n_estimators=150,verbose=-1,random_state=42,n_jobs=-1)
        m.fit(tr_r[feats].fillna(meds[r]),tr_r['excess_ret']);models[r]=m
    if len(models)<3:continue
    test['pred']=np.nan
    for r in range(6):
        if r not in models:continue
        te_r=test[test['regime']==r]
        if len(te_r)<50:continue
        test.loc[te_r.index,'pred']=models[r].predict(te_r[feats].fillna(meds.get(r,train[feats].median())))
    tv=test.dropna(subset=['pred','excess_ret'])
    if len(tv)<200:continue
    mask=~np.isnan(tv['pred'].values)&~np.isnan(tv['excess_ret'].values)
    ic,_=stats.spearmanr(tv['pred'].values[mask],tv['excess_ret'].values[mask]) if mask.sum()>30 else (0,1)
    tv['ym']=pd.to_datetime(tv['trade_date']).dt.to_period('M')
    mrets=[]
    for mo,g in tv.groupby('ym'):
        if len(g)<30:continue
        mrets.append(g.nlargest(30,'pred')['excess_ret'].mean())
    if len(mrets)<3:continue
    rets=np.array(mrets);sh=np.mean(rets)*12/(np.std(rets)*np.sqrt(12)) if np.std(rets)>0 else 0
    mdd=np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    regime_ms.append({'ic':ic,'sh':sh,'mdd':mdd})

if single_ms:
    a_ic=np.mean([m['ic'] for m in single_ms]);a_sh=np.mean([m['sh'] for m in single_ms]);a_mdd=np.mean([m['mdd'] for m in single_ms])
    print('%-22s %+.4f %8.3f %+7.1f%% (%dw)'%('Single(24feat)',a_ic,a_sh,a_mdd*100,len(single_ms)))
if regime_ms:
    a_ic2=np.mean([m['ic'] for m in regime_ms]);a_sh2=np.mean([m['sh'] for m in regime_ms]);a_mdd2=np.mean([m['mdd'] for m in regime_ms])
    print('%-22s %+.4f %8.3f %+7.1f%% (%dw)'%('6-Regime Models',a_ic2,a_sh2,a_mdd2*100,len(regime_ms)))

print('='*65)
if single_ms and regime_ms:
    d=a_sh2-a_sh
    print('Regime models vs Single: Sharpe %+.3f  %s'%(d,'WIN' if d>0 else 'LOSS'))
    print('-> 因子体系切换 %s 单模型'%('优于' if d>0 else '不如'))
