# -*- coding: utf-8 -*-
import sys, io, os, time
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb, numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats

CACHE='D:/AgentQuant/our/cache/factors_all.parquet'
DB='D:/FreeFinanceData/data/duckdb/finance.db'

def get_db():
    for i in range(5):
        try: c=duckdb.connect(DB,read_only=True);c.execute('SELECT 1');return c
        except: time.sleep(min(2**i,10))
    return duckdb.connect(DB,read_only=True)

print('Loading cache...',end=' ',flush=True)
data=pd.read_parquet(CACHE)
c=get_db()

target=c.execute('''
    WITH sf AS (SELECT ts_code,trade_date,close,LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) fc
                FROM kline_daily WHERE trade_date BETWEEN '2015-01-01' AND '2026-06-16'),
         xf AS (SELECT trade_date,close,LEAD(close,20) OVER(ORDER BY trade_date) fc
                FROM kline_daily WHERE ts_code='sh000300')
    SELECT s.ts_code,s.trade_date,(s.fc/s.close-1)-(x.fc/x.close-1) excess_ret
    FROM sf s JOIN xf x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
''').df()

# VIX regime
vix_data=c.execute('''
    WITH base AS (
        SELECT m.trade_date,m.vix,mt.margin_balance,
               LAG(mt.margin_balance,20) OVER w AS mb20a,
               SUM(nb.net_flow) OVER w2 AS north_20d,
               k.close,MAX(k.close) OVER w3 AS peak_250
        FROM macro_indicators m
        LEFT JOIN margin_trading mt ON m.trade_date=mt.trade_date
        LEFT JOIN north_bound_flow nb ON m.trade_date=nb.trade_date
        LEFT JOIN kline_daily k ON m.trade_date=k.trade_date AND k.ts_code='sh000300'
        WHERE m.vix IS NOT NULL
        WINDOW w AS (ORDER BY m.trade_date),
               w2 AS (ORDER BY m.trade_date ROWS 19 PRECEDING),
               w3 AS (ORDER BY k.trade_date ROWS 249 PRECEDING)
    )
    SELECT trade_date,vix,
           vix-LAG(vix,5) OVER(ORDER BY trade_date) vel5,
           (margin_balance/NULLIF(mb20a,0)-1)*100 mg20,
           north_20d,close/NULLIF(peak_250,0)-1 dd
    FROM base
''').df()
c.close()

def fingerprint(row):
    v=row['vix'];vel5=row.get('vel5',0)or 0;mg=row.get('mg20',0)or 0
    nf=row.get('north_20d',0)or 0;dd=row.get('dd',0)or 0
    if pd.isna(v):return -1
    b=5 if v>35 else(4 if v>28 else(3 if v>22 else(2 if v>16 else(1 if v>12 else 0))))
    if vel5>5:b=min(5,b+1)
    elif vel5<-3 and b>0:b-=1
    if mg<-10 and b<5:b+=1
    if nf<-200 and b<5:b+=1
    if mg>5 and nf>100 and b>0:b-=1
    if dd<-0.25 and b<5:b+=1
    return min(5,max(0,int(b)))

vix_data['regime']=vix_data.apply(fingerprint,axis=1)
data=data.merge(vix_data[['trade_date','regime']],on='trade_date',how='left')
data['regime']=data['regime'].fillna(2).astype(int)

feat_base=[c for c in data.columns if c not in
    ('ts_code','trade_date','excess_ret','close','factor_group','_k','report_date')]

# Generate interaction features
print('Base features: %d, VIX regimes: 6' % len(feat_base))
interact_cols=[]
for f in feat_base:
    if f in ('regime',):continue
    for r in range(6):
        cn='%s_R%d'%(f,r)
        data[cn]=np.where(data['regime']==r,data[f].fillna(0),0)
        interact_cols.append(cn)

feat_baseline=[c for c in feat_base if c!='regime']

windows=[
    ('W1',2015,2017,2018),('W2',2016,2018,2019),('W3',2017,2019,2020),
    ('W4',2018,2020,2021),('W5',2019,2021,2022),('W6',2020,2022,2023),
]

print('='*60)
print('  Factor x VIX Regime Rolling Backtest')
print('='*60)
print('  %-25s %7s %8s %8s %8s' % ('Model','IC','Sharpe','MDD','Excess'))
print('  '+'-'*55)

results=[]
for label,feats in [('Baseline(24)',feat_baseline),('Interaction(~140)',interact_cols)]:
    combo_m=[]
    for wname,tr_start,tr_end,test_yr in windows:
        train=data[(data['trade_date']>=str(tr_start)+'-01-01')&(data['trade_date']<=str(tr_end)+'-12-31')]
        test=data[(data['trade_date']>=str(test_yr)+'-01-01')&(data['trade_date']<=str(test_yr)+'-12-31')]
        valid=[c for c in feats if c in train.columns]
        target_col = 'excess_ret' if 'excess_ret' in train.columns else ('fwd_ret' if 'fwd_ret' in train.columns else None)
        if target_col is None: continue
        train=train.dropna(subset=valid+[target_col])
        test=test.dropna(subset=valid+[target_col])
        if len(train)<2000 or len(test)<500:continue
        X_tr=train[valid].fillna(train[valid].median());y_tr=train['excess_ret']
        X_te=test[valid].fillna(train[valid].median());y_te=test['excess_ret']
        model=LGBMRegressor(learning_rate=0.05,num_leaves=63,max_depth=10,
                             subsample=0.8,colsample_bytree=0.8,
                             n_estimators=200,verbose=-1,random_state=42,n_jobs=-1)
        model.fit(X_tr,y_tr)
        pred=model.predict(X_te)
        mask=~np.isnan(pred)&~np.isnan(y_te.values)
        ic,_=stats.spearmanr(pred[mask],y_te.values[mask]) if mask.sum()>30 else (0,1)
        test_pred=test.copy();test_pred['pred']=pred
        test_pred['ym']=pd.to_datetime(test_pred['trade_date']).dt.to_period('M')
        mrets=[]
        for m,g in test_pred.groupby('ym'):
            if len(g)<30:continue
            top=g.nlargest(30,'pred');mrets.append(top['excess_ret'].mean())
        if len(mrets)<3:continue
        rets=np.array(mrets);ann=np.mean(rets)*12;vol=np.std(rets,ddof=1)*np.sqrt(12)
        sh=ann/vol if vol>0 else 0
        cum=np.cumprod(1+rets);mdd=np.min(cum/np.maximum.accumulate(cum)-1)
        combo_m.append({'ic':ic,'sharpe':sh,'mdd':mdd,'n':len(mrets)})
    if combo_m:
        avg_ic=np.mean([m['ic'] for m in combo_m])
        avg_sh=np.mean([m['sharpe'] for m in combo_m])
        avg_mdd=np.mean([m['mdd'] for m in combo_m])
        avg_n=np.mean([m['n'] for m in combo_m])
        print('  %-25s %+.4f %8.3f %+7.1f%% %+7.1f%% (%dwin)' % (label,avg_ic,avg_sh,avg_mdd*100,avg_ic*100,len(combo_m)))
        results.append({'label':label,'ic':avg_ic,'sharpe':avg_sh,'mdd':avg_mdd})

print('='*60)
if len(results)>=2:
    better=results[1]['sharpe']-results[0]['sharpe']
    ic_diff=results[1]['ic']-results[0]['ic']
    mdd_diff=results[1]['mdd']-results[0]['mdd']
    print('  Interaction vs Baseline:')
    print('    IC:   %+.4f'%ic_diff)
    print('    Sharpe: %+.3f'%better)
    print('    MDD:  %+.1f%%'%(mdd_diff*100))
    if better>0: print('  WIN: Interaction features better')
    else: print('  LOSS: Baseline better, interaction = noise')
    if ic_diff>0: print('  IC improved -> regime-dependent factors work')
print('='*60)
