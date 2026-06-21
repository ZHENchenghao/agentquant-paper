# -*- coding: utf-8 -*-
"""24年全周期滚动窗口检验"""
import sys,io;sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from lightgbm import LGBMRegressor
from scipy import stats

data=pd.read_parquet('cache/factors_all.parquet')
c=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

target=c.execute("""SELECT s.ts_code,s.trade_date,(s.fc/s.close-1)-(x.fc/x.close-1) excess_ret
FROM (SELECT ts_code,trade_date,close,LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) fc
      FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
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

# Base features (no regime features)
base_feats=[c for c in data.columns if c not in ('ts_code','trade_date','close','factor_group','_k','report_date','regime')]

# + VIX fingerprint features (regime + velocity + dd)
regime_feats=base_feats+['regime']

# 6 跨周期窗口 (覆盖不同牛熊)
# 6窗口 覆盖2016-2024不同牛熊周期
# W1: 训练2016-2018 测试2019(贸易摩擦反弹)
# W2: 训练2017-2019 测试2020(新冠冲击)
# W3: 训练2018-2020 测试2021(结构牛)
# W4: 训练2019-2021 测试2022(熊市)
# W5: 训练2020-2022 测试2023(震荡修复)
# W6: 训练2021-2023 测试2024(微盘危机)
windows=[(2016,2018,2019),(2017,2019,2020),(2018,2020,2021),(2019,2021,2022),(2020,2022,2023),(2021,2023,2024)]

print('24-Year Rolling Window Validation')
print('='*80)
print('%-25s %7s %8s %8s %8s %8s'%('Window','IC','Sharpe','MDD','Excess','Cycle'))
print('-'*75)

print('Starting tests... train_range=%s test_range=%s'%(str(merged['trade_date'].min())[:10],str(merged['trade_date'].max())[:10]))
for label,feats in [('Baseline(24)',base_feats),('+VIX_Fingerprint(25)',regime_feats)]:
    all_metrics=[]
    for tr_s,tr_e,te_yr in windows:
        train=merged[(merged['trade_date']>=str(tr_s)+'-01-01')&(merged['trade_date']<=str(tr_e)+'-12-31')]
        test=merged[(merged['trade_date']>=str(te_yr)+'-01-01')&(merged['trade_date']<=str(te_yr)+'-12-31')]
        valid=[c for c in feats if c in train.columns]
        try:
            train=train.dropna(subset=valid+['excess_ret'])
            test=test.dropna(subset=valid+['excess_ret'])
        except:continue
        print('  DEBUG %s W%d: train=%d test=%d'%(label,tr_s,len(train),len(test)),flush=True)
        if len(train)<2000 or len(test)<500:print('  SKIP: too small');continue

        X_tr=train[valid].fillna(train[valid].median());y_tr=train['excess_ret']
        X_te=test[valid].fillna(train[valid].median());y_te=test['excess_ret']

        try:
            m=LGBMRegressor(learning_rate=0.05,num_leaves=63,max_depth=10,subsample=0.8,colsample_bytree=0.8,
                             n_estimators=200,verbose=-1,random_state=42,n_jobs=-1)
            m.fit(X_tr,y_tr);pred=m.predict(X_te)
        except Exception as e:
            print('  ERR model: %s'%str(e)[:80]);continue

        mask=~np.isnan(pred)&~np.isnan(y_te.values)
        if mask.sum()<30:print('  SKIP: valid pred<30');continue
        ic,_=stats.spearmanr(pred[mask],y_te.values[mask])

        test2=test.copy();test2['pred']=pred
        test2['ym']=pd.to_datetime(test2['trade_date']).dt.to_period('M')
        mrets=[]
        for mo,g in test2.groupby('ym'):
            if len(g)<30:continue
            top=g.nlargest(30,'pred')
            mrets.append(top['excess_ret'].mean())
        if len(mrets)<3:print('  SKIP: months<3 (%d)'%len(mrets));continue
        rets=np.array(mrets);ann=np.mean(rets)*12;vol=np.std(rets,ddof=1)*np.sqrt(12)
        sh=ann/vol if vol>0 else 0;mdd=np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)

        # 周期标签
        cycle=''
        if te_yr==2019:cycle='贸易摩擦反弹'
        elif te_yr==2020:cycle='新冠冲击'
        elif te_yr==2021:cycle='结构牛'
        elif te_yr==2022:cycle='熊市'
        elif te_yr==2023:cycle='震荡修复'
        elif te_yr==2024:cycle='微盘危机'

        all_metrics.append({'ic':ic,'sh':sh,'mdd':mdd,'cycle':cycle})
        print('  %-10s %s %+.4f %8.3f %+7.1f%% %+7.1f%% %s'%(label,cycle,ic,sh,mdd*100,ic*100,cycle))

    if all_metrics:
        avg_ic=np.mean([m['ic'] for m in all_metrics])
        avg_sh=np.mean([m['sh'] for m in all_metrics])
        avg_mdd=np.mean([m['mdd'] for m in all_metrics])
        pos_ic=sum(1 for m in all_metrics if m['ic']>0)
        pos_sh=sum(1 for m in all_metrics if m['sh']>0)
        print('%-25s %+.4f %8.3f %+7.1f%% %+7.1f%% | IC>0:%d/%d Sh>0:%d/%d'%(
            label+' SUMMARY',avg_ic,avg_sh,avg_mdd*100,avg_ic*100,pos_ic,len(all_metrics),pos_sh,len(all_metrics)))
        print('-'*75)
print('='*80)
