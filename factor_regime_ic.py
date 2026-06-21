# -*- coding: utf-8 -*-
import sys,io;sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from scipy import stats

data=pd.read_parquet('cache/factors_all.parquet')
c=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

q="""SELECT s.ts_code,s.trade_date,(s.fc/s.close-1)-(x.fc/x.close-1) excess_ret
FROM (SELECT ts_code,trade_date,close,LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) fc
      FROM kline_daily WHERE trade_date BETWEEN '2015-01-01' AND '2026-06-16') s
JOIN (SELECT trade_date,close,LEAD(close,20) OVER(ORDER BY trade_date) fc
      FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date
WHERE s.fc IS NOT NULL"""
target=c.execute(q).df()

q2="""SELECT trade_date,vix,vix-LAG(vix,5) OVER w vel5,
       (margin_balance/NULLIF(LAG(margin_balance,20) OVER w,0)-1)*100 mg20,
       SUM(net_flow) OVER w2 north_20d,
       close/NULLIF(MAX(close) OVER w3,0)-1 dd
FROM macro_indicators m
LEFT JOIN margin_trading mt USING(trade_date)
LEFT JOIN north_bound_flow nb USING(trade_date)
LEFT JOIN kline_daily k ON m.trade_date=k.trade_date AND k.ts_code='sh000300'
WHERE m.vix IS NOT NULL
WINDOW w AS (ORDER BY m.trade_date),w2 AS (ORDER BY m.trade_date ROWS 19 PRECEDING),w3 AS (ORDER BY k.trade_date ROWS 249 PRECEDING)"""
vix=c.execute(q2).df()
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

vix['regime']=vix.apply(fingerprint,axis=1)
merged=data.merge(target,on=['ts_code','trade_date'],how='inner')
merged=merged.merge(vix[['trade_date','regime']],on='trade_date',how='left')
merged['regime']=merged['regime'].fillna(2).astype(int)

factors=[c for c in data.columns if c not in ('ts_code','trade_date','close','factor_group','_k','report_date')][:15]
rnames={0:'Calm',1:'Low',2:'Normal',3:'Alert',4:'Danger',5:'Crisis'}

print('Per-Factor IC by VIX Regime')
print('='*90)
hdr='%-20s'%'Factor'
for r in range(6):hdr+='%8s'%rnames[r]
print(hdr+' %7s %7s %s'%('Range','|IC|','Sensitive?'))
print('-'*90)

sensitive=[]
for f in factors:
    if f not in merged.columns:continue
    ics=[]
    for r in range(6):
        sub=merged[(merged['regime']==r)&merged[f].notna()&merged['excess_ret'].notna()]
        if len(sub)<100:ics.append(0);continue
        ic,_=stats.spearmanr(sub[f],sub['excess_ret']);ics.append(ic)
    rng=max(ics)-min(ics)
    icm=np.mean([abs(x) for x in ics])
    sens=rng>0.025
    if sens:sensitive.append((f,rng,ics,icm))
    row='%-20s'%f[:20]
    for ic in ics:row+='%+7.3f'%ic
    row+=' %+7.3f %+7.3f %s'%(rng,icm,'*SENSITIVE*' if sens else '')
    print(row)

print()
print('Sensitive factors (IC flips across VIX regimes):')
for f,rng,ics,icm in sorted(sensitive,key=lambda x:-x[1]):
    best=rnames[ics.index(max(ics))]
    worst=rnames[ics.index(min(ics))]
    print('  %-20s range=%+.3f best=%s(%+.3f) worst=%s(%+.3f)'%(f[:20],rng,best,max(ics),worst,min(ics)))
print('='*90)
