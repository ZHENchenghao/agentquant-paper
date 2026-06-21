import duckdb,pandas as pd,numpy as np
c=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

q="""SELECT m.trade_date,vix,vix-LAG(vix,5) OVER(ORDER BY m.trade_date) vel5,
       close/NULLIF(MAX(close) OVER(ORDER BY k.trade_date ROWS 249 PRECEDING),0)-1 dd
FROM macro_indicators m
LEFT JOIN kline_daily k ON m.trade_date=k.trade_date AND k.ts_code='sh000300'
WHERE m.vix IS NOT NULL AND m.trade_date>='2002-01-01'"""
df=c.execute(q).df()
c.close()

def fp(row):
    v=row['vix'];vel5=row.get('vel5',0)or 0;dd=row.get('dd',0)or 0
    if pd.isna(v):return 2
    b=5 if v>35 else(4 if v>28 else(3 if v>22 else(2 if v>16 else(1 if v>12 else 0))))
    if vel5>5:b=min(5,b+1)
    elif vel5<-3 and b>0:b-=1
    if dd<-0.25 and b<5:b+=1
    return min(5,max(0,int(b)))

df['regime']=df.apply(fp,axis=1)
rnames={0:'Calm',1:'Low',2:'Normal',3:'Alert',4:'Danger',5:'Crisis'}
for r in range(6):
    n=(df['regime']==r).sum()
    print('%d %-8s: %5d days (%4.1f%%)'%(r,rnames[r],n,n/len(df)*100))
print('Total: %d days (2002-2026, 24yr)'%len(df))
