# GTJA vs 现有因子 相关性矩阵
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()

# 1. 计算全部12因子(GTJA5+现有6+price) 在一天内完成
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
print('12因子Spearman相关矩阵 (2010-2026 月频)')
print('='*60)
q='''
WITH daily AS (
    SELECT ts_code, trade_date, open, high, low, close, vol, turnover_rate,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
           close/LAG(close,5) OVER w1 AS ret_5d,
           open/LAG(close) OVER w1-1 AS gap,
           close/LAG(close,12) OVER w1 AS c12,
           close/LAG(close,3) OVER w1 AS c3,
           close/LAG(close,6) OVER w1 AS c6,
           LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol) OVER w1,1)) AS log_vd,
           ABS(ret)/NULLIF(GREATEST(vol*close,1.0),0)*1e10 AS ill_d,
           -1.0/NULLIF(close,0) AS price_rev
    FROM kline_daily WHERE trade_date>=DATE '2010-01-01'
    WINDOW w1 AS (PARTITION BY ts_code ORDER BY trade_date)
),
feats AS (
    SELECT ts_code, trade_date,
           LN(1.0+AVG(ill_d) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING)) AS amihud,
           -AVG(turnover_rate) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS turnover_rev,
           -MAX(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS max_rev,
           -ret_5d AS sr5, gap, price_rev,
           -AVG((close-c12)/NULLIF(c12,0)*vol) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 11 PRECEDING) AS a134_rev,
           -AVG((close-c3)/NULLIF(c3,0)*100+(close-c6)/NULLIF(c6,0)*100) OVER(PARTITION BY ts_code ORDER BY trade_DATE ROWS 11 PRECEDING) AS a027_rev,
           0.0 AS a035,  -- computed separately below
           -SUM(((close-low)-(high-close))/NULLIF(high-low,0)*vol) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS a060_rev,
           -SUM(CASE WHEN ret>0 THEN vol WHEN ret<0 THEN -vol ELSE 0 END) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 5 PRECEDING) AS a043_rev
    FROM daily WHERE ret IS NOT NULL
)
SELECT * FROM feats
WHERE amihud IS NOT NULL AND turnover_rev IS NOT NULL
'''
df=con.execute(q).df()
df['trade_date']=pd.to_datetime(df['trade_date'])
con.close()
print(f'数据: {len(df):,}行 {df["ts_code"].nunique():,}只')

# a035单独算(避免嵌套窗口)
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
a35=con.execute('''
WITH step1 AS (SELECT ts_code,trade_date,-(open-LAG(open)OVER(PARTITION BY ts_code ORDER BY trade_date)) AS d_open FROM kline_daily WHERE trade_date>=DATE '2010-01-01')
SELECT ts_code,trade_date,AVG(d_open)OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 14 PRECEDING) AS a035 FROM step1 WHERE d_open IS NOT NULL
''').df(); a35['trade_date']=pd.to_datetime(a35['trade_date'])
con.close()
df=df.drop(columns=['a035']).merge(a35,on=['ts_code','trade_date'],how='inner')

# VP_Corr额外加
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
vp_r=con.execute('''
WITH d AS (SELECT ts_code,trade_date,
    close/LAG(close)OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
    LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol)OVER(PARTITION BY ts_code ORDER BY trade_date),1)) AS lvd
    FROM kline_daily WHERE trade_date>=DATE '2010-01-01'),
r AS (SELECT ts_code,trade_date,PERCENT_RANK()OVER(PARTITION BY trade_date ORDER BY ret)AS rr,PERCENT_RANK()OVER(PARTITION BY trade_date ORDER BY lvd)AS rv FROM d WHERE ret IS NOT NULL)
SELECT ts_code,trade_date,rr,rv FROM r ORDER BY ts_code,trade_date
''').df()
con.close()
vp_r['trade_date']=pd.to_datetime(vp_r['trade_date'])
codes=vp_r['ts_code'].unique()
res=[]
for i in range(0,len(codes),800):
    b=codes[i:i+800]; bd=vp_r[vp_r['ts_code'].isin(b)]
    for ts,g in bd.groupby('ts_code'):
        g=g.set_index('trade_date').sort_index()
        if len(g)<10: continue
        c=g['rr'].rolling(6,min_periods=5).corr(g['rv'])
        res.append(pd.DataFrame({'ts_code':ts,'trade_date':g.index,'vp_corr':-c.values}))
    if i%4000==0 and i>0: print(f'  VP {i}/{len(codes)}')
vp=pd.concat(res,ignore_index=True); vp['trade_date']=pd.to_datetime(vp['trade_date'])
df=df.merge(vp,on=['ts_code','trade_date'],how='inner'); del vp_r,res,vp; gc.collect()

# 2. 月频Spearman相关
factors=['vp_corr','sr5','amihud','turnover_rev','max_rev','gap','price_rev','a134_rev','a027_rev','a035','a043_rev','a060_rev']
names=['VP_Corr','Short_Rev','Amihud','Turn(-)','Max_Ret(-)','Gap','Price(-)','G134(-)','G027(-)','G035','G043(-)','G060(-)']

df['ym']=df['trade_date'].dt.strftime('%Y-%m')
mc=[]
for ym,g in df.groupby('ym'):
    if len(g)<100: continue
    mc.append(g[factors].rank().corr(method='spearman').values)
avg=np.mean(mc,axis=0)

print(f'\n{"":>13s}',end='')
for n in names: print(f'{n:>10s}',end='')
print()
for i,n in enumerate(names):
    print(f'{n:>13s}',end='')
    for j in range(len(names)):
        v=avg[i][j]
        b='' if i==j else ('!!' if abs(v)>0.5 else ('##' if abs(v)>0.35 else ' .'))
        print(f'{v:>+8.3f}{b}',end='')
    print()

# GTJA与旧因子最大相关
print('\nGTJA→旧因子:' if any('G' in n for n in names) else '')
for i,n in enumerate(names):
    if 'G' in n:
        mr=0; mn=''
        for j,n2 in enumerate(names):
            if 'G' not in n2 and i!=j:
                r=abs(avg[i][j])
                if r>mr: mr=r; mn=n2
        flag=' ⚠冗余' if mr>0.5 else (' 接近' if mr>0.35 else ' ✅独立')
        print(f'  {n}: max r={mr:.3f} with {mn}{flag}')

print(f'\n{time.time()-t0:.0f}s')
