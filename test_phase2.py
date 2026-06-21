# Phase 2: 相关性矩阵 + 因子组合
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0=time.time()
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
print('Phase 2: 相关性矩阵 + 因子组合')
print('='*60)

# === 1. 批量计算6因子 ===
print('[1/3] 批量计算因子...')
big=con.execute("""
WITH daily AS (
    SELECT ts_code, trade_date, open, high, low, close, vol, amount, turnover_rate,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
           close/LAG(close,5) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_5d,
           LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date),1)) AS log_vol_diff,
           ABS(close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1)/NULLIF(amount,0)*1e10 AS illiq_daily,
           open/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS gap
    FROM kline_daily WHERE trade_date>='2010-01-01'
),
ranked AS (
    SELECT *, PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY ret) AS r_ret,
              PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY log_vol_diff) AS r_vol
    FROM daily WHERE ret IS NOT NULL
),
roll AS (
    SELECT ts_code, trade_date, ret_5d, gap,
           AVG(illiq_daily) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amihud,
           AVG(turnover_rate) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS turnover_5d,
           MAX(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS max_ret_5d
    FROM ranked
)
SELECT ts_code, trade_date,
       -ret_5d AS sr5,
       gap,
       amihud,
       -turnover_5d AS turnover_rev,
       -max_ret_5d AS max_rev
FROM roll
WHERE ret_5d IS NOT NULL AND amihud IS NOT NULL AND turnover_5d IS NOT NULL
""").df()
con.close()
big['trade_date']=pd.to_datetime(big['trade_date'])
print(f'  基础因子: {len(big):,}行, {big["ts_code"].nunique():,}只')

# VP_Corr单独rolling corr
print('  计算VP_Corr...')
con2=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
raw=con2.execute("""
WITH daily AS (
    SELECT ts_code, trade_date,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
           LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date),1)) AS log_vol_diff
    FROM kline_daily WHERE trade_date>='2010-01-01'
),
ranked AS (
    SELECT ts_code, trade_date, ret, log_vol_diff,
           PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY ret) AS rank_ret,
           PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY log_vol_diff) AS rank_vol
    FROM daily WHERE ret IS NOT NULL AND log_vol_diff IS NOT NULL
)
SELECT ts_code, trade_date, rank_ret, rank_vol FROM ranked ORDER BY ts_code, trade_date
""").df()
con2.close()

codes=raw['ts_code'].unique()
results=[]
for i in range(0,len(codes),800):
    batch=codes[i:i+800]
    bdf=raw[raw['ts_code'].isin(batch)]
    for ts, g in bdf.groupby('ts_code'):
        g=g.set_index('trade_date').sort_index()
        if len(g)<10: continue
        corr=g['rank_ret'].rolling(6,min_periods=5).corr(g['rank_vol'])
        results.append(pd.DataFrame({'ts_code':ts,'trade_date':g.index,'vp_corr_raw':corr.values}))
    if (i+800)%4000==0: print(f'    {i+800}/{len(codes)}...')

vp=pd.concat(results,ignore_index=True)
vp['vp_corr']=-vp['vp_corr_raw']
vp['trade_date']=pd.to_datetime(vp['trade_date'])
print(f'  VP_Corr: {len(vp):,}行')

big=big.merge(vp[['ts_code','trade_date','vp_corr']],on=['ts_code','trade_date'],how='inner')
big=big.dropna()
print(f'  合并: {len(big):,}行\n')

# === 2. 相关性矩阵 ===
print('[2/3] 截面Spearman相关矩阵...')
FACTORS=['vp_corr','sr5','amihud','turnover_rev','max_rev','gap']
FNAMES={'vp_corr':'VP_Corr','sr5':'Short_Rev','amihud':'Amihud','turnover_rev':'Turnover(-)','max_rev':'Max_Ret(-)','gap':'Gap'}

# 按月抽样
big['ym']=big['trade_date'].dt.strftime('%Y-%m')
monthly_corrs=[]
for ym, g in big.groupby('ym'):
    if len(g)<100: continue
    vals=g[FACTORS].rank()
    corr=vals.corr(method='spearman')
    monthly_corrs.append(corr.values)

avg_corr=np.mean(monthly_corrs,axis=0)
names=[FNAMES[f] for f in FACTORS]

# 矩阵输出
print(f'  {"":>14s}',end='')
for n in names: print(f'{n:>10s}',end='')
print()
for i,n in enumerate(names):
    print(f'  {n:>14s}',end='')
    for j in range(len(names)):
        v=avg_corr[i][j]
        bar='' if i==j else ('!!' if abs(v)>0.5 else ('##' if abs(v)>0.35 else ('#' if abs(v)>0.2 else ' .')))
        print(f'{v:>+8.3f}{bar}',end='')
    print()

# 高相关对
print(f'\n  高相关对(|r|>0.35):')
pairs=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        if abs(avg_corr[i][j])>0.35:
            print(f'    {names[i]} <-> {names[j]}: {avg_corr[i][j]:+.3f}')
            pairs.append((i,j))

# === 3. 因子分组 ===
print(f'\n[3/3] 因子分组 + 合成...')
# 简单分组: 如果r>0.35，归为一族，每族取IR最高的
# 从矩阵看分组情况
grouped=set()
groups=[]
for i in range(len(names)):
    if i in grouped: continue
    grp=[i]
    for j in range(i+1,len(names)):
        if j in grouped: continue
        if abs(avg_corr[i][j])>0.35:
            grp.append(j)
            grouped.add(j)
    grouped.add(i)
    groups.append(grp)

print(f'  因子分组:')
for grp in groups:
    gn=[names[i] for i in grp]
    gs=' + '.join(gn)
    print(f'    {gs}')

# 等权合成
for f in FACTORS:
    mu=big.groupby('trade_date')[f].transform('mean')
    std=big.groupby('trade_date')[f].transform('std')
    big[f'{f}_z']=(big[f]-mu)/std.clip(lower=1e-10)

big['combo']=big[[f'{f}_z' for f in FACTORS]].mean(axis=1)

# IC
con3=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
target=con3.execute("""
    SELECT s.ts_code,s.trade_date,(s.fc/s.close-1.0)-(x.fc/x.close-1.0) AS excess_ret_20d
    FROM (SELECT ts_code,trade_date,close,LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date>='2010-01-01') s
    JOIN (SELECT trade_date,close,LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2010-01-01') x
    ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df(); target['trade_date']=pd.to_datetime(target['trade_date'])
con3.close()

merged=big.merge(target,on=['ts_code','trade_date'],how='inner').dropna()

# 合成因子IC
daily_ic=merged.groupby('trade_date').apply(
    lambda g: pd.Series({'rankic': g['combo'].rank().corr(g['excess_ret_20d'].rank()) if len(g)>=30 else np.nan})
).reset_index().dropna(subset=['rankic'])
daily_ic['year']=daily_ic['trade_date'].dt.year

avg_ic=daily_ic['rankic'].mean(); ir=avg_ic/daily_ic['rankic'].std()
pos=(daily_ic['rankic']>0).mean()*100

print(f'\n  === 合成因子 ===')
print(f'  RankIC: {avg_ic:+.4f} | IR: {ir:.2f} | IC>0: {pos:.1f}%')
print(f'  分年:')
for yr in range(2010,2027):
    d=daily_ic[daily_ic['year']==yr]
    if len(d)>20:
        ic_m=d['rankic'].mean(); ic_s=d['rankic'].std()
        pos_d=(d['rankic']>0).mean()*100
        ir_d=ic_m/ic_s if ic_s>0 else 0
        print(f'    {yr}: IC={ic_m:+.4f} IR={ir_d:.2f} pos={pos_d:.0f}%')

# 对比单因子
print(f'\n  === 单因子对比(同区间2010-2026) ===')
for f in FACTORS:
    dic=merged.groupby('trade_date').apply(
        lambda g: pd.Series({'rankic': g[f].rank().corr(g['excess_ret_20d'].rank()) if len(g)>=30 else np.nan})
    ).reset_index().dropna(subset=['rankic'])
    am=dic['rankic'].mean(); ir_s=am/dic['rankic'].std()
    pos_s=(dic['rankic']>0).mean()*100
    print(f'  {FNAMES[f]:>14s}: IC={am:+.4f} IR={ir_s:.2f} IC>0={pos_s:.0f}%')

print(f'\n耗时: {time.time()-t0:.0f}s')
