# 全乘法: 9因子版(含Price/Alpha060/Ab_Sell) vs 6因子版
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()

# 加载9因子
f9=pd.read_parquet('D:/AgentQuant/our/cache/factors_9f.parquet')
f9['trade_date']=pd.to_datetime(f9['trade_date'])
print(f'9f: {len(f9):,}行 {f9["trade_date"].min().date()}~{f9["trade_date"].max().date()}')

# 加载6因子(对照)
f6=pd.read_parquet('D:/AgentQuant/our/cache/factors_new6_v2.parquet')
f6['trade_date']=pd.to_datetime(f6['trade_date'])

# 价格映射
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kl=con.execute('''
    SELECT ts_code, trade_date, open, close,
           COALESCE(close*total_share/10000, GREATEST(amount,close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>=DATE '2010-01-01'
''').df(); kl['trade_date']=pd.to_datetime(kl['trade_date'])
con.close()

dates=sorted(f9['trade_date'].unique()); md=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

rd_map={}
for i in range(len(md)-1):
    c=md[i]; n=md[i+1]
    cp=kl[kl['trade_date']==c][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_=kl[kl['trade_date']==n][['ts_code','open']].rename(columns={'open':'no'}).set_index('ts_code')
    m=cp.join(np_,how='inner'); m['fwd_ret']=m['no']/m['close']-1; rd_map[c]=m
del kl; gc.collect()

TOP=15; C=0.0033

def run_mul(fn, pairs, label):
    """全乘法回测"""
    R=[]
    for rd in md:
        if rd not in rd_map: continue
        d=fn[fn['trade_date']==rd].copy(); p=rd_map[rd]
        if len(d)<100: continue
        valid=set(p.index); d=d[d['ts_code'].isin(valid)]
        if len(d)<50: continue

        pv=p.loc[d['ts_code'].values]
        d['mcap']=pv['mcap'].values; d['ret_1d']=pv['ret_1d'].values; d['fwd_ret']=pv['fwd_ret'].values
        d=d[d['mcap'].rank(pct=True)>=0.20]; d=d[d['ret_1d']<0.095]; d=d[d['fwd_ret'].notna()]
        if len(d)<50: continue

        # 排名: 提取所有涉及因子
        all_f=list(set([x for p in pairs for x in p]))
        for f in all_f:
            if f in d.columns: d[f'{f}_r']=d[f].rank(pct=True)

        # 全乘法
        d['score']=0
        for a,b in pairs:
            if f'{a}_r' in d.columns and f'{b}_r' in d.columns:
                d['score']+=d[f'{a}_r']*d[f'{b}_r']

        top=d.nlargest(TOP,'score')
        if len(top)<5: continue
        R.append({'ret':top['fwd_ret'].mean()-C,'yr':rd.year})
    return R

# 6因子: 已有最佳4对
pairs_6f=[('amihud','sr5'),('amihud','turnover_rev'),('sr5','vp_corr'),('turnover_rev','max_rev')]

# 9因子: 加Price/Alpha060/Ab_Sell后重新选最强对
# 先看新因子和旧因子的交互
all_9f=['amihud','sr5','turnover_rev','max_rev','vp_corr','gap','price_rev','a060_rev','ab_sell_rev']
# 选最强的6对(基于之前交互测试+文献逻辑)
pairs_9f=[('amihud','sr5'),('amihud','turnover_rev'),('sr5','vp_corr'),('turnover_rev','max_rev'),
          ('amihud','price_rev'),('amihud','ab_sell_rev')]

R6=run_mul(f6, pairs_6f, '6f')
R9=run_mul(f9, pairs_9f, '9f')
RS={'6f':R6,'9f':R9}

print('全乘法: 6因子 vs 9因子')
print('='*50)
for lbl in ['6f','9f']:
    r=np.array([x['ret'] for x in RS[lbl]])
    if len(r)<5: continue
    a=np.mean(r)*12; s=a/(np.std(r)*np.sqrt(12)) if np.std(r)>0 else 0
    c=np.cumprod(1+r); m=np.min(c/np.maximum.accumulate(c)-1)
    w=(r>0).mean()*100
    print(f'{lbl}: Sharpe={s:+.2f} 年化={a*100:+.1f}% MDD={m*100:.1f}% Win={w:.0f}% n={len(r)}')

# 分年
for lbl in ['6f','9f']:
    rr=RS[lbl]
    print(f'\n{lbl}分年:')
    for yr in range(2011,2026):
        ry=[x['ret'] for x in rr if x['yr']==yr]
        if len(ry)>=3:
            ay=np.mean(ry)*12
            print(f'  {yr}: {ay*100:+.1f}% ',end='')
    print()

print(f'\n{time.time()-t0:.0f}s')
