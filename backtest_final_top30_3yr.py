# -*- coding: utf-8 -*-
"""
小众战法 Top30 · DD_SMART v2 终验
exit=-12% reentry=+10% floor=10%
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()
TOP_N=30; COST=0.0033; TRAIN_YEARS=3; MCAP_FLOOR=0.20; LIMIT_UP=0.095
EXIT_THRESH=-0.12; REENTRY_THRESH=0.10; FLOOR=0.10
FEATS=['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS=[('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')]

print('='*60)
print('小众战法 Top30 DD_SMART v2 终验')
print('exit=%.0f%% reentry=%.0f%% floor=%.0f%%'%(EXIT_THRESH*100,REENTRY_THRESH*100,FLOOR*100))
print('='*60)

fn=pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date']=pd.to_datetime(fn['trade_date'])

con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
kline=con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
    COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date']=pd.to_datetime(kline['trade_date'])
hs300=con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date'])
hs300['ma50']=hs300['close'].rolling(50).mean(); hs300['high_2y']=hs300['close'].rolling(504).max()
hs300['low_1y']=hs300['close'].rolling(252).min()
con.close()

dates=sorted(fn['trade_date'].unique())
monthly_dates=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates=sorted(monthly_dates)

rd_map={}
for i in range(len(monthly_dates)-1):
    cur=monthly_dates[i]; nxt=monthly_dates[i+1]
    cp=kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_=kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m=cp.join(np_,how='inner'); m['fwd_ret']=m['next_open']/m['close']-1; rd_map[cur]=m
del kline; gc.collect()

hs300_m={}
for d in monthly_dates:
    row=hs300[hs300['trade_date']==d]
    if len(row)>0: r=row.iloc[0]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby=hs300[hs300['trade_date']<=d]
        if len(nearby)>0: r=nearby.iloc[-1]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart_v2(cur_date,state):
    if cur_date not in hs300_m: return 1.0,state
    info=hs300_m[cur_date]; close=info['close']; ma50=info['ma50']
    high_2y=info['high_2y']; low_1y=info['low_1y']
    if pd.isna(high_2y) or pd.isna(ma50): return 1.0,state
    if state['in_market']:
        dd_2y=close/high_2y-1
        if dd_2y<EXIT_THRESH-0.05: return FLOOR,{'in_market':False,'exit_date':cur_date}
        elif dd_2y<EXIT_THRESH: return FLOOR*2,{'in_market':False,'exit_date':cur_date}
        else: return 1.0,state
    else:
        recovery=close/low_1y-1 if pd.notna(low_1y) and low_1y>0 else 0
        above_ma50=close>ma50
        if recovery>REENTRY_THRESH and above_ma50: return 0.7,{'in_market':True,'exit_date':None}
        elif recovery>REENTRY_THRESH*0.7: return FLOOR*2,state
        elif recovery>0.05 and above_ma50: return FLOOR,state
        else: return FLOOR,state

YEARS=sorted(set(d.year for d in monthly_dates)); FIRST_TEST_YR=YEARS[0]+TRAIN_YEARS

fold_pairs={}
for test_yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    train_mds=[d for d in monthly_dates if test_yr-TRAIN_YEARS<=d.year<test_yr]
    if len(train_mds)<24: continue
    pair_ir={}
    for (fa,fb) in ALL_PAIRS:
        spreads=[]
        for rd in train_mds:
            if rd not in rd_map: continue
            day=fn[fn['trade_date']==rd].copy(); px=rd_map[rd]
            valid=set(px.index); day=day[day['ts_code'].isin(valid)]
            if len(day)<100: continue
            for f in [fa,fb]:
                if f in day.columns: day[f+'_r']=day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score']=day[fa+'_r']*day[fb+'_r']
            day['fwd_ret']=px.loc[day['ts_code'].values]['fwd_ret'].values
            vd=day.dropna(subset=['score','fwd_ret'])
            if len(vd)<50: continue
            nq=int(len(vd)*0.2)
            spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
        if len(spreads)>=12: mu=np.mean(spreads); std=np.std(spreads); pair_ir[(fa,fb)]=mu/std if std>0 else 0
    sorted_pairs=sorted(pair_ir.items(),key=lambda x:x[1],reverse=True)
    fold_pairs[test_yr]=[p for p,ir in sorted_pairs[:4]]

all_results=[]; state={'in_market':True,'exit_date':None}
for test_yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    if test_yr not in fold_pairs: continue
    top4=fold_pairs[test_yr]; test_mds=[d for d in monthly_dates if d.year==test_yr]
    if len(test_mds)<3: continue
    for rd in test_mds:
        if rd not in rd_map: continue
        pos,state=dd_smart_v2(rd,state)
        if pos<0.01: all_results.append({'date':str(rd)[:7],'ret':0.0,'yr':rd.year,'pos':pos}); continue
        day=fn[fn['trade_date']==rd].copy(); px=rd_map[rd]
        valid=set(px.index); day=day[day['ts_code'].isin(valid)]
        if len(day)<100: continue
        all_f=list(set([x for p in top4 for x in p]))
        for f in all_f:
            if f in day.columns: day[f+'_r']=day[f].rank(pct=True)
        day['score']=0
        for fa,fb in top4:
            if fa+'_r' in day.columns and fb+'_r' in day.columns: day['score']+=day[fa+'_r']*day[fb+'_r']
        px_match=px.loc[day['ts_code'].values]
        day['mcap']=px_match['mcap'].values; day['ret_1d']=px_match['ret_1d'].values
        day['fwd_ret']=px_match['fwd_ret'].values; day['mcap_r']=day['mcap'].rank(pct=True)
        day=day[day['mcap_r']>=MCAP_FLOOR]; day=day[day['ret_1d']<LIMIT_UP]; day=day[day['fwd_ret'].notna()]
        if len(day)<80: continue
        top=day.nlargest(TOP_N,'score')
        if len(top)<15: continue
        all_results.append({'date':str(rd)[:7],'ret':(top['fwd_ret'].mean()-COST)*pos,'yr':rd.year,'pos':pos})

r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
cum=np.cumprod(1+r_all); mdd=np.min(cum/np.maximum.accumulate(cum)-1)
win=(r_all>0).mean()*100; calmar=ann/abs(mdd) if mdd!=0 else 0
avg_pos=np.mean([x['pos'] for x in all_results if x['pos']>0.01])*100
total_ret=np.prod(1+r_all)-1

print('\n=== DD_SMART v2 终验绩效 ===')
print('年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | Calmar: %+.2f'%(ann*100,sh,mdd*100,calmar))
print('胜率: %.0f%% | 累计: %+.0f%% | 均仓: %.0f%%'%(win,total_ret*100,avg_pos))

crash=[2008,2011,2017,2018,2022]; bull=[2007,2009,2015,2019,2021,2025]
cr_ret=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bl_ret=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1
print('5熊累计: %+.1f%% | 6牛累计: %+.1f%%'%(cr_ret*100,bl_ret*100))

print('\n年      收益   Sharpe    MDD    仓位')
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array(dr); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        ap=np.mean([x['pos'] for x in all_results if x['yr']==yr and x['pos']>0.01])*100
        print('%d %+7.1f%% %+6.2f %+6.1f%% %4.0f%%'%(yr,(np.prod(1+rr)-1)*100,s,m*100,ap))

print('\n耗时: %.0fs'%(time.time()-t0))
