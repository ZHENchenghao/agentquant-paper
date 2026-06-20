# -*- coding: utf-8 -*-
"""
DD_SMART门禁参数网格搜索
=========================
网格: 出场阈值(-12%/-15%/-18%/-20%) × 回场动量(10%/15%/20%) × 底仓(10%/15%/20%)
目标: 最小化MDD, 同时保持Sharpe>0.5
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings, itertools
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N=30; COST=0.0033; TRAIN_YEARS=5; MCAP_FLOOR=0.20; LIMIT_UP=0.095
FEATS=['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS=[('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')]

print("="*60)
print("DD_SMART 参数网格搜索")
print("="*60)

# 加载
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
hs300['ma50']=hs300['close'].rolling(50).mean()
hs300['high_2y']=hs300['close'].rolling(504).max()
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
    m=cp.join(np_,how='inner'); m['fwd_ret']=m['next_open']/m['close']-1
    rd_map[cur]=m
del kline; gc.collect()

hs300_m={}
for d in monthly_dates:
    row=hs300[hs300['trade_date']==d]
    if len(row)>0:
        r=row.iloc[0]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby=hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r=nearby.iloc[-1]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

YEARS=sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR=YEARS[0]+TRAIN_YEARS

# 训练选对(共享)
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
                if f in day.columns: day[f'{f}_r']=day[f].rank(pct=True)
            if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns: continue
            day['score']=day[f'{fa}_r']*day[f'{fb}_r']
            day['fwd_ret']=px.loc[day['ts_code'].values]['fwd_ret'].values
            vd=day.dropna(subset=['score','fwd_ret'])
            if len(vd)<50: continue
            nq=int(len(vd)*0.2)
            spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
        if len(spreads)>=12:
            mu=np.mean(spreads); std=np.std(spreads); pair_ir[(fa,fb)]=mu/std if std>0 else 0
    sorted_pairs=sorted(pair_ir.items(),key=lambda x:x[1],reverse=True)
    fold_pairs[test_yr]=[p for p,ir in sorted_pairs[:4]]

print(f"[1] 训练完成: {len(fold_pairs)}折")

# 网格搜索
EXIT_LEVELS=[-0.20,-0.18,-0.15,-0.12]  # 出场阈值
REENTRY_REC=[0.10,0.15,0.20]            # 回场动量
FLOOR_LEVELS=[0.10,0.15,0.20]           # 底仓

def dd_smart_v2(cur_date,state,exit_thresh,reentry_thresh,floor):
    if cur_date not in hs300_m: return 1.0, state
    info=hs300_m[cur_date]
    close=info['close']; ma50=info['ma50']
    high_2y=info['high_2y']; low_1y=info['low_1y']
    if pd.isna(high_2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y=close/high_2y-1
        if dd_2y<exit_thresh-0.05: return floor, {'in_market':False,'exit_date':cur_date}
        elif dd_2y<exit_thresh: return floor*2, {'in_market':False,'exit_date':cur_date}
        else: return 1.0, state
    else:
        recovery=close/low_1y-1 if pd.notna(low_1y) and low_1y>0 else 0
        above_ma50=close>ma50
        if recovery>reentry_thresh and above_ma50: return 0.7, {'in_market':True,'exit_date':None}
        elif recovery>reentry_thresh*0.7: return floor*2, state
        elif recovery>0.05 and above_ma50: return floor, state
        else: return floor, state

results=[]
total_combos=len(EXIT_LEVELS)*len(REENTRY_REC)*len(FLOOR_LEVELS)
combo_num=0

for exit_th, reentry, floor in itertools.product(EXIT_LEVELS, REENTRY_REC, FLOOR_LEVELS):
    combo_num+=1
    all_rets=[]; state={'in_market':True,'exit_date':None}

    for test_yr in range(FIRST_TEST_YR,YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4=fold_pairs[test_yr]
        test_mds=[d for d in monthly_dates if d.year==test_yr]
        if len(test_mds)<3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos,state=dd_smart_v2(rd,state,exit_th,reentry,floor)
            if pos<0.01:
                all_rets.append(0.0); continue

            day=fn[fn['trade_date']==rd].copy(); px=rd_map[rd]
            valid=set(px.index); day=day[day['ts_code'].isin(valid)]
            if len(day)<100: continue

            all_f=list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f'{f}_r']=day[f].rank(pct=True)

            day['score']=0
            for fa,fb in top4:
                if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                    day['score']+=day[f'{fa}_r']*day[f'{fb}_r']

            px_match=px.loc[day['ts_code'].values]
            day['mcap']=px_match['mcap'].values
            day['ret_1d']=px_match['ret_1d'].values
            day['fwd_ret']=px_match['fwd_ret'].values
            day['mcap_r']=day['mcap'].rank(pct=True)
            day=day[day['mcap_r']>=MCAP_FLOOR]
            day=day[day['ret_1d']<LIMIT_UP]
            day=day[day['fwd_ret'].notna()]
            if len(day)<80: continue

            top=day.nlargest(TOP_N,'score')
            if len(top)<15: continue
            all_rets.append((top['fwd_ret'].mean()-COST)*pos)

    if len(all_rets)<100: continue
    r=np.array(all_rets)
    ann=np.mean(r)*12; vol=np.std(r)*np.sqrt(12)
    sh=ann/vol if vol>0 else 0
    cum=np.cumprod(1+r); mdd=np.min(cum/np.maximum.accumulate(cum)-1)
    calmar=ann/abs(mdd) if mdd!=0 else 0
    avg_pos=np.mean([p for p in all_rets if abs(p)>1e-6])*100 if any(abs(p)>1e-6 for p in all_rets) else 0

    results.append({'exit':exit_th,'reentry':reentry,'floor':floor,
                    'ann':ann,'sharpe':sh,'mdd':mdd,'calmar':calmar,'months':len(r)})

    # Print best so far
    if combo_num%9==0:
        best=sorted(results,key=lambda x:x['calmar'],reverse=True)[0]
        ba,v1=best['ann']*100,best['sharpe']; bm,v2=best['mdd']*100,best['calmar']
        be=best['exit']; br=best['reentry']; bf=best['floor']
        print('  [%d/%d] best: exit=%+.0f reentry=%+.0f floor=%+.0f ann=%+.1f%% sh=%+.2f mdd=%.1f%% calmar=%+.2f' % (combo_num,total_combos,be*100,br*100,bf*100,ba,v1,bm,v2))

# 输出Top10
print('\n' + '='*60)
print('Top10 参数组合')
print('%6s %8s %6s %8s %7s %7s %7s' % ('exit','reentry','floor','年化','Sharpe','MDD','Calmar'))
print('-'*55)
for r in sorted(results,key=lambda x:x['calmar'],reverse=True)[:10]:
    ra=r['ann']*100; rm=r['mdd']*100; rc=r['calmar']; rs=r['sharpe']
    print('%+6.0f%% %+7.0f%% %5.0f%% %+7.1f%% %+6.2f %+6.1f%% %+6.2f' % (r['exit']*100,r['reentry']*100,r['floor']*100,ra,rs,rm,rc))

# 对比baseline
base=sorted(results,key=lambda x:(x['exit']==-0.20 and x['reentry']==0.15 and x['floor']==0.15),reverse=True)
if base:
    b=base[0]
    ba2=b['ann']*100; bm2=b['mdd']*100
    print('\nBaseline(-20%%,15%%,15%%): ann=%+.1f%% sh=%+.2f mdd=%.1f%%' % (ba2,b['sharpe'],bm2))

print(f'\n耗时: {(time.time()-t0)/60:.1f}分')
