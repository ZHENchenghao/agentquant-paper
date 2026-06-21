# -*- coding: utf-8 -*-
"""宏观打分卡叠加测试: 小众+ETF 双策略"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()
COST=0.0033; TRAIN_YEARS=5; MCAP_FLOOR=0.20; LIMIT_UP=0.095
EXIT_THRESH=-0.12; REENTRY_THRESH=0.10; FLOOR=0.10

print("="*60)
print("宏观打分卡叠加测试")
print("="*60)

# ===== 加载 =====
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
# 行为因子
fn=pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date']=pd.to_datetime(fn['trade_date'])
# 行业
ind=con.execute("SELECT industry,trade_date,close FROM proxy_industry_daily WHERE trade_date>='2005-01-01' ORDER BY industry,trade_date").df()
ind['trade_date']=pd.to_datetime(ind['trade_date'])
# HS300
hs300=con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date'])
hs300['ma50']=hs300['close'].rolling(50).mean(); hs300['high_2y']=hs300['close'].rolling(504).max()
hs300['low_1y']=hs300['close'].rolling(252).min()
# 宏观
macro=con.execute("SELECT trade_date,m1_growth,m2_growth FROM macro_indicators WHERE m1_growth IS NOT NULL ORDER BY trade_date").df()
macro['trade_date']=pd.to_datetime(macro['trade_date']); macro=macro.set_index('trade_date')
ext=con.execute("SELECT trade_date,vix FROM macro_indicators WHERE vix IS NOT NULL ORDER BY trade_date").df()
ext['trade_date']=pd.to_datetime(ext['trade_date']); ext=ext.set_index('trade_date')
nb=con.execute("SELECT trade_date,net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date']=pd.to_datetime(nb['trade_date']); nb=nb.set_index('trade_date')
margin=con.execute("SELECT trade_date,margin_balance FROM margin_trading ORDER BY trade_date").df()
margin['trade_date']=pd.to_datetime(margin['trade_date']); margin=margin.set_index('trade_date')
# 价格
kline=con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2005-01-01'""").df()
kline['trade_date']=pd.to_datetime(kline['trade_date'])
con.close()

# ===== 月度 =====
dates=sorted(fn['trade_date'].unique())
md=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

# HS300信号
hs_m={}
for d in md:
    row=hs300[hs300['trade_date']==d]
    if len(row)>0: r=row.iloc[0]; hs_m[d]={'c':r['close'],'m50':r['ma50'],'h2y':r['high_2y'],'l1y':r['low_1y']}
    else:
        nb_=hs300[hs300['trade_date']<=d]
        if len(nb_)>0: r=nb_.iloc[-1]; hs_m[d]={'c':r['close'],'m50':r['ma50'],'h2y':r['high_2y'],'l1y':r['low_1y']}

def gate(rd,state):
    if rd not in hs_m: return 1.0,state
    i=hs_m[rd]; c=i['c']; m50=i['m50']; h2y=i['h2y']; l1y=i['l1y']
    if pd.isna(h2y) or pd.isna(m50): return 1.0,state
    if state['in']:
        dd=c/h2y-1
        if dd<EXIT_THRESH-0.05: return FLOOR,{'in':False,'exit':rd}
        elif dd<EXIT_THRESH: return FLOOR*2,{'in':False,'exit':rd}
        else: return 1.0,state
    else:
        rec=c/l1y-1 if pd.notna(l1y) and l1y>0 else 0; abv=c>m50
        if rec>REENTRY_THRESH and abv: return 0.7,{'in':True,'exit':None}
        elif rec>REENTRY_THRESH*0.7: return FLOOR*2,state
        elif rec>0.05 and abv: return FLOOR,state
        else: return FLOOR,state

# ===== 宏观打分卡 =====
def align(series,mdates):
    r={}; sr=series.sort_index()
    for d in mdates:
        v=sr[sr.index<=d]
        if len(v)>0: r[d]=v.iloc[-1]
    return pd.Series(r)

scissor=(macro['m1_growth']-macro['m2_growth']).dropna()
scissor_m=align(scissor,md)
vix_m=align(ext['vix'],md)
nb_m=align(nb['net_flow'],md); nb_cum3=nb_m.rolling(3).sum()
mg_m=align(margin['margin_balance'],md); mg_pct3=mg_m.pct_change(3)
hs300_ret=hs300.set_index('trade_date')['close'].pct_change()
hs300_vol=hs300_ret.rolling(20).std()*np.sqrt(252)
vol_m=align(hs300_vol,md)

def macro_score(rd):
    """0-4分: 每满足一个条件+1分"""
    s=0
    # M1-M2>0
    if rd in scissor_m.index and scissor_m[rd]>0: s+=1
    # VIX<30
    if rd in vix_m.index and vix_m[rd]<30: s+=1
    # 北向3月累计>0
    if rd in nb_cum3.index and nb_cum3[rd]>0: s+=1
    # 两融3月>0
    if rd in mg_pct3.index and mg_pct3[rd]>0: s+=1
    # 低波动(<30%分位)
    if rd in vol_m.index and vol_m[rd]<vol_m.quantile(0.30): s+=1
    return s/5.0  # 归一化到0-1

# ===== 小众战法(简化) =====
# 行业ETF因子
ind=ind.sort_values(['industry','trade_date'])
ind['ret_20d']=ind.groupby('industry')['close'].pct_change(20)
ind['ret_60d']=ind.groupby('industry')['close'].pct_change(60)
ind['ret_1d']=ind.groupby('industry')['close'].pct_change(); ind['ret_5d']=ind.groupby('industry')['close'].pct_change(5)
ind['vol_20d']=ind.groupby('industry')['ret_1d'].transform(lambda x:x.rolling(20).std())
ind['high20']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).max()); ind['low20']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).min())
ma60=ind.groupby('industry')['close'].transform(lambda x:x.rolling(60).mean()); ind['div_ma60']=ind['close']/ma60-1
mt=ind[ind['trade_date'].isin(md)].copy()
mt['mom_20']=mt['ret_20d']; mt['mom_60']=mt['ret_60d']; mt['rev_5']=-mt['ret_5d']
mt['crowd']=-mt['vol_20d']; pr=mt['high20']-mt['low20']; mt['rps']=mt['ret_20d']/pr.replace(0,0.01)
MOM=['mom_20','mom_60','rev_5','crowd','rps','div_ma60']

# 小众因子
FEATS= ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS=[('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')]

# 价格映射
rd_map={}
for i in range(len(md)-1):
    cur=md[i]; nxt=md[i+1]
    cp=kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_=kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'no'}).set_index('ts_code')
    m=cp.join(np_,how='inner'); m['fwd_ret']=m['no']/m['close']-1; rd_map[cur]=m
del kline; gc.collect()

# ETF目标
tg={}
for i in range(len(md)-1):
    cur=md[i]; nxt=md[i+1]
    cd=ind[ind['trade_date']==cur]; nd=ind[ind['trade_date']==nxt]
    if len(cd)>=10 and len(nd)>=10: tg[cur]=nd.set_index('industry')['close']/cd.set_index('industry')['close']-1

YEARS=sorted(set(d.year for d in md)); FY=YEARS[0]+TRAIN_YEARS+1

# ===== 两个版本跑WF: 无宏观 vs 有宏观 =====
for STRAT in ['小众','ETF']:
    for MACRO_ON in [False, True]:
        label='%s%s'%(STRAT,'+宏观' if MACRO_ON else '')
        all_r=[]; state={'in':True,'exit':None}

        # 小众训练选对
        if STRAT=='小众':
            fold_pairs={}
            for test_yr in range(FY,YEARS[-1]+1):
                train_mds=[d for d in md if test_yr-TRAIN_YEARS<=d.year<test_yr]
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
                        day['score']=day[fa+'_r']*day[fb+'_r']; day['fwd_ret']=px.loc[day['ts_code'].values]['fwd_ret'].values
                        vd=day.dropna(subset=['score','fwd_ret'])
                        if len(vd)<50: continue
                        nq=int(len(vd)*0.2)
                        spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
                    if len(spreads)>=12: mu=np.mean(spreads); std=np.std(spreads); pair_ir[(fa,fb)]=mu/std if std>0 else 0
                sorted_pairs=sorted(pair_ir.items(),key=lambda x:x[1],reverse=True)
                fold_pairs[test_yr]=[p for p,ir in sorted_pairs[:4]]

        for test_yr in range(FY,YEARS[-1]+1):
            te_dt=pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=2)
            ts_dt=pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
            tss=pd.Timestamp('%d-01-01'%test_yr); tee=pd.Timestamp('%d-12-31'%test_yr)

            if STRAT=='ETF':
                tr=mt[(mt['trade_date']>=ts_dt)&(mt['trade_date']<te_dt)]
                te=mt[(mt['trade_date']>=tss)&(mt['trade_date']<=tee)]
                if len(tr)<500 or len(te)<50: continue
                iw={}
                for f in MOM:
                    ics=[]
                    for rd in sorted(tr['trade_date'].unique()):
                        if rd not in tg: continue
                        v=tr[(tr['trade_date']==rd)&(tr[f].notna())]
                        if len(v)<8: continue
                        d=v.set_index('industry'); fw=tg[rd]; cm=d.index.intersection(fw.index)
                        if len(cm)<8: continue
                        ic=d.loc[cm,f].rank().corr(fw[cm].rank())
                        if not np.isnan(ic): ics.append(ic)
                    iw[f]=abs(np.nanmean(ics)) if ics else 0.05
                tw=sum(iw.values()) or 1
                for f in iw: iw[f]/=tw
            else:
                if test_yr not in fold_pairs: continue
                top4=fold_pairs[test_yr]

            for rd in sorted((te if STRAT=='ETF' else fn[fn['trade_date'].between(tss,tee)])['trade_date'].unique() if STRAT=='ETF' else [d for d in md if d.year==test_yr]):
                if STRAT=='ETF' and rd not in tg: continue
                if STRAT=='小众' and rd not in rd_map: continue

                pos,state=gate(rd,state)
                ms=macro_score(rd)
                # 宏观叠加: 调整仓位±30%
                if MACRO_ON and pos>0.01:
                    pos=pos*(0.7+0.6*ms)  # ms=0→70%仓位, ms=1→130%仓位(上限100%)
                    pos=min(1.0,pos)

                if pos<0.01: all_r.append(0.0); continue

                if STRAT=='小众':
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
                    top=day.nlargest(30,'score')
                    if len(top)<15: continue
                    all_r.append((top['fwd_ret'].mean()-COST)*pos)
                else:
                    day=te[te['trade_date']==rd].set_index('industry'); fw=tg[rd]
                    day=day.dropna(subset=MOM)
                    if len(day)<8: continue
                    day['score']=sum(iw[f]*day[f].rank(pct=True) for f in MOM if f in day.columns)
                    top=day.nlargest(5,'score'); cm=top.index.intersection(fw.index)
                    if len(cm)>=3: all_r.append((fw[cm].mean()-COST)*pos)

        if len(all_r)<30: continue
        ra=np.array(all_r); ann=np.mean(ra)*12; vol=np.std(ra)*np.sqrt(12)
        sh=ann/vol if vol>0 else 0; mdd=np.min(np.cumprod(1+ra)/np.maximum.accumulate(np.cumprod(1+ra))-1)
        total=np.prod(1+ra)-1
        print('%s: ann=%+.1f%% sh=%+.2f mdd=%.1f%% total=%+.0f%%'%(label,ann*100,sh,mdd*100,total*100))

print('\n耗时: %.0fs'%(time.time()-t0))
