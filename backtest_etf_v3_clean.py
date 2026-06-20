# -*- coding: utf-8 -*-
"""ETF v3 · 弹性仓位升级 · 从v2已验证管道改造"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0=time.time()
COST=0.003; TRAIN_YEARS=5; EXIT_THRESH=-0.12; REENTRY_THRESH=0.10; FLOOR=0.10

print("="*60); print("ETF v3 · 弹性仓位 · Walk-Forward"); print("="*60)

# 数据(复用v2逻辑)
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
ind=con.execute("SELECT industry,trade_date,close FROM proxy_industry_daily WHERE trade_date>='2005-01-01' ORDER BY industry,trade_date").df()
ind['trade_date']=pd.to_datetime(ind['trade_date'])
hs300=con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date'])
hs300['ma50']=hs300['close'].rolling(50).mean(); hs300['high_2y']=hs300['close'].rolling(504).max()
hs300['low_1y']=hs300['close'].rolling(252).min()
con.close()

dates=sorted(ind['trade_date'].unique()); md=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

# HS300信号(完全复用v2)
hs_m={}
for d in md:
    row=hs300[hs300['trade_date']==d]
    if len(row)>0: r=row.iloc[0]; hs_m[d]={'c':r['close'],'m50':r['ma50'],'h2y':r['high_2y'],'l1y':r['low_1y']}
    else:
        nb=hs300[hs300['trade_date']<=d]
        if len(nb)>0: r=nb.iloc[-1]; hs_m[d]={'c':r['close'],'m50':r['ma50'],'h2y':r['high_2y'],'l1y':r['low_1y']}

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

# 因子(复用v2)
ind=ind.sort_values(['industry','trade_date'])
ind['ret_1d']=ind.groupby('industry')['close'].pct_change()
ind['ret_5d']=ind.groupby('industry')['close'].pct_change(5)
ind['ret_20d']=ind.groupby('industry')['close'].pct_change(20)
ind['ret_60d']=ind.groupby('industry')['close'].pct_change(60)
ind['vol_20d']=ind.groupby('industry')['ret_1d'].transform(lambda x:x.rolling(20).std())
ind['high20']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).max())
ind['low20']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).min())
ma60=ind.groupby('industry')['close'].transform(lambda x:x.rolling(60).mean()); ind['div_ma60']=ind['close']/ma60-1

mt=ind[ind['trade_date'].isin(md)].copy()
mt['mom_20']=mt['ret_20d']; mt['mom_60']=mt['ret_60d']; mt['rev_5']=-mt['ret_5d']
mt['crowd']=-mt['vol_20d']; pr=mt['high20']-mt['low20']; mt['rps']=mt['ret_20d']/pr.replace(0,0.01)
MOM=['mom_20','mom_60','rev_5','crowd','rps','div_ma60']

# 目标(复用v2)
tg={}
for i in range(len(md)-1):
    cur=md[i]; nxt=md[i+1]
    cd=ind[ind['trade_date']==cur]; nd=ind[ind['trade_date']==nxt]
    if len(cd)<10 or len(nd)<10: continue
    tg[cur]=nd.set_index('industry')['close']/cd.set_index('industry')['close']-1

# Walk-Forward
print("[1] WF...")
YEARS=sorted(set(d.year for d in md)); FY=YEARS[0]+TRAIN_YEARS+1
all_r=[]; state={'in':True,'exit':None}

for test_yr in range(FY,YEARS[-1]+1):
    te_dt=pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=2)
    ts_dt=pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    tss=pd.Timestamp('%d-01-01'%test_yr); tee=pd.Timestamp('%d-12-31'%test_yr)
    tr=mt[(mt['trade_date']>=ts_dt)&(mt['trade_date']<te_dt)]
    te=mt[(mt['trade_date']>=tss)&(mt['trade_date']<=tee)]
    if len(tr)<500 or len(te)<50: continue

    # IC权重
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

    fold_r=[]
    for rd in sorted(te['trade_date'].unique()):
        if rd not in tg: continue
        pos,state=gate(rd,state)
        if pos<0.01: fold_r.append(0.0); all_r.append({'date':str(rd)[:7],'ret':0.0,'yr':rd.year}); continue

        day=te[te['trade_date']==rd].set_index('industry'); fw=tg[rd]

        # 得分
        day=day.dropna(subset=MOM)
        if len(day)<8: continue
        day['score']=sum(iw[f]*day[f].rank(pct=True) for f in MOM if f in day.columns)

        # === 弹性仓位: 唯一改动 ===
        if pos>=0.9:    # FULL: 集中进攻Top3
            n,tw_v=3,[0.40,0.30,0.30]
        elif pos>=0.5:  # CAUTION: 分散Top5
            n,tw_v=5,[0.25,0.20,0.20,0.20,0.15]
        else:           # REDUCE: 小仓Top3
            n,tw_v=3,[0.40,0.30,0.30]

        top=day.nlargest(n,'score'); cm=top.index.intersection(fw.index)
        if len(cm)<max(2,n//2): continue
        wr=sum(tw_v[j]*fw[top.index[j]] for j in range(min(n,len(top.index))) if top.index[j] in fw.index)
        sw=sum(tw_v[j] for j in range(min(n,len(top.index))) if top.index[j] in fw.index)
        if sw>0: fold_r.append((wr/sw-COST)*pos); all_r.append({'date':str(rd)[:7],'ret':(wr/sw-COST)*pos,'yr':rd.year,'pos':pos})

    if fold_r:
        r=np.array(fold_r); a=np.mean(r)*12; v=np.std(r)*np.sqrt(12)
        s=a/v if v>0 else 0; mdd=np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%% active=%d"%(test_yr,a*100,s,mdd*100,len(fold_r)))

# 总结
if len(all_r)>10:
    ra=np.array([x['ret'] for x in all_r])
    ann=np.mean(ra)*12; vol=np.std(ra)*np.sqrt(12); sh=ann/vol if vol>0 else 0
    mdd=np.min(np.cumprod(1+ra)/np.maximum.accumulate(np.cumprod(1+ra))-1)
    total=np.prod(1+ra)-1
    print("\n"+"="*60)
    print("ETF v3 终验")
    print("年化:%+.1f%% Sharpe:%+.2f MDD:%.1f%% 累计:%+.0f%%"%(ann*100,sh,mdd*100,total*100))

    # 对比
    print("\n=== ETF进化 ===")
    print("v1(MA200等权):    年化+1.6%%  Sharpe0.10  MDD-58.3%%")
    print("v2(多因子Top5):   年化+7.9%%  Sharpe0.44  MDD-45.6%%")
    print("v3(弹性仓位Top3/5):年化%+.1f%%  Sharpe%+.2f  MDD%.1f%%"%(ann*100,sh,mdd*100))

print("\n耗时:%.0fs"%(time.time()-t0))
