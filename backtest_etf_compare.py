# -*- coding: utf-8 -*-
"""ETF旧版vs新版对比"""
import duckdb, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
ind = con.execute("SELECT industry, trade_date, close FROM proxy_industry_daily WHERE trade_date>='2005-01-01' ORDER BY industry,trade_date").df()
ind['trade_date']=pd.to_datetime(ind['trade_date'])
hs300 = con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date'])
con.close()

dates=sorted(ind['trade_date'].unique()); md=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

# 目标
tg={}
for i in range(len(md)-1):
    cur=md[i]; nxt=md[i+1]
    c_d=ind[ind['trade_date']==cur]; n_d=ind[ind['trade_date']==nxt]
    if len(c_d)<10 or len(n_d)<10: continue
    tg[cur]=n_d.set_index('industry')['close']/c_d.set_index('industry')['close']-1

# HS300信号
hs_m={}
for d in md:
    row=hs300[hs300['trade_date']==d]
    if len(row)>0: hs_m[d]=row['close'].iloc[0]
    else:
        nb=hs300[hs300['trade_date']<=d]
        if len(nb)>0: hs_m[d]=nb['close'].iloc[-1]

px=hs300.set_index('trade_date')['close']
COST=0.003; EXIT_THRESH=-0.12; REENTRY_THRESH=0.10; FLOOR=0.10

# 因子
ind=ind.sort_values(['industry','trade_date'])
ind['ret_1d']=ind.groupby('industry')['close'].pct_change()
ind['ret_5d']=ind.groupby('industry')['close'].pct_change(5)
ind['ret_20d']=ind.groupby('industry')['close'].pct_change(20)
ind['ret_60d']=ind.groupby('industry')['close'].pct_change(60)
ind['vol_20d']=ind.groupby('industry')['ret_1d'].transform(lambda x:x.rolling(20).std())
ind['high_20d']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).max())
ind['low_20d']=ind.groupby('industry')['close'].transform(lambda x:x.rolling(20).min())
ma60=ind.groupby('industry')['close'].transform(lambda x:x.rolling(60).mean())
ind['div_ma60']=ind['close']/ma60-1

mt=ind[ind['trade_date'].isin(md)].copy()
mt['mom_20']=mt['ret_20d']; mt['mom_60']=mt['ret_60d']; mt['rev_5']=-mt['ret_5d']
mt['crowd']=-mt['vol_20d']; pr=mt['high_20d']-mt['low_20d']; mt['rps']=mt['ret_20d']/pr.replace(0,0.01)
MOM=['mom_20','mom_60','rev_5','crowd','rps','div_ma60']

old_r=[]; new_r=[]
state={'in_market':True,'exit_date':None}
FIRST_TEST_YR=sorted(set(d.year for d in md))[0]+5+1

for test_yr in range(FIRST_TEST_YR,sorted(set(d.year for d in md))[-1]+1):
    t_end=pd.Timestamp('%d-01-01'%test_yr)-pd.DateOffset(months=2)
    t_start=pd.Timestamp('%d-01-01'%(test_yr-5))
    ts_pd=pd.Timestamp('%d-01-01'%test_yr); te_pd=pd.Timestamp('%d-12-31'%test_yr)
    tr=mt[(mt['trade_date']>=t_start)&(mt['trade_date']<t_end)]
    ts_df=mt[(mt['trade_date']>=ts_pd)&(mt['trade_date']<=te_pd)]
    if len(tr)<500 or len(ts_df)<50: continue

    ic_w={f:abs(np.nanmean([tr[(tr['trade_date']==rd)&(tr[f].notna())].set_index('industry')[f].rank().corr(
        tg[rd][tr[(tr['trade_date']==rd)&(tr[f].notna())].set_index('industry').index.intersection(tg[rd].index)].rank())
        for rd in sorted(tr['trade_date'].unique()) if rd in tg and len(
            tr[(tr['trade_date']==rd)&(tr[f].notna())].set_index('industry').index.intersection(tg[rd].index))>=8
    ])) for f in MOM}
    tw=sum(ic_w.values()) or 1
    for f in ic_w: ic_w[f]=max(ic_w[f],0.05)/tw  # floor 5%

    for rd in sorted(ts_df['trade_date'].unique()):
        if rd not in tg or rd not in hs_m: continue

        # DD_SMART
        c=hs_m[rd]; ma50_px=px[px.index<=rd].tail(50).mean()
        h2y_px=px[px.index<=rd].tail(504).max(); l1y_px=px[px.index<=rd].tail(252).min()
        if state['in_market']:
            dd_2y=c/h2y_px-1
            if dd_2y<EXIT_THRESH-0.05: pos,state=FLOOR,{'in_market':False,'exit_date':rd}
            elif dd_2y<EXIT_THRESH: pos,state=FLOOR*2,{'in_market':False,'exit_date':rd}
            else: pos=1.0
        else:
            rec=c/l1y_px-1; above=c>ma50_px
            if rec>REENTRY_THRESH and above: pos,state=0.7,{'in_market':True,'exit_date':None}
            elif rec>REENTRY_THRESH*0.7: pos=FLOOR*2
            elif rec>0.05 and above: pos=FLOOR
            else: pos=FLOOR

        day=ts_df[ts_df['trade_date']==rd].set_index('industry'); fwd=tg[rd]
        px200=px[px.index<=rd].tail(200).mean()

        # OLD: MA200等权全行业
        old_r.append((fwd.mean()-COST) if c>px200 else 0.0)

        # NEW: Top5
        if pos<0.01: new_r.append(0.0); continue
        day['score']=sum(ic_w[f]*day[f].rank(pct=True) for f in MOM if f in day.columns)
        top5=day.nlargest(5,'score'); cm=top5.index.intersection(fwd.index)
        new_r.append((fwd[cm].mean()-COST)*pos if len(cm)>=3 else 0.0)

oa,na=np.array(old_r),np.array(new_r)
o_ann=np.mean(oa)*12; n_ann=np.mean(na)*12
o_vol=np.std(oa)*np.sqrt(12); n_vol=np.std(na)*np.sqrt(12)
o_sh=o_ann/o_vol if o_vol>0 else 0; n_sh=n_ann/n_vol if n_vol>0 else 0
o_mdd=np.min(np.cumprod(1+oa)/np.maximum.accumulate(np.cumprod(1+oa))-1)
n_mdd=np.min(np.cumprod(1+na)/np.maximum.accumulate(np.cumprod(1+na))-1)

print('=== ETF新旧对比 ===')
print('旧MA200等权全行业: 年化%+.1f%% Sharpe%+.2f MDD%.1f%% 累计%+.0f%%'%(o_ann*100,o_sh,o_mdd*100,(np.prod(1+oa)-1)*100))
print('新多因子Top5行业: 年化%+.1f%% Sharpe%+.2f MDD%.1f%% 累计%+.0f%%'%(n_ann*100,n_sh,n_mdd*100,(np.prod(1+na)-1)*100))
print('月数: %d'%len(oa))

# 分年
print('\n年    旧MA200  新多因子')
for yr in range(FIRST_TEST_YR,sorted(set(d.year for d in md))[-1]+1):
    oy=oa[[i for i,x in enumerate(md) if x.year==yr and i<len(oa)]]
    ny=na[[i for i,x in enumerate(md) if x.year==yr and i<len(na)]]
    if len(oy)>=6:
        print('%d %+7.1f%% %+7.1f%%'%(yr,(np.prod(1+oy)-1)*100,(np.prod(1+ny)-1)*100))
