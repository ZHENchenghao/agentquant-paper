# -*- coding: utf-8 -*-
"""Man Group相似性检测 v2 · FRED完整数据版"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0=time.time()
K=5; FWD=3; ROLL=60

print("="*60); print("Man Group v2 · FRED完整数据"); print("="*60)

# ===== 加载全部数据 =====
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)
hs300=con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date']=pd.to_datetime(hs300['trade_date']); hs300=hs300.set_index('trade_date')['close']
macro=con.execute("SELECT trade_date,vix,usdcny,m1_growth,m2_growth FROM macro_indicators ORDER BY trade_date").df()
macro['trade_date']=pd.to_datetime(macro['trade_date']); macro=macro.set_index('trade_date')
nb=con.execute("SELECT trade_date,net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date']=pd.to_datetime(nb['trade_date']); nb=nb.set_index('trade_date')['net_flow']
mg=con.execute("SELECT trade_date,margin_balance FROM margin_trading ORDER BY trade_date").df()
mg['trade_date']=pd.to_datetime(mg['trade_date']); mg=mg.set_index('trade_date')['margin_balance']
con.close()

# FRED数据
fred=pd.read_csv('D:/AgentQuant/our/cache/macro_fred.csv',parse_dates=['DATE']).set_index('DATE')

# ===== 月度 =====
dates=sorted(set(hs300.index)|set(macro.index)|set(fred.index))
md=[];
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): md.append(g.iloc[0])
md=sorted(md)

def align(s,mdates):
    s=s.sort_index().dropna(); r={}
    for d in mdates:
        v=s[s.index<=d]
        if len(v)>0: r[d]=v.iloc[-1]
    return pd.Series(r)

# 7变量状态向量(现在有完整数据)
sv={}
sv['HS300_mom']=align(hs300.pct_change(20),md)
sv['VIX']=align(fred['VIX_fred'],md)  # FRED版VIX, 2000起
sv['US10Y']=align(fred['US10Y'],md)
sv['WTI']=align(fred['WTI'],md)
sv['DXY']=align(fred['DXY'],md)
sv['M1M2']=align(macro['m1_growth']-macro['m2_growth'],md)
sv['Vol']=align(hs300.pct_change().rolling(20).std()*np.sqrt(252),md)

st=pd.DataFrame(sv).dropna()
print("有效月度: %d (%s~%s)"%(len(st),st.index[0].date(),st.index[-1].date()))

# ===== 相似性匹配+Walk-Forward =====
VARS=list(st.columns)
FW_YEARS=5; FY=sorted(set(d.year for d in st.index))[0]+FW_YEARS+1
print("WF: %d-%d"%(FY,sorted(set(d.year for d in st.index))[-1]))

all_signals={}; all_actual={}
for test_yr in range(FY,sorted(set(d.year for d in st.index))[-1]+1):
    train_end=pd.Timestamp('%d-01-01'%test_yr)
    test_start=pd.Timestamp('%d-01-01'%test_yr); test_end=pd.Timestamp('%d-12-31'%test_yr)

    te_dates=[d for d in st.index if test_start<=d<=test_end]
    tr_dates=[d for d in st.index if d<train_end]

    for today in te_dates:
        today_idx=list(st.index).index(today)
        # 只用训练期数据做参考
        hist=st.loc[[d for d in tr_dates if d<today]]
        if len(hist)<ROLL: continue

        # z-score用训练期滚动窗
        mu=st.iloc[max(0,today_idx-ROLL):today_idx].mean()
        std=st.iloc[max(0,today_idx-ROLL):today_idx].std().replace(0,1)
        today_z=(st.iloc[today_idx]-mu)/std
        hist_z=(hist-mu)/std

        # 欧氏距离
        dist=np.sqrt(((hist_z-today_z.values)**2).sum(axis=1))
        if len(dist)<K: continue
        nearest=dist.nsmallest(K).index

        # 这些相似月之后的FWD月收益
        fwd_rets=[]
        for hd in nearest:
            hp=list(st.index).index(hd)
            if hp+FWD<today_idx:
                fwd_date=st.index[hp+FWD]; hist_date=st.index[hp]
                # 用align处理日期不匹配
                hs_fwd=hs300[hs300.index<=fwd_date].iloc[-1] if len(hs300[hs300.index<=fwd_date])>0 else hs300.iloc[-1]
                hs_hist=hs300[hs300.index<=hist_date].iloc[-1] if len(hs300[hs300.index<=hist_date])>0 else hs300.iloc[-1]
                fwd_rets.append(hs_fwd/hs_hist-1)
        if fwd_rets: all_signals[today]=np.mean(fwd_rets)

    # 实际收益
    for i in range(len(st.index)-FWD):
        today=st.index[i]
        future=st.index[i+FWD]
        if today in all_signals and today.year==test_yr:
            hs_f=hs300[hs300.index<=future].iloc[-1] if len(hs300[hs300.index<=future])>0 else hs300.iloc[-1]
            hs_t=hs300[hs300.index<=today].iloc[-1] if len(hs300[hs300.index<=today])>0 else hs300.iloc[-1]
            all_actual[today]=hs_f/hs_t-1

# ===== 评估 =====
common=sorted(set(all_signals.keys())&set(all_actual.keys()))
pred=np.array([all_signals[d] for d in common])
actual=np.array([all_actual[d] for d in common])
ic=np.corrcoef(pred,actual)[0,1]; dh=np.mean((pred>0)==(actual>0))

# 分桶单调性
buckets=[(0,25),(25,50),(50,75),(75,100)]
print("\n"+"="*60)
print("相似性检测 v2 评估")
print("="*60)
print("有效信号: %d月 | IC: %.3f | 方向命中: %.1f%%"%(len(common),ic,dh*100))
print("\n分桶单调性检验:")
prev=None; mono=True
for lo,hi in buckets:
    mask=(pred>=np.percentile(pred,lo))&(pred<np.percentile(pred,hi)) if hi<100 else (pred>=np.percentile(pred,lo))
    if mask.sum()>0:
        r=np.mean(actual[mask])*100; n=mask.sum()
        print("  %d-%d%%: %+.1f%% (%d月)"%(lo,hi,r,n))
        if prev is not None and r<prev: mono=False
        prev=r

# DD_SMART相关性
dd_st=align(hs300/hs300.rolling(504).max()-1,md)
com_dd=sorted(set(all_signals.keys())&set(dd_st.dropna().index))
sig_dd_corr=np.corrcoef([all_signals[d] for d in com_dd],[dd_st[d] for d in com_dd])[0,1]
print("\nvs DD_SMART相关: %.3f %s"%(sig_dd_corr,'互补✅' if abs(sig_dd_corr)<0.2 else '冗余⚠️'))
print("单调性: %s"%('通过✅' if mono else '不通过⚠️'))
print("\n耗时: %.0fs"%(time.time()-t0))
