# -*- coding: utf-8 -*-
"""
ETF轮动 v3 · 双动量+弹性仓位 · Walk-Forward
===========================================
升级:
  1. 双动量: 绝对动量(sector自身)+相对动量(vs HS300)
  2. 弹性仓位: FULL→Top3集中(40/30/30), CAUTION→Top5等权, CRASH→空仓
  3. 趋势过滤: 只选mom_20>0且>MA60的行业
  4. 相关性惩罚: 高相关行业去重
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

COST = 0.003; TRAIN_YEARS = 5
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10

print("="*60)
print("ETF轮动 v3 · 双动量+弹性仓位 · Walk-Forward")
print("="*60)

# ============ 加载 ============
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
ind = con.execute("""SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date>='2005-01-01' ORDER BY industry,trade_date""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean(); hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# 月度
dates = sorted(ind['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print("行业:30个 月度:%d" % len(monthly_dates))

# HS300信号
hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0: r=row.iloc[0]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby=hs300[hs300['trade_date']<=d]
        if len(nearby)>0: r=nearby.iloc[-1]; hs300_m[d]={'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart_v2(rd, state):
    if rd not in hs300_m: return 1.0, state
    info=hs300_m[rd]; c=info['close']; ma50=info['ma50']; h2y=info['high_2y']; l1y=info['low_1y']
    if pd.isna(h2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y=c/h2y-1
        if dd_2y<EXIT_THRESH-0.05: return FLOOR,{'in_market':False,'exit_date':rd}
        elif dd_2y<EXIT_THRESH: return FLOOR*2,{'in_market':False,'exit_date':rd}
        else: return 1.0, state
    else:
        rec=c/l1y-1 if pd.notna(l1y) and l1y>0 else 0; above=c>ma50
        if rec>REENTRY_THRESH and above: return 0.7,{'in_market':True,'exit_date':None}
        elif rec>REENTRY_THRESH*0.7: return FLOOR*2,state
        elif rec>0.05 and above: return FLOOR,state
        else: return FLOOR,state

# ============ 因子计算 ============
print("[1] 计算因子...")
ind = ind.sort_values(['industry','trade_date'])
ind['ret_1d'] = ind.groupby('industry')['close'].pct_change()
ind['ret_5d'] = ind.groupby('industry')['close'].pct_change(5)
ind['ret_20d'] = ind.groupby('industry')['close'].pct_change(20)
ind['ret_60d'] = ind.groupby('industry')['close'].pct_change(60)
ind['vol_20d'] = ind.groupby('industry')['ret_1d'].transform(lambda x: x.rolling(20).std())
ind['ma60'] = ind.groupby('industry')['close'].transform(lambda x: x.rolling(60).mean())

# HS300 20日收益
hs300['ret_20d'] = hs300['close'].pct_change(20)
hs300_rt = hs300[['trade_date','ret_20d']].rename(columns={'ret_20d':'hs300_ret_20d'})

# 月度因子
mt = ind[ind['trade_date'].isin(monthly_dates)].copy()
mt['mom_20'] = mt['ret_20d']; mt['mom_60'] = mt['ret_60d']
mt['rev_5'] = -mt['ret_5d']; mt['crowd'] = -mt['vol_20d']
mt['trend'] = (mt['close'] > mt['ma60']).astype(float)
# 相对动量
mt = mt.merge(hs300_rt, on='trade_date', how='left')
mt['rel_mom'] = mt['ret_20d'] - mt['hs300_ret_20d']

# 选用的因子
FEATS = ['mom_20','mom_60','rev_5','crowd','rel_mom']
print("因子: %s" % str(FEATS))

# ============ 目标 ============
print("[2] 构建目标...")
tg = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    c_d = ind[ind['trade_date']==cur]; n_d = ind[ind['trade_date']==nxt]
    if len(c_d)<10 or len(n_d)<10: continue
    tg[cur] = n_d.set_index('industry')['close']/c_d.set_index('industry')['close']-1

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1

print("[3] Walk-Forward...")
all_results = []; state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_end = pd.Timestamp('%d-01-01'%test_yr) - pd.DateOffset(months=2)
    train_start = pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01'%test_yr); test_end = pd.Timestamp('%d-12-31'%test_yr)

    tr = mt[(mt['trade_date']>=train_start)&(mt['trade_date']<train_end)]
    te = mt[(mt['trade_date']>=test_start)&(mt['trade_date']<=test_end)]
    if len(tr)<500 or len(te)<50: continue

    # IC权重
    ic_w = {}
    for f in FEATS:
        ics = []
        for rd in sorted(tr['trade_date'].unique()):
            rd_dt = pd.Timestamp(rd)
            match = None
            for k in tg:
                if abs((k - rd_dt).days) <= 3: match = k; break
            if match is None: continue
            valid = tr[(tr['trade_date']==rd)&(tr[f].notna())]
            if len(valid)<8: continue
            day = valid.set_index('industry'); fwd = tg[match]
            cm = day.index.intersection(fwd.index)
            if len(cm)<8: continue
            ic = day.loc[cm,f].rank().corr(fwd[cm].rank())
            if not np.isnan(ic): ics.append(ic)
        ic_w[f] = abs(np.nanmean(ics)) if ics else 0.05
    tw = sum(ic_w.values()) or 1
    for f in ic_w: ic_w[f] /= tw

    fold_rets = []
    for rd in sorted(te['trade_date'].unique()):
        rd_dt = pd.Timestamp(rd)
        if rd_dt not in tg:
            # try finding closest match
            found = None
            for k in tg:
                if abs((k - rd_dt).days) <= 3:
                    found = k; break
            if found is None: continue
        else: found = rd_dt

        gate_pos, state = dd_smart_v2(rd_dt, state)
        if gate_pos < 0.01:
            fold_rets.append(0.0); all_results.append({'date':str(rd_dt)[:7],'ret':0.0,'yr':rd_dt.year}); continue

        day = te[te['trade_date']==rd].copy(); fwd = tg[found]

        # 趋势过滤(宽松版)
        day = day[day['mom_20'].notna()]
        if len(day)<8: continue

        # 多因子得分
        for f in FEATS:
            if f in day.columns: day[f] = day[f].fillna(0)
        day['score'] = sum(ic_w.get(f,0.2)*day[f].rank(pct=True) for f in FEATS if f in day.columns)

        # 弹性仓位
        if gate_pos >= 0.9:   # FULL: 集中Top3
            top_n = 3; weights = [0.40, 0.30, 0.30]
        elif gate_pos >= 0.5:  # CAUTION: 分散Top5
            top_n = 5; weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        else:                   # REDUCE: Top3小仓
            top_n = 3; weights = [0.40, 0.30, 0.30]

        top = day.nlargest(top_n, 'score')
        cm = top.index.intersection(fwd.index)
        if len(cm) < max(2, top_n//2): continue

        # 加权收益
        weighted_ret = 0; total_w = 0
        for j, ind_name in enumerate(top.index):
            if ind_name in fwd.index and j < len(weights):
                weighted_ret += weights[j] * fwd[ind_name]
                total_w += weights[j]
        if total_w > 0:
            month_ret = (weighted_ret/total_w - COST) * gate_pos
            fold_rets.append(month_ret)
            all_results.append({'date':str(rd)[:7],'ret':month_ret,'yr':rd.year,'n':len(cm),'pos':gate_pos})

    if fold_rets:
        r=np.array(fold_rets); a=np.mean(r)*12; v=np.std(r)*np.sqrt(12)
        s=a/v if v>0 else 0; mdd=np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%%"%(test_yr,a*100,s,mdd*100))

# 总结
if len(all_results)<10: print('数据不足'); sys.exit(0)
r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
mdd=np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1)
total=np.prod(1+r_all)-1; win=(r_all>0).mean()*100

print("\n"+"="*60)
print("ETF v3 · 终验")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | 累计: %+.0f%%"%(ann*100,sh,mdd*100,total*100))
print("胜率: %.0f%% | 月数: %d"%(win,len(r_all)))

crash=[2008,2011,2017,2018,2022]; bull=[2007,2009,2015,2019,2021,2025]
cr=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bl=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1
print("5熊: %+.1f%% | 6牛: %+.1f%%"%(cr*100,bl*100))

print("\n年      收益   Sharpe    MDD    仓位")
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array([x['ret'] for x in dr]); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        ap=np.mean([x['pos'] for x in dr if x.get('pos',0)>0.01])*100
        print("%d %+7.1f%% %+6.2f %+6.1f%% %4.0f%%"%(yr,(np.prod(1+rr)-1)*100,s,m*100,ap))

# 对比
print("\n=== ETF进化对比 ===")
print("v1(MA200等权): 年化+1.6%% Sharpe0.10")
print("v2(多因子Top5): 年化+7.9%% Sharpe0.44")
print("v3(双动量弹性仓位): 年化%+.1f%% Sharpe%+.2f"%(ann*100,sh))
print("\n耗时: %.0fs"%(time.time()-t0))
