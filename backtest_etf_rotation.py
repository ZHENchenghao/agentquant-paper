# -*- coding: utf-8 -*-
"""
ETF行业轮动 v2 · 多因子动量 · Walk-Forward
===========================================
升级: 单一MA200 → 多因子行业打分(动量+拥挤度+情绪)
参考: 开源证券行业轮动3.0 + 申万技术面轮动
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 5; COST = 0.003; TRAIN_YEARS = 5
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10

print("="*60)
print("ETF行业轮动 v2 · 多因子 · Walk-Forward")
print("="*60)

# ============ 加载 ============
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 行业数据
ind = con.execute("""SELECT industry, stock_code, trade_date, close
    FROM proxy_industry_daily WHERE trade_date >= '2005-01-01'
    ORDER BY industry, trade_date""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])
print("[1] 行业: %d个, %d行" % (ind['industry'].nunique(), len(ind)))

# HS300
hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean(); hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# 月度调仓日
dates = sorted(ind['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]): monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print("月度: %d" % len(monthly_dates))

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

# ============ 月度因子计算 ============
print("[2] 计算月度因子...")

# 先算日频因子
ind = ind.sort_values(['industry','trade_date'])
ind['ret_1d'] = ind.groupby('industry')['close'].pct_change()
ind['ret_5d'] = ind.groupby('industry')['close'].pct_change(5)
ind['ret_20d'] = ind.groupby('industry')['close'].pct_change(20)
ind['ret_60d'] = ind.groupby('industry')['close'].pct_change(60)
ind['vol_20d'] = ind.groupby('industry')['ret_1d'].transform(lambda x: x.rolling(20).std())
ind['high_20d'] = ind.groupby('industry')['close'].transform(lambda x: x.rolling(20).max())
ind['low_20d'] = ind.groupby('industry')['close'].transform(lambda x: x.rolling(20).min())

# 月度汇总: 取每月第一个交易日
monthly_ind = ind[ind['trade_date'].isin(monthly_dates)].copy()

# 动量因子(月频):
# 1. 中期动量: 20日收益 (多头, 开源/天相均验证)
monthly_ind['mom_20'] = monthly_ind['ret_20d']
# 2. 长期动量: 60日收益
monthly_ind['mom_60'] = monthly_ind['ret_60d']
# 3. 短期反转: -5日收益 (A股5日内反转)
monthly_ind['rev_5'] = -monthly_ind['ret_5d']
# 4. 拥挤度: -20日波动率 (高波动=拥挤, 负向)
monthly_ind['crowd'] = -monthly_ind['vol_20d']
# 5. 相对强度: 20日收益/(20日最大-最小)
price_range = monthly_ind['high_20d'] - monthly_ind['low_20d']
monthly_ind['rps'] = monthly_ind['ret_20d'] / price_range.replace(0, 0.01)
# 6. MA偏离: 收盘/MA60-1
ma60 = ind.groupby('industry')['close'].transform(lambda x: x.rolling(60).mean())
ind['div_ma60'] = ind['close']/ma60 - 1
monthly_ma = ind[ind['trade_date'].isin(monthly_dates)][['industry','trade_date','div_ma60']]
monthly_ind = monthly_ind.merge(monthly_ma, on=['industry','trade_date'], how='left')

MOM_FEATS = ['mom_20','mom_60','rev_5','crowd','rps','div_ma60']
print("因子: %s" % str(MOM_FEATS))

# ============ 构建目标 ============
print("[3] 构建月度目标...")
# 每月选Top5行业, 等权持有到下月, 收益=下月行业收益均值
monthly_ind = monthly_ind.sort_values(['trade_date','industry'])
targets = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cur_data = monthly_ind[monthly_ind['trade_date']==cur]
    nxt_data = monthly_ind[monthly_ind['trade_date']==nxt]
    if len(cur_data)<10 or len(nxt_data)<10: continue
    fwd = nxt_data.set_index('industry')['close'] / cur_data.set_index('industry')['close'] - 1
    targets[cur] = fwd

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1

print("[4] Walk-Forward...")
all_results = []; state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_end = pd.Timestamp('%d-01-01'%test_yr) - pd.DateOffset(months=2)
    train_start = pd.Timestamp('%d-01-01'%(test_yr-TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01'%test_yr); test_end = pd.Timestamp('%d-12-31'%test_yr)

    tr = monthly_ind[(monthly_ind['trade_date']>=train_start)&(monthly_ind['trade_date']<train_end)].copy()
    te = monthly_ind[(monthly_ind['trade_date']>=test_start)&(monthly_ind['trade_date']<=test_end)].copy()
    if len(tr) < 500 or len(te) < 50: continue

    # 训练期: 计算每个因子的IC(与下月收益的截面相关性)
    ic_weights = {}
    for f in MOM_FEATS:
        ics = []
        train_dates = sorted(tr['trade_date'].unique())
        for rd in train_dates:
            if rd not in targets: continue
            day = tr[tr['trade_date']==rd].set_index('industry')
            fwd = targets[rd]
            common = day.index.intersection(fwd.index)
            if len(common) < 8: continue
            ic = day.loc[common, f].rank().corr(fwd[common].rank())
            if not np.isnan(ic): ics.append(ic)
        ic_weights[f] = abs(np.mean(ics)) if ics else 0

    # 归一化权重
    total_w = sum(ic_weights.values()) or 1
    for f in ic_weights: ic_weights[f] /= total_w

    # 测试期: 每月选Top5行业
    fold_rets = []
    for rd in sorted(te['trade_date'].unique()):
        if rd not in targets: continue
        pos, state = dd_smart_v2(rd, state)
        if pos < 0.01: fold_rets.append(0.0); all_results.append({'date':str(rd)[:7],'ret':0.0,'yr':rd.year}); continue

        day = te[te['trade_date']==rd].set_index('industry')
        fwd = targets[rd]

        # 多因子打分
        day['score'] = 0
        for f in MOM_FEATS:
            if f in day.columns:
                day['score'] += ic_weights.get(f, 1/len(MOM_FEATS)) * day[f].rank(pct=True)

        top5 = day.nlargest(TOP_N, 'score')
        common = top5.index.intersection(fwd.index)
        if len(common) < 3: continue

        month_ret = (fwd[common].mean() - COST) * pos
        fold_rets.append(month_ret)
        all_results.append({'date':str(rd)[:7],'ret':month_ret,'yr':rd.year,'n':len(common)})

    if fold_rets:
        r=np.array(fold_rets); a=np.mean(r)*12; v=np.std(r)*np.sqrt(12)
        s=a/v if v>0 else 0; mdd=np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        top_f = sorted(ic_weights.items(), key=lambda x:x[1], reverse=True)[:3]
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%% 因子:%s"%(test_yr,a*100,s,mdd*100,str([f[0][:4] for f in top_f])))

# 总结
r_all=np.array([x['ret'] for x in all_results])
ann=np.mean(r_all)*12; vol=np.std(r_all)*np.sqrt(12); sh=ann/vol if vol>0 else 0
mdd=np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1)
total=np.prod(1+r_all)-1; win=(r_all>0).mean()*100

print("\n"+"="*60)
print("ETF轮动 v2 · 终验")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | 累计: %+.0f%%"%(ann*100,sh,mdd*100,total*100))
print("胜率: %.0f%% | 月数: %d"%(win,len(r_all)))

crash=[2008,2011,2017,2018,2022]; bull=[2007,2009,2015,2019,2021,2025]
cr=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bl=np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1
print("5熊: %+.1f%% | 6牛: %+.1f%%"%(cr*100,bl*100))

print("\n年      收益   Sharpe    MDD")
for yr in range(FIRST_TEST_YR,YEARS[-1]+1):
    dr=[x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr=np.array(dr); a=np.mean(rr)*12; v=np.std(rr)*np.sqrt(12)
        s=a/v if v>0 else 0; m=np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        print("%d %+7.1f%% %+6.2f %+6.1f%%"%(yr,(np.prod(1+rr)-1)*100,s,m*100))

# 对比旧版ETF
print("\n=== 新旧ETF对比 ===")
print("旧版(MA200择时): 简单牛熊判断, 固定5只ETF")
print("新版(多因子轮动): 6因子行业打分, 每月选Top5行业, DD_SMART保护")
print("年化: %+.1f%% | 旧版预期: ~10%% (无严格回测)"% (ann*100))

print("\n耗时: %.0fs"%(time.time()-t0))
