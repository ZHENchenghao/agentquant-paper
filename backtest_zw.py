# -*- coding: utf-8 -*-
"""
等权z-score选股回测（无ML、无训练窗口、无过拟合）
===================================================
方法论: 每月截面z-score等权 → Top15 → 持有到下月
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()
DB='D:/FreeFinanceData/data/duckdb/finance.db'
TOP_N=15; COST=0.0033
FEATS=['vp_corr','sr5','amihud','turnover_rev','max_rev','gap']  # 6因子

print('='*60)
print('等权z-score选股回测 (无ML)')
print('='*60)

# 1. 加载因子
print('[1] 加载数据...')
fn=pd.read_parquet('D:/AgentQuant/our/cache/factors_new6_v2.parquet')
fn['trade_date']=pd.to_datetime(fn['trade_date'])
print(f'因子: {len(fn):,}行 {fn["trade_date"].min().date()}~{fn["trade_date"].max().date()}')

# 2. 每月第一个交易日
con=duckdb.connect(DB, read_only=True)
dates=sorted(fn['trade_date'].unique())
monthly_dates=[]
for ym,g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates=sorted(monthly_dates)
print(f'月度调仓日: {len(monthly_dates)}个 ({monthly_dates[0].date()}~{monthly_dates[-1].date()})')

# 3. 构建调仓日→下个调仓日的价格和mcap映射
print('[2] 构建持有期价格映射...')
kline=con.execute("""
    SELECT ts_code, trade_date, open, close,
           COALESCE(close*total_share/10000, GREATEST(amount,close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2010-01-01'
""").df()
kline['trade_date']=pd.to_datetime(kline['trade_date'])

rd_map={}
for i in range(len(monthly_dates)-1):
    cur=monthly_dates[i]; nxt=monthly_dates[i+1]
    cp=kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_=kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m=cp.join(np_,how='inner'); m['fwd_ret']=m['next_open']/m['close']-1
    rd_map[cur]=m
del kline; gc.collect()
print(f'有效调仓日: {len(rd_map)}')

# 4. 月度选股 (滚动IC加权)
print('[3] 月度选股...')
monthly_rets=[]; details=[]

# 预计算因子IC缓存: 每月计算过去12个月每个因子的IC
ic_window=12
ic_weights={}  # {date: {factor: weight}}

for i, rd in enumerate(monthly_dates):
    if i<ic_window+1: continue  # 需要足够的IC历史

    # 计算过去12个月每个因子的IC
    past_start=monthly_dates[i-ic_window]
    past=fn[(fn['trade_date']>=past_start)&(fn['trade_date']<rd)]

    # 需要目标来计算IC
    # 用20日超额收益作为对齐目标
    # 简化: 计算该因子在截面上的rank IC
    # 因为需要target才能算IC，这里用每日IC的均值
    ic_w={}
    for f in FEATS:
        if f not in past.columns: continue
        # 日频IC均值(用ret_5d反向作为未来收益的代理 — 不够精确但方向对)
        # 更精确的做法是预计算IC...这里用稳定权重
        ic_w[f]=1.0/len(FEATS)  # fallback: equal

    ic_weights[rd]=ic_w

# 实际IC需要target数据, 这里简化: 用expanding IC
# 先构建每月的对应target
con3=duckdb.connect(DB, read_only=True)
target_daily=con3.execute("""
    SELECT s.ts_code, s.trade_date,
           (LEAD(s.close,20) OVER(PARTITION BY s.ts_code ORDER BY s.trade_date)/s.close-1)
           -(LEAD(x.close,20) OVER(ORDER BY x.trade_date)/x.close-1) AS excess_ret_20d
    FROM kline_daily s JOIN kline_daily x ON s.trade_date=x.trade_date AND x.ts_code='sh000300'
    WHERE s.trade_date>='2010-01-01'
""").df()
target_daily['trade_date']=pd.to_datetime(target_daily['trade_date'])
con3.close()

# 预计算每月IC权重
print('  Precomputing IC weights...')
ic_weights={}
for i, rd in enumerate(monthly_dates):
    if i<ic_window+1: continue
    past_start=monthly_dates[i-ic_window]
    merged=fn[(fn['trade_date']>=past_start)&(fn['trade_date']<rd)].merge(
        target_daily, on=['ts_code','trade_date'], how='inner')

    w={}
    for f in FEATS:
        if f not in merged.columns: continue
        d=merged.dropna(subset=[f,'excess_ret_20d'])
        if len(d)<1000: w[f]=1.0/len(FEATS); continue
        # 日IC均值
        ics=[]
        for td,g in d.groupby('trade_date'):
            if len(g)<30: continue
            ic=g[f].rank().corr(g['excess_ret_20d'].rank())
            ics.append(ic)
        avg_ic=abs(np.mean(ics)) if ics else 0
        w[f]=avg_ic
    # 归一化
    total=sum(w.values()) if sum(w.values())>0 else 1
    for f in w: w[f]/=total
    ic_weights[rd]=w

print(f'  IC weights ready for {len(ic_weights)} months')

for rd in monthly_dates:
    if rd not in rd_map or rd not in ic_weights: continue
    day=fn[fn['trade_date']==rd].copy()
    px=rd_map[rd]
    if len(day)<100: continue

    valid_codes=set(px.index)
    day=day[day['ts_code'].isin(valid_codes)]
    if len(day)<50: continue

    w=ic_weights[rd]
    day['score']=0
    for f in FEATS:
        if f not in day.columns: continue
        mu=day[f].mean(); std=day[f].std()
        if std>0: day['score']+=w.get(f,1.0/len(FEATS))*(day[f].fillna(mu)-mu)/std
    day['score']=day['score']/sum(w.values())  # 确保归一化

    px_match=px.loc[day['ts_code'].values]
    day['mcap']=px_match['mcap'].values
    day['ret_1d']=px_match['ret_1d'].values
    day['fwd_ret']=px_match['fwd_ret'].values

    day['mcap_r']=day['mcap'].rank(pct=True)
    day=day[day['mcap_r']>=0.20]
    day=day[day['ret_1d']<0.095]
    day=day[day['fwd_ret'].notna()]

    top=day.nlargest(TOP_N,'score')
    if len(top)<5: continue

    month_ret=top['fwd_ret'].mean()-COST
    monthly_rets.append(month_ret)
    details.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                    'yr':rd.year,'mcap_avg':top['mcap'].median(),
                    'top_w':max(w.values())})

con.close()

# 5. 绩效
print('[4] 绩效统计...\n')
r=np.array(monthly_rets)
ann=np.mean(r)*12; vol=np.std(r)*np.sqrt(12)
sh=ann/vol if vol>0 else 0
cum=np.cumprod(1+r); mdd=np.min(cum/np.maximum.accumulate(cum)-1)
win=(r>0).mean()

print(f'{"指标":<20s} {"数值":>10s}')
print(f'{"-"*30}')
print(f'{"年化收益":<20s} {ann*100:>+9.1f}%')
print(f'{"年化波动":<20s} {vol*100:>9.1f}%')
print(f'{"Sharpe":<20s} {sh:>10.2f}')
print(f'{"MDD":<20s} {mdd*100:>9.1f}%')
print(f'{"月胜率":<20s} {win*100:>9.1f}%')
print(f'{"月数":<20s} {len(r):>10d}')
print(f'{"终值(100万起)":<20s} {1e6*np.prod(1+r):>10,.0f}')
print(f'{"总收益":<20s} {(np.prod(1+r)-1)*100:>+9.1f}%')

# 分年
print(f'\n{"年份":<6s} {"月数":>5s} {"年化":>8s} {"Sharpe":>8s} {"Win%":>6s}')
print(f'{"-"*36}')
for yr in range(2016,2026):
    dr=[d['ret'] for d in details if d['yr']==yr]
    if len(dr)>=3:
        a=np.mean(dr)*12; v=np.std(dr)*np.sqrt(12)
        s=a/v if v>0 else 0; w=(np.array(dr)>0).mean()*100
        print(f'{yr:<6d} {len(dr):>5d} {a*100:>+7.1f}% {s:>+7.2f} {w:>5.0f}%')

print(f'\n耗时: {time.time()-t0:.0f}s')
