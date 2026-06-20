# -*- coding: utf-8 -*-
"""
小众战法 · 最终生产版 · Top30
==============================
规格:
  因子: 6因子 (amihud,max_rev,price_rev,turnover_rev,sr5,vp_corr)
  方法: 15对交互→每折选IR最高4对→纯乘法
  持仓: Top30, 月度等权调仓
  门禁: DD_SMART, 沪深300为唯一基准
  风控: 市值后20%剔除, 涨停不买, 双边0.66%
  验证: 2002-2026 Walk-Forward, 5年训练→1年OOS
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

# ============ 终极参数 ============
TOP_N = 30          # ← 唯一改动: 15→30
COST = 0.0033       # 双边0.33%/月
TRAIN_YEARS = 5
MCAP_FLOOR = 0.20
LIMIT_UP = 0.095

FEATS = ['amihud', 'max_rev', 'price_rev', 'turnover_rev', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')
]

print("=" * 70)
print("小众战法 · 最终生产版 · Top30")
print("=" * 70)
print(f"因子: {len(FEATS)}个 | 交互对: {len(ALL_PAIRS)}→选4 | 持仓: Top{TOP_N}")
print(f"门禁: DD_SMART(HS300) | 风控: 市值后{MCAP_FLOOR*100:.0f}%剔除+涨停+{COST*100:.2f}%/月成本")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
print(f"因子: {len(fn):,}行 {fn['trade_date'].min().date()}~{fn['trade_date'].max().date()}")

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df(); kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df(); hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# ============ 预处理 ============
print("[2] 构建调仓映射+HS300信号...")
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[d] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[d] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart(cur_date, state):
    """DD_SMART门禁: HS300为唯一基准"""
    if cur_date not in hs300_m: return 1.0, state
    info = hs300_m[cur_date]
    close = info['close']; ma50 = info['ma50']
    high_2y = info['high_2y']; low_1y = info['low_1y']
    if pd.isna(high_2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y = close/high_2y - 1
        if dd_2y < -0.20: return 0.2, {'in_market':False,'exit_date':cur_date}
        elif dd_2y < -0.15: return 0.4, {'in_market':False,'exit_date':cur_date}
        else: return 1.0, state
    else:
        recovery = close/low_1y - 1 if pd.notna(low_1y) and low_1y>0 else 0
        above_ma50 = close > ma50
        if recovery > 0.15 and above_ma50: return 0.7, {'in_market':True,'exit_date':None}
        elif recovery > 0.10: return 0.4, state
        elif recovery > 0.05 and above_ma50: return 0.3, state
        else: return 0.15, state

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

print(f"[3] Walk-Forward ({FIRST_TEST_YR}-{YEARS[-1]})...\n")

# 训练选对
fold_pairs = {}
for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_mds = [d for d in monthly_dates if test_yr-TRAIN_YEARS <= d.year < test_yr]
    if len(train_mds) < 24: continue
    pair_ir = {}
    for (fa,fb) in ALL_PAIRS:
        spreads = []
        for rd in train_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue
            for f in [fa,fb]:
                if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)
            if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns: continue
            day['score'] = day[f'{fa}_r']*day[f'{fb}_r']
            day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
            vd = day.dropna(subset=['score','fwd_ret'])
            if len(vd) < 50: continue
            nq = int(len(vd)*0.2)
            spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
        if len(spreads)>=12:
            mu=np.mean(spreads); std=np.std(spreads)
            pair_ir[(fa,fb)]=mu/std if std>0 else 0
    sorted_pairs = sorted(pair_ir.items(), key=lambda x:x[1], reverse=True)
    fold_pairs[test_yr] = [p for p,ir in sorted_pairs[:4]]

# 选对频率
pair_freq = {}
for test_yr, top4 in fold_pairs.items():
    for p in top4:
        k = f'{p[0][:4]}×{p[1][:4]}'
        pair_freq[k] = pair_freq.get(k,0)+1

# OOS
print("选对频率 (20折):")
for k,v in sorted(pair_freq.items(), key=lambda x:x[1], reverse=True)[:5]:
    print(f"  {k}: {v}/20 ({v/20*100:.0f}%)")

all_results = []; stock_picks = []
state = {'in_market': True, 'exit_date': None}

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    if test_yr not in fold_pairs: continue
    top4 = fold_pairs[test_yr]
    test_mds = [d for d in monthly_dates if d.year==test_yr]
    if len(test_mds) < 3: continue

    for rd in test_mds:
        if rd not in rd_map: continue
        pos, state = dd_smart(rd, state)
        if pos < 0.01:
            all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':0.0})
            continue

        day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
        valid = set(px.index); day = day[day['ts_code'].isin(valid)]
        if len(day) < 100: continue

        all_f = list(set([x for p in top4 for x in p]))
        for f in all_f:
            if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

        day['score'] = 0; ok = True
        for fa,fb in top4:
            if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                day['score'] += day[f'{fa}_r']*day[f'{fb}_r']
            else: ok = False
        if not ok: continue

        px_match = px.loc[day['ts_code'].values]
        day['mcap'] = px_match['mcap'].values
        day['ret_1d'] = px_match['ret_1d'].values
        day['fwd_ret'] = px_match['fwd_ret'].values
        day['mcap_r'] = day['mcap'].rank(pct=True)
        day = day[day['mcap_r']>=MCAP_FLOOR]
        day = day[day['ret_1d']<LIMIT_UP]
        day = day[day['fwd_ret'].notna()]
        if len(day) < 80: continue  # Top30需要至少80只可选

        top = day.nlargest(TOP_N, 'score')
        if len(top) < 15: continue  # 至少选出一半

        month_ret = (top['fwd_ret'].mean() - COST) * pos
        all_results.append({
            'date':str(rd)[:7],'ret':month_ret,'n':len(top),
            'yr':rd.year,'pos':pos,'mcap_med':top['mcap'].median()
        })

        # 记录选股
        for _, row in top.iterrows():
            stock_picks.append({
                'date':str(rd)[:7],'yr':rd.year,
                'ts_code':row['ts_code'],'score':row['score'],
                'fwd_ret':row['fwd_ret'],'pos':pos,'mcap':row['mcap']
            })

# ============ 绩效 ============
r_all = np.array([x['ret'] for x in all_results])
ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
sh = ann/vol if vol>0 else 0
cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
win = (r_all>0).mean()*100
calmar = ann/abs(mdd) if mdd!=0 else 0
avg_pos = np.mean([x['pos'] for x in all_results if x['pos']>0.01])*100
active = sum(1 for x in all_results if x['pos']>0.01)

print(f"\n{'='*70}")
print("最终绩效 · 小众战法 Top30 · 2007-2026 OOS")
print(f"{'='*70}")
print(f"  年化收益:  {ann*100:+.1f}%")
print(f"  年化波动:  {vol*100:.1f}%")
print(f"  Sharpe:    {sh:+.2f}")
print(f"  最大回撤:  {mdd*100:.1f}%")
print(f"  Calmar:    {calmar:+.2f}")
print(f"  月胜率:    {win:.0f}%")
print(f"  累计收益:  {np.prod(1+r_all)-1:+.1%}")
print(f"  活跃仓位:  {avg_pos:.0f}% ({active}/{len(all_results)}月)")
print(f"  OOS月数:   {len(all_results)}")
print(f"  沪深300年化: +5.2% (同期)")

# 分年
print(f"\n{'年':<6s} {'收益':>9s} {'Sharpe':>7s} {'MDD':>7s} {'仓位':>5s}")
print("-"*40)
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    dr = [x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr = np.array(dr)
        a = np.mean(rr)*12; v = np.std(rr)*np.sqrt(12)
        s = a/v if v>0 else 0
        cum_y = np.cumprod(1+rr); m = np.min(cum_y/np.maximum.accumulate(cum_y)-1)
        avg_p = np.mean([x['pos'] for x in all_results if x['yr']==yr and x['pos']>0.01])*100
        print(f"{yr:<6d} {(np.prod(1+rr)-1)*100:>+8.1f}% {s:>+6.2f} {m*100:>+6.1f}% {avg_p:>4.0f}%")

# 统计
pos_yrs = sum(1 for yr in range(FIRST_TEST_YR, YEARS[-1]+1)
              if len([x for x in all_results if x['yr']==yr])>=6
              and np.prod(1+np.array([x['ret'] for x in all_results if x['yr']==yr]))>1)
n_yrs = sum(1 for yr in range(FIRST_TEST_YR, YEARS[-1]+1)
            if len([x for x in all_results if x['yr']==yr])>=6)
crash = [2008,2011,2017,2018,2022]
bull = [2007,2009,2015,2019,2021,2025]
crash_ret = np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in crash]))-1
bull_ret = np.prod(1+np.array([x['ret'] for x in all_results if x['yr'] in bull]))-1

print(f"\n  盈利年: {pos_yrs}/{n_yrs}")
print(f"  5熊年累计: {crash_ret*100:+.1f}%")
print(f"  6牛年累计: {bull_ret*100:+.1f}%")

# 选股统计
df_picks = pd.DataFrame(stock_picks)
print(f"\n  总选股: {len(df_picks)}次")
print(f"  月均选股: {df_picks.groupby('date').size().mean():.1f}只")
print(f"  选中中位市值: {df_picks['mcap'].median()/1e4:.0f}亿")

# 导出
df_picks.to_csv('D:/AgentQuant/our/cache/final_top30_picks.csv', index=False)
pd.DataFrame(all_results).to_csv('D:/AgentQuant/our/cache/final_top30_monthly.csv', index=False)
print(f"\n  明细导出: cache/final_top30_picks.csv + final_top30_monthly.csv")
print(f"  耗时: {time.time()-t0:.0f}s")
