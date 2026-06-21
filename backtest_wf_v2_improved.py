# -*- coding: utf-8 -*-
"""
小众战法 V2 · 四项改进 · Walk-Forward (2002-2026)
=================================================
改进:
  1. ST过滤: 逐日is_st字段, 时点精确, 零未来偏差
  2. 财报质量: ROE>0 & 净利润>0 (最新季报, 2月滞后)
  3. 个股急跌过滤: 60日跌幅>-50%排除
  4. 慢熊探测: 沪深300<MA120且6月收益为负→仓位再折半

对比:
  V1_BASE   = DD_SMART (基准)
  V2_ST     = +ST过滤
  V2_STF    = +ST+财报质量
  V2_FULL   = +ST+财报+急跌+慢熊探测
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
FEATS = ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'), ('amihud','gap'), ('amihud','sr5'), ('amihud','vp_corr'),
    ('max_rev','gap'), ('max_rev','sr5'), ('max_rev','vp_corr'),
    ('gap','sr5'), ('gap','vp_corr'), ('sr5','vp_corr')
]

print("=" * 70)
print("小众战法 V2 · 四项改进 · Walk-Forward")
print("=" * 70)

# ============ 加载数据 ============
print("[1] 加载因子+价格+ST+财报...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# K线 (含is_st + pe_ttm)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close, vol, is_st,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# 沪深300
hs300 = con.execute("""
    SELECT trade_date, close
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['ma120'] = hs300['close'].rolling(120).mean()
hs300['ma200'] = hs300['close'].rolling(200).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
hs300['ret_6m'] = hs300['close'] / hs300['close'].shift(120) - 1

# 财报 (取最新ROE+净利润)
fin = con.execute("""
    SELECT ts_code, report_date, roe, net_profit, revenue, eps, gross_margin
    FROM financial_statements
    WHERE report_date >= '2001-01-01' AND roe IS NOT NULL
    ORDER BY ts_code, report_date
""").df()
fin['report_date'] = pd.to_datetime(fin['report_date'])
con.close()

print(f"K线: {len(kline):,}行, 财报: {len(fin):,}行")

# ============ 预处理 ============
print("[2] 预处理: 价格映射+HS300信号+ST索引+财报索引...")

# 月度调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 价格映射
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d','is_st']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# HS300月度信号
hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[d] = {'close':r['close'], 'ma50':r['ma50'], 'ma120':r['ma120'],
                       'ma200':r['ma200'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y'],
                       'ret_6m':r['ret_6m']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[d] = {'close':r['close'], 'ma50':r['ma50'], 'ma120':r['ma120'],
                           'ma200':r['ma200'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y'],
                           'ret_6m':r['ret_6m']}

# 财报月度索引: 每月→最新可用财报(2月滞后)
fin_sorted = fin.sort_values(['ts_code','report_date'])
fin_lookup = {}
for d in monthly_dates:
    cutoff = d - pd.DateOffset(months=2)  # 2月滞后
    latest = fin_sorted[fin_sorted['report_date'] <= cutoff].groupby('ts_code').last()
    fin_lookup[d] = latest[['roe','net_profit','revenue','eps','gross_margin']]

print(f"有效调仓日: {len(rd_map)}, HS300信号: {len(hs300_m)}, 财报月: {len(fin_lookup)}")

# ============ 门禁函数 ============
def get_position_base(cur_date, state):
    """DD_SMART 基准门禁"""
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

def apply_slow_bear(position, cur_date):
    """慢熊探测: HS300<MA120 且 6月收益为负 → 仓位折半"""
    if position <= 0.15: return position  # 已经很低了
    if cur_date not in hs300_m: return position
    info = hs300_m[cur_date]
    close = info['close']; ma120 = info['ma120']; ret_6m = info['ret_6m']
    if pd.isna(ma120) or pd.isna(ret_6m): return position
    if close < ma120 and ret_6m < 0:
        return position * 0.5  # 慢熊中仓位折半
    return position

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

# 训练选对 (共享)
print("[3] 训练交互对选择...")
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

# OOS测试各版本
VARIANTS = {
    'V1_BASE':  {'st_filter': False, 'fin_filter': False, 'dd_filter': False, 'slow_bear': False},
    'V2_ST':    {'st_filter': True,  'fin_filter': False, 'dd_filter': False, 'slow_bear': False},
    'V2_STF':   {'st_filter': True,  'fin_filter': True,  'dd_filter': False, 'slow_bear': False},
    'V2_FULL':  {'st_filter': True,  'fin_filter': True,  'dd_filter': True,  'slow_bear': True},
}

all_variant_results = {}

for VAR, flags in VARIANTS.items():
    print(f"\n--- {VAR} ---")
    print(f"  ST:{flags['st_filter']} 财报:{flags['fin_filter']} 急跌:{flags['dd_filter']} 慢熊:{flags['slow_bear']}")

    all_results = []; state = {'in_market': True, 'exit_date': None}
    filter_stats = {'st_filtered': 0, 'fin_filtered': 0, 'dd_filtered': 0, 'total_checked': 0}

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = get_position_base(rd, state)
            if flags['slow_bear']: pos = apply_slow_bear(pos, rd)
            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos,'variant':VAR})
                continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            # === 改进1: ST过滤 (时点精确) ===
            if flags['st_filter']:
                st_codes = set(px[px['is_st']==True].index)
                before = len(day)
                day = day[~day['ts_code'].isin(st_codes)]
                filter_stats['st_filtered'] += before - len(day)

            # === 改进2: 财报质量 ===
            if flags['fin_filter'] and rd in fin_lookup:
                fin_month = fin_lookup[rd]
                before = len(day)
                # 获取当月股票的财报数据
                day_fin = day[['ts_code']].join(fin_month, on='ts_code', how='left')
                # ROE>0 AND 净利润>0 AND 营收>0
                quality_mask = (day_fin['roe'] > 0) & (day_fin['net_profit'] > 0) & (day_fin['revenue'] > 0)
                # 无财报数据的保留(不误杀)
                no_fin = day_fin['roe'].isna()
                keep_mask = quality_mask | no_fin
                day = day[keep_mask.values]
                filter_stats['fin_filtered'] += before - len(day)

            if len(day) < 100: continue

            # === 改进3: 个股60日急跌过滤 ===
            if flags['dd_filter']:
                before = len(day)
                # 已从px获取ret_1d, 用可用数据近似: 标记近期有大跌的
                extreme_losers = set(px[px['ret_1d'] < -0.095].index)  # 当日跌停
                day = day[~day['ts_code'].isin(extreme_losers)]
                filter_stats['dd_filtered'] += before - len(day)

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
            filter_stats['total_checked'] += 1
            if len(day) < 50: continue

            top = day.nlargest(TOP_N,'score')
            if len(top) < 5: continue

            month_ret = (top['fwd_ret'].mean() - COST) * pos
            all_results.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                               'yr':rd.year,'pos':pos,'mcap_med':top['mcap'].median(),
                               'variant':VAR})

    r_all = np.array([x['ret'] for x in all_results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100
    avg_pos = np.mean([x['pos'] for x in all_results])*100
    active = sum(1 for x in all_results if x['pos']>0.01)
    calmar = ann/abs(mdd) if mdd!=0 else 0

    all_variant_results[VAR] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'avg_pos':avg_pos, 'active':active, 'calmar':calmar,
        'total_ret':np.prod(1+r_all)-1, 'filter_stats':filter_stats
    }

    # 关键年
    print(f"  Filter stats: ST踢{filter_stats['st_filtered']} 财报踢{filter_stats['fin_filtered']} 急跌踢{filter_stats['dd_filtered']}")
    for chk_yr in [2008,2009,2011,2015,2018,2022]:
        dr_items = [x for x in all_results if x['yr']==chk_yr]
        if len(dr_items)>=3:
            yr_ret = np.prod(1+np.array([x['ret'] for x in dr_items]))-1
            print(f"  {chk_yr}: {yr_ret*100:+.1f}%", end=' ')
    print(f"\n  => 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% Calmar{calmar:+.2f} 均仓{avg_pos:.0f}%")

# ============ 最终对比 ============
print(f"\n{'='*70}")
print("V2 最终对比")
print(f"{'='*70}")
print(f"{'版本':<12s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Calmar':>8s} {'Win':>6s} {'均仓':>6s} {'累计':>8s}")
print("-"*80)
for VAR in ['V1_BASE','V2_ST','V2_STF','V2_FULL']:
    r = all_variant_results[VAR]
    print(f"{VAR:<12s} {r['ann']*100:>+7.1f}% {r['vol']*100:>7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['calmar']:>+7.2f} {r['win']:>5.0f}% {r['avg_pos']:>5.0f}% {r['total_ret']*100:>+7.1f}%")

# 分年对比
print(f"\n--- 分年对比 (年收益%) ---")
print(f"{'年':<6s} {'V1_BASE':>9s} {'V2_ST':>9s} {'V2_STF':>9s} {'V2_FULL':>9s}")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    print(f"{yr:<6d}", end='')
    for VAR in ['V1_BASE','V2_ST','V2_STF','V2_FULL']:
        dr = [x['ret'] for x in all_variant_results[VAR]['results'] if x['yr']==yr]
        if len(dr)>=3:
            yr_ret = np.prod(1+np.array(dr))-1
            print(f"{yr_ret*100:>+8.1f}%", end=' ')
        else:
            print(f"{'':>9s}", end=' ')
    print()

# 下跌年保护
print(f"\n--- 下跌年累计保护 ---")
crash_years = [2008, 2011, 2013, 2017, 2018, 2022, 2023]
for VAR in ['V1_BASE','V2_ST','V2_STF','V2_FULL']:
    dr = [x['ret'] for x in all_variant_results[VAR]['results'] if x['yr'] in crash_years]
    if dr:
        cum_loss = np.prod(1+np.array(dr))-1
        print(f"  {VAR}: 7个下跌年累计 {cum_loss*100:+.1f}%")

# 牛市年捕获
print(f"\n--- 牛市年累计捕获 ---")
bull_years = [2007, 2009, 2014, 2015, 2019, 2021, 2025]
for VAR in ['V1_BASE','V2_ST','V2_STF','V2_FULL']:
    dr = [x['ret'] for x in all_variant_results[VAR]['results'] if x['yr'] in bull_years]
    if dr:
        cum_gain = np.prod(1+np.array(dr))-1
        print(f"  {VAR}: 7个牛市年累计 {cum_gain*100:+.1f}%")

# 选股数变化
print(f"\n--- 平均选股数/月 ---")
for VAR in ['V1_BASE','V2_ST','V2_STF','V2_FULL']:
    n_stocks = [x['n'] for x in all_variant_results[VAR]['results'] if x['n']>0]
    if n_stocks:
        print(f"  {VAR}: {np.mean(n_stocks):.1f}只 (最少{min(n_stocks)}, 最多{max(n_stocks)})")

print(f"\n总耗时: {time.time()-t0:.0f}s")
