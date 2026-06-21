# -*- coding: utf-8 -*-
"""
小众战法 V3 · ST精细化控仓 · Walk-Forward (2002-2026)
=====================================================
四种方案:
  V1_BASE   = 无ST过滤(基准)
  V2_ALLST  = 踢全部ST (is_st=True)
  V3_MAX3   = 每月最多3只ST, 超出替补非ST
  V4_NOSTAR = 只踢*ST(退市风险), 留ST(其他风险) — 用stock_basic当前名称
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
print("小众战法 V3 · ST精细化控仓")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# K线 (含is_st)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close, is_st,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# 沪深300
hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()

# 当前*ST和ST名单 (从stock_basic名称区分)
star_st_codes = set(con.execute(
    "SELECT ts_code FROM stock_basic WHERE is_st = true AND name LIKE '*%'"
).fetchdf()['ts_code'].tolist())

reg_st_codes = set(con.execute(
    "SELECT ts_code FROM stock_basic WHERE is_st = true AND name LIKE 'ST%' AND name NOT LIKE '*%'"
).fetchdf()['ts_code'].tolist())

con.close()

print(f"当前*ST: {len(star_st_codes)}只, ST: {len(reg_st_codes)}只")

# ============ 预处理 ============
print("[2] 预处理...")
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d','is_st']].set_index('ts_code')
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

def get_position_base(cur_date, state):
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

# 训练选对
print("[3] 训练交互对...")
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

# OOS各方案
VARIANTS = {
    'V1_BASE':   'none',       # 不过滤
    'V2_ALLST':  'all_st',     # 踢全部ST (is_st=True)
    'V3_MAX3':   'max3_st',    # 每月最多3只ST
    'V4_NOSTAR': 'no_star',    # 只踢*ST
}

all_variant_results = {}

for VAR, mode in VARIANTS.items():
    print(f"\n--- {VAR} ({mode}) ---")
    all_results = []; state = {'in_market': True, 'exit_date': None}
    st_stats = {'st_included': 0, 'st_excluded': 0, 'star_excluded': 0, 'total_picks': 0}

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = get_position_base(rd, state)
            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos,'variant':VAR})
                continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            # === ST过滤逻辑 ===
            st_codes_today = set(px[px['is_st']==True].index)

            if mode == 'all_st':
                # V2: 踢全部ST
                before = len(day)
                day = day[~day['ts_code'].isin(st_codes_today)]
                st_stats['st_excluded'] += before - len(day)

            elif mode == 'no_star':
                # V4: 只踢*ST (用当前名单作为代理)
                before = len(day)
                day = day[~day['ts_code'].isin(star_st_codes)]
                st_stats['star_excluded'] += before - len(day)

            elif mode == 'max3_st':
                # V3: 不过滤, 选股后限制ST数量
                pass

            if len(day) < 100: continue

            # 因子排名+乘法
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
            day['is_stock_st'] = day['ts_code'].isin(st_codes_today).values
            day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]
            day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue

            if mode == 'max3_st':
                # 先取Top15, 如果ST>3只, 替换多余的
                day_sorted = day.sort_values('score', ascending=False)
                selected = []; st_count = 0
                for idx, row in day_sorted.iterrows():
                    if len(selected) >= TOP_N: break
                    is_st = row['is_stock_st']
                    if is_st and st_count >= 3:
                        continue  # 跳过超过3只的ST
                    selected.append(row)
                    if is_st: st_count += 1

                if len(selected) < 5: continue
                top = pd.DataFrame(selected)
                st_stats['st_included'] += st_count
                st_stats['total_picks'] += len(selected)
            else:
                top = day.nlargest(TOP_N, 'score')
                if len(top) < 5: continue
                n_st = top['is_stock_st'].sum()
                st_stats['st_included'] += n_st
                st_stats['total_picks'] += len(top)

            month_ret = (top['fwd_ret'].mean() - COST) * pos
            all_results.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                               'yr':rd.year,'pos':pos,'variant':VAR})

    r_all = np.array([x['ret'] for x in all_results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100
    avg_pos = np.mean([x['pos'] for x in all_results])*100
    calmar = ann/abs(mdd) if mdd!=0 else 0

    all_variant_results[VAR] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'avg_pos':avg_pos, 'calmar':calmar,
        'total_ret':np.prod(1+r_all)-1, 'st_stats':st_stats
    }

    print(f"  ST纳入: {st_stats['st_included']}/{st_stats['total_picks']}只次 "
          f"({st_stats['st_included']/max(1,st_stats['total_picks'])*100:.1f}%)"
          f" | 踢ST: {st_stats['st_excluded']} | 踢*ST: {st_stats['star_excluded']}")
    for chk_yr in [2008,2009,2011,2015,2018,2022]:
        dr_items = [x for x in all_results if x['yr']==chk_yr]
        if len(dr_items)>=3:
            yr_ret = np.prod(1+np.array([x['ret'] for x in dr_items]))-1
            print(f"  {chk_yr}: {yr_ret*100:+.1f}%", end=' ')
    print(f"\n  => 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% Calmar{calmar:+.2f}")

# ============ 对比 ============
print(f"\n{'='*70}")
print("V3 最终对比")
print(f"{'='*70}")
print(f"{'方案':<12s} {'年化':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Calmar':>8s} {'累计':>8s} {'均仓':>6s} {'ST%':>6s}")
print("-"*72)
for VAR in ['V1_BASE','V2_ALLST','V3_MAX3','V4_NOSTAR']:
    r = all_variant_results[VAR]
    st_pct = r['st_stats']['st_included']/max(1,r['st_stats']['total_picks'])*100
    print(f"{VAR:<12s} {r['ann']*100:>+7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['calmar']:>+7.2f} {r['total_ret']*100:>+7.1f}% {r['avg_pos']:>5.0f}% {st_pct:>5.0f}%")

# 分年
print(f"\n--- 分年对比 ---")
print(f"{'年':<6s} {'V1_BASE':>9s} {'V2_ALLST':>9s} {'V3_MAX3':>9s} {'V4_NOSTAR':>9s}")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    print(f"{yr:<6d}", end='')
    for VAR in ['V1_BASE','V2_ALLST','V3_MAX3','V4_NOSTAR']:
        dr = [x['ret'] for x in all_variant_results[VAR]['results'] if x['yr']==yr]
        if len(dr)>=3:
            yr_ret = np.prod(1+np.array(dr))-1
            print(f"{yr_ret*100:>+8.1f}%", end=' ')
        else:
            print(f"{'':>9s}", end=' ')
    print()

# 下跌年vs牛市年
crash_years = [2008, 2011, 2013, 2017, 2018, 2022, 2023]
bull_years = [2007, 2009, 2014, 2015, 2019, 2021, 2025]
print(f"\n{'方案':<12s} {'7熊年累计':>11s} {'7牛年累计':>11s} {'牛熊比':>8s}")
for VAR in ['V1_BASE','V2_ALLST','V3_MAX3','V4_NOSTAR']:
    crash_ret = np.prod(1+np.array([x['ret'] for x in all_variant_results[VAR]['results'] if x['yr'] in crash_years]))-1
    bull_ret = np.prod(1+np.array([x['ret'] for x in all_variant_results[VAR]['results'] if x['yr'] in bull_years]))-1
    ratio = (1+bull_ret)/abs(crash_ret) if crash_ret != -1 else 0
    print(f"{VAR:<12s} {crash_ret*100:>+10.1f}% {bull_ret*100:>+10.1f}% {ratio:>7.1f}x")

print(f"\n总耗时: {time.time()-t0:.0f}s")
