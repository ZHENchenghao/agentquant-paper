# -*- coding: utf-8 -*-
"""
5因子 vs 原始6因子 · DD_SMART门禁 · Walk-Forward (2002-2026)
===========================================================
5f: amihud, max_rev, gap, sr5, vp_corr (10对→选4)
6f: amihud, max_rev, price_rev, turnover_rev, sr5, vp_corr (15对→选4)
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095

print("=" * 70)
print("5f vs 6f · DD_SMART门禁 · Walk-Forward")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载数据...")
fn5 = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn5['trade_date'] = pd.to_datetime(fn5['trade_date'])
fn6 = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn6['trade_date'] = pd.to_datetime(fn6['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close, is_st,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# ============ 预处理 ============
print("[2] 预处理...")
dates5 = sorted(fn5['trade_date'].unique())
dates6 = sorted(fn6['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates5).groupby([d.strftime('%Y-%m') for d in dates5]):
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

def get_position_dd_smart(cur_date, state):
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

YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

# ============ 两个版本 ============
CONFIGS = {
    '5f': {
        'fn': fn5,
        'feats': ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr'],
        'pairs': [
            ('amihud','max_rev'),('amihud','gap'),('amihud','sr5'),('amihud','vp_corr'),
            ('max_rev','gap'),('max_rev','sr5'),('max_rev','vp_corr'),
            ('gap','sr5'),('gap','vp_corr'),('sr5','vp_corr')
        ]
    },
    '6f': {
        'fn': fn6,
        'feats': ['amihud', 'max_rev', 'price_rev', 'turnover_rev', 'sr5', 'vp_corr'],
        'pairs': [
            ('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
            ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
            ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
            ('turnover_rev','sr5'),('turnover_rev','vp_corr'),
            ('sr5','vp_corr')
        ]
    }
}

all_results_dict = {}

for VER, cfg in CONFIGS.items():
    fn_use = cfg['fn']; ALL_PAIRS = cfg['pairs']
    print(f"\n{'='*70}")
    print(f"  {VER}: {cfg['feats']}")
    print(f"  交互对: {len(ALL_PAIRS)}对→每折选4")
    print(f"{'='*70}")

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
                day = fn_use[fn_use['trade_date']==rd].copy(); px = rd_map[rd]
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

    # 统计选对频率
    pair_freq = {}
    for test_yr, top4 in fold_pairs.items():
        for p in top4:
            k = f'{p[0][:4]}x{p[1][:4]}'
            pair_freq[k] = pair_freq.get(k,0)+1

    # OOS
    all_results = []; state = {'in_market': True, 'exit_date': None}
    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = get_position_dd_smart(rd, state)
            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos,'ver':VER})
                continue

            day = fn_use[fn_use['trade_date']==rd].copy(); px = rd_map[rd]
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
            if len(day) < 50: continue

            top = day.nlargest(TOP_N,'score')
            if len(top) < 5: continue

            month_ret = (top['fwd_ret'].mean() - COST) * pos
            all_results.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                               'yr':rd.year,'pos':pos,'ver':VER})

    r_all = np.array([x['ret'] for x in all_results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100
    avg_pos = np.mean([x['pos'] for x in all_results])*100
    calmar = ann/abs(mdd) if mdd!=0 else 0

    all_results_dict[VER] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'avg_pos':avg_pos, 'calmar':calmar,
        'total_ret':np.prod(1+r_all)-1, 'pair_freq':pair_freq, 'fold_pairs':fold_pairs
    }

    # 选对频率
    print(f"\n  高频选对:")
    for k,v in sorted(pair_freq.items(), key=lambda x:x[1], reverse=True)[:5]:
        pct = v/len(fold_pairs)*100
        print(f"    {k}: {v}/{len(fold_pairs)} ({pct:.0f}%)")

    # 关键年
    for chk_yr in [2008,2009,2011,2015,2018,2022]:
        dr_items = [x for x in all_results if x['yr']==chk_yr]
        if len(dr_items)>=3:
            yr_ret = np.prod(1+np.array([x['ret'] for x in dr_items]))-1
            print(f"    {chk_yr}: {yr_ret*100:+.1f}%", end=' ')
    print(f"\n  => 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% 累计{np.prod(1+r_all)-1:+.1%}")

# ============ 最终对比 ============
print(f"\n{'='*70}")
print("5f vs 6f 最终对比")
print(f"{'='*70}")
print(f"{'版本':<6s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Calmar':>8s} {'累计':>9s} {'均仓':>6s}")
print("-"*68)
for v in ['5f','6f']:
    r = all_results_dict[v]
    print(f"{v:<6s} {r['ann']*100:>+7.1f}% {r['vol']*100:>7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['calmar']:>+7.2f} {r['total_ret']*100:>+8.1f}% {r['avg_pos']:>5.0f}%")

# 分年
print(f"\n--- 分年 ---")
print(f"{'年':<6s} {'5f':>9s} {'6f':>9s} {'差异':>9s}")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    d5 = [x['ret'] for x in all_results_dict['5f']['results'] if x['yr']==yr]
    d6 = [x['ret'] for x in all_results_dict['6f']['results'] if x['yr']==yr]
    if len(d5)>=3 and len(d6)>=3:
        r5 = np.prod(1+np.array(d5))-1
        r6 = np.prod(1+np.array(d6))-1
        diff = r6 - r5
        marker = ' ←' if diff > 0.05 else (' →' if diff < -0.05 else '')
        print(f"{yr:<6d} {r5*100:>+8.1f}% {r6*100:>+8.1f}% {diff*100:>+8.1f}%{marker}")

# 牛市vs熊市
crash_years = [2008, 2011, 2013, 2017, 2018, 2022, 2023]
bull_years = [2007, 2009, 2014, 2015, 2019, 2021, 2025]
print(f"\n{'版本':<6s} {'7熊累计':>10s} {'7牛累计':>10s}")
for v in ['5f','6f']:
    crash = np.prod(1+np.array([x['ret'] for x in all_results_dict[v]['results'] if x['yr'] in crash_years]))-1
    bull = np.prod(1+np.array([x['ret'] for x in all_results_dict[v]['results'] if x['yr'] in bull_years]))-1
    print(f"{v:<6s} {crash*100:>+9.1f}% {bull*100:>+9.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
