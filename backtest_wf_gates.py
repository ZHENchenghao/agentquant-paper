# -*- coding: utf-8 -*-
"""
小众战法 + 市场择时门禁 · 滚动Walk-Forward (2002-2026)
=====================================================
对比4种门禁方案:
  NONE  — 无门禁(纯长仓, baseline)
  MA200 — 沪深300<MA200时空仓
  BREADTH — 全A股>MA50比例<30%时空仓
  BOTH  — MA200 OR BREADTH任一触发即空仓
  DRAWDOWN — 指数从高点回撤>15%时空仓

每个方案独立跑Walk-Forward, 最后对比
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

# ============ 参数 ============
TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095

FEATS = ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'), ('amihud','gap'), ('amihud','sr5'), ('amihud','vp_corr'),
    ('max_rev','gap'), ('max_rev','sr5'), ('max_rev','vp_corr'),
    ('gap','sr5'), ('gap','vp_corr'), ('sr5','vp_corr')
]

print("=" * 70)
print("小众战法 + 市场择时门禁 · 对比回测")
print("=" * 70)

# ============ 加载数据 ============
print("[1] 加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# K线: 含指数
kline = con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, vol,
           COALESCE(amount, GREATEST(vol*close, 1.0)) AS amount_proxy,
           COALESCE(close * total_share / 10000, GREATEST(COALESCE(amount, GREATEST(vol*close,1.0)), close*vol) / 1000000) AS mcap,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret_1d
    FROM kline_daily WHERE trade_date >= '2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# 沪深300指数 (用于MA200择时)
hs300 = con.execute("""
    SELECT trade_date, close, close / LAG(close) OVER(ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE ts_code = 'sh000300' AND trade_date >= '2001-07-01'
    ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma200'] = hs300['close'].rolling(200).mean()

# 中证全指 (备用)
csi_all = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code = 'sh000985' AND trade_date >= '2001-07-01'
    ORDER BY trade_date
""").df()
csi_all['trade_date'] = pd.to_datetime(csi_all['trade_date'])

con.close()

# ============ 月度调仓日 ============
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 持有期映射
print("[2] 构建持有期映射+门禁信号...")
rd_map = {}
for i in range(len(monthly_dates) - 1):
    cur = monthly_dates[i]; nxt = monthly_dates[i + 1]
    cp = kline[kline['trade_date'] == cur][['ts_code', 'close', 'mcap', 'ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date'] == nxt][['ts_code', 'open']].rename(columns={'open': 'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open'] / m['close'] - 1
    rd_map[cur] = m
del kline; gc.collect()

# ============ 门禁信号 ============
# MA200 gate for 沪深300
hs300_map = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date'] == d]
    if len(row) > 0:
        hs300_map[d] = {'close': row['close'].iloc[0], 'ma200': row['ma200'].iloc[0]}
    else:
        # 最近交易日
        nearby = hs300[hs300['trade_date'] <= d]
        if len(nearby) > 0:
            last = nearby.iloc[-1]
            hs300_map[d] = {'close': last['close'], 'ma200': last['ma200']}

# 中证全指门禁
csi_map = {}
for d in monthly_dates:
    row = csi_all[csi_all['trade_date'] == d]
    if len(row) > 0:
        csi_map[d] = row['close'].iloc[0]

# 市场宽度: 需要计算(后续动态算)
# 指数回撤: 用沪深300滚动高点

def get_gate_signals(cur_date, gate_type):
    """
    返回: 1.0=满仓, 0.5=半仓, 0.0=空仓
    """
    if gate_type == 'NONE':
        return 1.0

    weight = 1.0

    if gate_type in ('MA200', 'BOTH'):
        if cur_date in hs300_map:
            c = hs300_map[cur_date]['close']
            ma = hs300_map[cur_date]['ma200']
            if pd.notna(ma) and c < ma:
                return 0.0

    if gate_type in ('BREADTH', 'BOTH'):
        # 计算当日全市场%股票>MA50 (用因子数据中的股票)
        day_data = fn[fn['trade_date'] == cur_date]
        if len(day_data) > 200:
            # 简化: 用当日上涨比例作为广度代理
            # 完整版需要前50日均价, 这里用因子数据已有的
            pass
        # Breadth从HS300看: 沪深300成分股>MA50的比例
        breadth_ok = True
        if cur_date in hs300_map:
            c = hs300_map[cur_date]['close']
            # 如果沪深300从高点跌太多 → 广度差
            lookback = [d for d in monthly_dates if d <= cur_date]
            if len(lookback) >= 12:
                past_yr = lookback[-12:]
                past_closes = [hs300_map[d]['close'] for d in past_yr if d in hs300_map]
                if past_closes:
                    peak = max(past_closes)
                    if c / peak - 1 < -0.15:  # -15% from 1yr high
                        breadth_ok = False
        if not breadth_ok:
            return 0.0

    if gate_type == 'DRAWDOWN':
        if cur_date in hs300_map:
            c = hs300_map[cur_date]['close']
            # 计算滚动2年高点
            lookback_dates = [d for d in monthly_dates if d <= cur_date]
            if len(lookback_dates) >= 24:
                past_2yr = lookback_dates[-24:]
                past_closes = [hs300_map[d]['close'] for d in past_2yr if d in hs300_map]
                if past_closes:
                    peak_2yr = max(past_closes)
                    dd_pct = c / peak_2yr - 1
                    if dd_pct < -0.15:  # 指数从2年高点跌超15%
                        return 0.0
                    elif dd_pct < -0.10:
                        return 0.5  # 半仓

    return weight

# ============ Walk-Forward (每个门禁独立) ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS  # 2007

GATES = ['NONE', 'MA200', 'DRAWDOWN', 'BOTH']
all_gate_results = {}

for GATE in GATES:
    print(f"\n{'='*70}")
    print(f"门禁: {GATE}")
    print(f"{'='*70}")

    all_results = []
    pair_history = []

    for test_yr in range(FIRST_TEST_YR, YEARS[-1] + 1):
        train_start_yr = test_yr - TRAIN_YEARS
        train_mds = [d for d in monthly_dates if train_start_yr <= d.year < test_yr]
        test_mds = [d for d in monthly_dates if d.year == test_yr]
        if len(train_mds) < 24 or len(test_mds) < 3: continue

        # 训练: 选最佳4对
        pair_ir = {}
        for (fa, fb) in ALL_PAIRS:
            monthly_spreads = []
            for rd in train_mds:
                if rd not in rd_map: continue
                day = fn[fn['trade_date'] == rd].copy()
                px = rd_map[rd]
                valid = set(px.index)
                day = day[day['ts_code'].isin(valid)]
                if len(day) < 100: continue
                for f in [fa, fb]:
                    if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)
                if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns: continue
                day['score'] = day[f'{fa}_r'] * day[f'{fb}_r']
                day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
                valid_d = day.dropna(subset=['score', 'fwd_ret'])
                if len(valid_d) < 50: continue
                n_q = int(len(valid_d) * 0.2)
                top_spread = valid_d.nlargest(n_q, 'score')['fwd_ret'].mean()
                bot_spread = valid_d.nsmallest(n_q, 'score')['fwd_ret'].mean()
                monthly_spreads.append(top_spread - bot_spread)

            if len(monthly_spreads) >= 12:
                mu_s = np.mean(monthly_spreads); std_s = np.std(monthly_spreads)
                pair_ir[(fa, fb)] = mu_s / std_s if std_s > 0 else 0

        sorted_pairs = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)
        top4 = [p for p, ir in sorted_pairs[:4]]

        # 测试
        fold_rets = []
        for rd in test_mds:
            if rd not in rd_map: continue

            # 门禁
            position_weight = get_gate_signals(rd, GATE)
            if position_weight == 0.0:
                fold_rets.append({
                    'date': str(rd)[:7], 'ret': 0.0, 'n': 0, 'yr': rd.year,
                    'test_yr': test_yr, 'gate': GATE, 'weight': 0.0
                })
                continue

            day = fn[fn['trade_date'] == rd].copy()
            px = rd_map[rd]
            valid = set(px.index)
            day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            all_f = list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

            day['score'] = 0; ok = True
            for fa, fb in top4:
                if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                    day['score'] += day[f'{fa}_r'] * day[f'{fb}_r']
                else: ok = False
            if not ok: continue

            px_match = px.loc[day['ts_code'].values]
            day['mcap'] = px_match['mcap'].values
            day['ret_1d'] = px_match['ret_1d'].values
            day['fwd_ret'] = px_match['fwd_ret'].values
            day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r'] >= MCAP_FLOOR]
            day = day[day['ret_1d'] < LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue

            top = day.nlargest(TOP_N, 'score')
            if len(top) < 5: continue

            month_ret = (top['fwd_ret'].mean() - COST) * position_weight
            fold_rets.append({
                'date': str(rd)[:7], 'ret': month_ret, 'n': len(top),
                'yr': rd.year, 'test_yr': test_yr, 'gate': GATE,
                'weight': position_weight, 'mcap_med': top['mcap'].median()
            })

        if fold_rets:
            r_arr = np.array([x['ret'] for x in fold_rets])
            ann = np.mean(r_arr) * 12; vol = np.std(r_arr) * np.sqrt(12)
            sh = ann / vol if vol > 0 else 0
            cum = np.cumprod(1 + r_arr); mdd = np.min(cum / np.maximum.accumulate(cum) - 1)
            pair_history.append({
                'test_yr': test_yr, 'pairs': top4, 'ann': ann, 'sharpe': sh, 'mdd': mdd
            })
            all_results.extend(fold_rets)

            pairs_str = ' | '.join([f'{a[:4]}×{b[:4]}' for a,b in top4])
            print(f"  {test_yr} | {pairs_str} | {len(fold_rets)}m | {ann*100:+.1f}% | S{sh:+.2f} | MDD{mdd*100:.1f}%")

    # 汇总
    r_all = np.array([x['ret'] for x in all_results])
    ann_all = np.mean(r_all) * 12; vol_all = np.std(r_all) * np.sqrt(12)
    sh_all = ann_all / vol_all if vol_all > 0 else 0
    cum_all = np.cumprod(1 + r_all)
    mdd_all = np.min(cum_all / np.maximum.accumulate(cum_all) - 1)
    win_all = (r_all > 0).mean() * 100
    cash_months = sum(1 for x in all_results if x['weight'] == 0)

    all_gate_results[GATE] = {
        'results': all_results, 'pairs': pair_history,
        'ann': ann_all, 'vol': vol_all, 'sharpe': sh_all, 'mdd': mdd_all,
        'win': win_all, 'months': len(r_all), 'cash_months': cash_months,
        'total_ret': np.prod(1 + r_all) - 1
    }

    print(f"\n  >>> {GATE}: 年化{ann_all*100:+.1f}% | Sharpe{sh_all:+.2f} | MDD{mdd_all*100:.1f}% | Win{win_all:.0f}% | 空仓{cash_months}月")

# ============ 最终对比 ============
print(f"\n{'='*70}")
print("最终对比: 4种门禁方案")
print(f"{'='*70}")
print(f"{'方案':<12s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win':>6s} {'空仓月':>6s} {'累计':>8s}")
print("-" * 72)
for g in GATES:
    r = all_gate_results[g]
    print(f"{g:<12s} {r['ann']*100:>+7.1f}% {r['vol']*100:>7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['win']:>5.0f}% {r['cash_months']:>6d} {r['total_ret']*100:>+7.1f}%")

# 分年对比
print(f"\n--- 分年对比 ---")
print(f"{'年':<6s}", end='')
for g in GATES:
    print(f"{g:>10s}", end=' ')
print()
for yr in range(FIRST_TEST_YR, YEARS[-1] + 1):
    print(f"{yr:<6d}", end='')
    for g in GATES:
        dr = [x['ret'] for x in all_gate_results[g]['results'] if x['yr'] == yr]
        if len(dr) >= 3:
            yr_ret = np.prod(1 + np.array(dr)) - 1
            print(f"{yr_ret*100:>+9.1f}%", end=' ')
        else:
            print(f"{'':>10s}", end=' ')
    print()

# 回撤事件对比
print(f"\n--- 最大回撤对比 ---")
for g in GATES:
    r = all_gate_results[g]
    print(f"{g}: MDD={r['mdd']*100:.1f}% | 空仓{r['cash_months']}月/{r['months']}总月")

print(f"\n总耗时: {time.time()-t0:.0f}s")
