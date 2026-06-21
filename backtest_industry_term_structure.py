# -*- coding: utf-8 -*-
"""行业轮动 · 期限结构框架 · 中银证券方法论复现
==============================================
核心: 1年动量 + 2-3年反转 + 低拥挤度(换手率)
HHI行业集中度 → 出清周期 → 反转时机
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("行业轮动 · 期限结构框架 (中银方法论)")
print("=" * 70)

# ===== 1. 行业指数 + HHI =====
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

raw = con.execute("""
    SELECT industry, trade_date, close
    FROM proxy_industry_daily WHERE trade_date >= DATE '2005-01-01'
    ORDER BY industry, trade_date
""").df()
raw['trade_date'] = pd.to_datetime(raw['trade_date'])

# 行业HHI: 从个股集中度计算 (用 total_mv 在 kline_daily 中)
# HHI = sum((stock_mv/industry_mv)^2)
hhi_raw = con.execute("""
    SELECT si.ind_name as industry, k.trade_date,
           SUM(k.total_mv) as ind_mv,
           SUM(k.total_mv * k.total_mv) / (SUM(k.total_mv) * SUM(k.total_mv)) as hhi
    FROM kline_daily k
    JOIN stock_industry si ON k.ts_code = si.ts_code
    WHERE k.total_mv > 0 AND k.trade_date >= DATE '2010-01-01'
    GROUP BY si.ind_name, k.trade_date
""").df()
hhi_raw['trade_date'] = pd.to_datetime(hhi_raw['trade_date'])

con.close()

# ===== 2. 行业月度收益 + 期限结构因子 =====
raw = raw.sort_values(['industry', 'trade_date']).reset_index(drop=True)
raw['month'] = raw['trade_date'].dt.to_period('M')

monthly = raw.groupby(['industry', 'month'])['close'].last().reset_index()
monthly['month'] = monthly['month'].dt.to_timestamp()

# 各期限收益
for window, label in [(1, '1m'), (3, '3m'), (6, '6m'), (12, '12m'),
                        (18, '18m'), (24, '24m'), (30, '30m'), (36, '36m')]:
    monthly[f'ret_{label}'] = monthly.groupby('industry')['close'].pct_change(window)

# 期限结构因子
# 因子1: 12月动量 (剔除最近1月, 即 t-12 到 t-1)
monthly['mom_12m_ex1m'] = (monthly.groupby('industry')['close'].shift(1) /
                           monthly.groupby('industry')['close'].shift(12) - 1)

# 因子2: 24-36月反转 (t-36 到 t-24 的收益, 负值=前期跌→反转预期)
monthly['rev_24_36m'] = (monthly.groupby('industry')['close'].shift(24) /
                         monthly.groupby('industry')['close'].shift(36) - 1)

# 因子3: 30月反转 (t-30 到 t-1, 中银的另一个窗口)
monthly['rev_30m'] = monthly['ret_30m']

# 期限结构差: 短期-长期 (正值=短期动量强于长期趋势→拥挤预警)
monthly['term_structure'] = monthly['ret_6m'] - monthly['ret_36m']

# ===== 3. HHI 出清检测 =====
hhi_raw['month'] = hhi_raw['trade_date'].dt.to_period('M')
hhi_monthly = hhi_raw.groupby(['industry', 'month']).agg(
    hhi=('hhi', 'last'),
    ind_mv=('ind_mv', 'last')
).reset_index()
hhi_monthly['month'] = hhi_monthly['month'].dt.to_timestamp()

# HHI变化: 3年窗口HHI变化率
hhi_monthly['hhi_chg_36m'] = hhi_monthly.groupby('industry')['hhi'].pct_change(36)

# 合并
monthly = monthly.merge(hhi_monthly[['industry', 'month', 'hhi', 'hhi_chg_36m']],
                        on=['industry', 'month'], how='left')

# ===== 4. 低拥挤度 =====
# 用行业指数日线计算换手率代理(振幅/收盘)
raw['daily_range'] = (raw.groupby('industry')['close'].transform(
    lambda x: x.diff().abs() / x.shift(1)))

raw_m = raw.groupby(['industry', 'month']).agg(
    avg_range=('daily_range', 'mean'),
    vol_range=('daily_range', 'std'),
).reset_index()
raw_m['month'] = raw_m['month'].dt.to_timestamp()

# 拥挤度: 近期振幅/长期振幅 (高=交易拥挤)
raw_m['crowd_ratio'] = raw_m.groupby('industry')['avg_range'].transform(
    lambda x: x.rolling(3).mean() / x.rolling(12).mean())
raw_m['crowd_z'] = raw_m.groupby('month')['crowd_ratio'].rank(pct=True)

monthly = monthly.merge(raw_m[['industry', 'month', 'crowd_z']],
                        on=['industry', 'month'], how='left')

# ===== 5. 全因子清单 =====
FULL_FACTORS = {
    # 原动量因子
    'ret_1m': ('动量1月', 1),
    'ret_3m': ('动量3月', 1),
    'ret_6m': ('动量6月', 1),
    'ret_12m': ('动量12月', 1),

    # 🆕 期限结构(中银)
    'mom_12m_ex1m': ('12月动量(剔1月)', 1),
    'rev_24_36m': ('24-36月反转', -1),  # 前期跌→反转预期→看多
    'rev_30m': ('30月反转', -1),
    'term_structure': ('期限结构差', -1),  # 短>长→拥挤→看空

    # 🆕 HHI
    'hhi': ('HHI集中度', 1),  # 集中度高→龙头强
    'hhi_chg_36m': ('HHI变化3年', 1),  # 集中度提升→出清中

    # 拥挤度
    'crowd_z': ('拥挤度', -1),  # 高拥挤→回避
}

# 目标
monthly['fwd_ret'] = monthly.groupby('industry')['ret_1m'].shift(-1)
monthly = monthly.dropna(subset=['fwd_ret'])
print(f"[1] 数据: {len(monthly)}行, {monthly['month'].nunique()}月, "
      f"{monthly['industry'].nunique()}行业 ({monthly['month'].min().date()}~{monthly['month'].max().date()})")

# ===== 6. Walk-Forward IC (2011-2026) =====
YEARS = sorted(set(d.year for d in monthly['month']))
TRAIN_YEARS = 5
WF_START = YEARS[0] + TRAIN_YEARS + 1

print(f"\n[2] WF IC ({WF_START}-{YEARS[-1]})")

ic_results = {}
for factor, (name, expected_dir) in FULL_FACTORS.items():
    if factor not in monthly.columns:
        continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01')
        te = pd.Timestamp(f'{test_yr}-12-31')
        test = monthly[(monthly['month'] >= ts) & (monthly['month'] <= te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[factor, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[factor].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic):
                    ics.append(ic)
    if len(ics) > 10:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std > 0 else 0
        ir = mi/std*np.sqrt(12) if std > 0 else 0
        dir_ok = (mi > 0 and expected_dir > 0) or (mi < 0 and expected_dir < 0)
        ic_results[factor] = {'name': name, 'ic': mi, 'ir': ir, 't': t,
                               'n': len(ics), 'dir_ok': dir_ok,
                               'expected': expected_dir}

print(f"\n{'因子':<20s} {'预期':>4s} {'IC':>8s} {'IR':>7s} {'t':>7s} {'方向'}")
print("-" * 55)
for f, r in sorted(ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    dir_mark = 'OK' if r['dir_ok'] else 'XX'
    print(f"{r['name']:<20s} {'正' if r['expected']>0 else '负':>4s} "
          f"{r['ic']*100:+7.2f}% {r['ir']:+6.2f} {r['t']:+6.2f} {dir_mark}")

# ===== 7. 策略对比(真实WF) =====
print(f"\n[3] 策略WF对比")

STRATEGIES = {
    '原版(动量1m-12m)': ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m'],
    '期限结构(动量+反转)': ['mom_12m_ex1m', 'rev_24_36m'],
    '期限结构+HHI': ['mom_12m_ex1m', 'rev_24_36m', 'hhi', 'hhi_chg_36m'],
    '期限结构+HHI+拥挤度': ['mom_12m_ex1m', 'rev_24_36m', 'hhi_chg_36m', 'crowd_z'],
}

def run_wf(monthly, factors, train_years, wf_start, last_yr):
    long_rets = []; short_rets = []; ls_rets = []; eq_rets = []
    for test_yr in range(wf_start, last_yr+1):
        # 训练窗定方向
        train = monthly[(monthly['month'] >= pd.Timestamp(f'{test_yr-train_years}-01-01')) &
                        (monthly['month'] <= pd.Timestamp(f'{test_yr-1}-12-31'))]
        test = monthly[(monthly['month'] >= pd.Timestamp(f'{test_yr}-01-01')) &
                       (monthly['month'] <= pd.Timestamp(f'{test_yr}-12-31'))]
        if len(test) < 30 or len(train) < 60: continue

        factor_dirs = {}
        for f in factors:
            if f not in train.columns or f not in test.columns: continue
            ics = []
            for m, grp in train.groupby('month'):
                valid = grp.dropna(subset=[f, 'fwd_ret'])
                if len(valid) > 5:
                    ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                    if not np.isnan(ic): ics.append(ic)
            factor_dirs[f] = 1 if (len(ics) > 8 and np.mean(ics) > 0) else -1

        if not factor_dirs: continue

        test_copy = test.copy()
        for f in factors:
            if f in test_copy.columns and f in factor_dirs:
                test_copy[f'{f}_r'] = test_copy.groupby('month')[f].rank(pct=True) * factor_dirs[f]

        rank_cols = [f'{f}_r' for f in factors if f'{f}_r' in test_copy.columns]
        if not rank_cols: continue
        test_copy['score'] = test_copy[rank_cols].mean(axis=1)

        n_months_this_yr = 0
        for m, grp in test_copy.groupby('month'):
            if len(grp) < 10: continue
            n = max(1, len(grp)//4)
            top = grp.nlargest(n, 'score'); bot = grp.nsmallest(n, 'score')
            long_rets.append(top['fwd_ret'].mean() - 0.003)
            short_rets.append(bot['fwd_ret'].mean() - 0.003)
            ls_rets.append(top['fwd_ret'].mean() - bot['fwd_ret'].mean())
            eq_rets.append(grp['fwd_ret'].mean())
            n_months_this_yr += 1
        if n_months_this_yr == 0:
            continue
    return long_rets, short_rets, ls_rets, eq_rets

def stats(arr, label):
    arr = np.array(arr)
    if len(arr) < 3:
        return {'年化':0, '累积':0, 'Sharpe':0, 'MDD':0, '胜率':0, '月数':0}
    n = len(arr); cum = np.prod(1+arr)
    ann = cum ** (12/n) - 1
    vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
    hit = np.mean(arr > 0)
    return {'年化':ann, '累积':cum-1, 'Sharpe':sh, 'MDD':mdd, '胜率':hit, '月数':n}

print(f"\n{'策略':<26s} {'做多年化':>8s} {'做多MDD':>7s} {'多空年化':>8s} {'多空Sharpe':>7s} {'多空命中':>7s}")
print("-" * 80)

best_long = None
for sname, factors in STRATEGIES.items():
    l, s, ls, eq = run_wf(monthly, factors, TRAIN_YEARS, WF_START, YEARS[-1])
    lstats = stats(ls, '多空')
    lstats_long = stats(l, '做多')
    if best_long is None or lstats_long['年化'] > best_long['年化']:
        best_long = {'name': sname, **lstats_long, 'ls': lstats}
    print(f"{sname:<26s} {lstats_long['年化']*100:+7.1f}% {lstats_long['MDD']*100:+6.1f}% "
          f"{lstats['年化']*100:+7.1f}% {lstats['Sharpe']:+6.2f} {lstats['胜率']*100:+5.0f}%")

# ===== 8. 分年拆解最佳策略 =====
print(f"\n[4] 最佳策略 '{best_long['name']}' 分年")
print(f"{'年份':>6s} {'做多年':>8s} {'多空年':>8s} {'多空月均':>8s}")

factors = STRATEGIES[best_long['name']]
for test_yr in range(WF_START, YEARS[-1]+1):
    train = monthly[(monthly['month'] >= pd.Timestamp(f'{test_yr-TRAIN_YEARS}-01-01')) &
                    (monthly['month'] <= pd.Timestamp(f'{test_yr-1}-12-31'))]
    test = monthly[(monthly['month'] >= pd.Timestamp(f'{test_yr}-01-01')) &
                   (monthly['month'] <= pd.Timestamp(f'{test_yr}-12-31'))]
    if len(test) < 30 or len(train) < 60: continue

    factor_dirs = {}
    for f in factors:
        if f not in train.columns or f not in test.columns: continue
        ics = []
        for m, grp in train.groupby('month'):
            valid = grp.dropna(subset=[f, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
        factor_dirs[f] = 1 if (len(ics) > 8 and np.mean(ics) > 0) else -1

    if not factor_dirs: continue

    test_copy = test.copy()
    for f in factors:
        if f in test_copy.columns and f in factor_dirs:
            test_copy[f'{f}_r'] = test_copy.groupby('month')[f].rank(pct=True) * factor_dirs[f]
    rank_cols = [f'{f}_r' for f in factors if f'{f}_r' in test_copy.columns]
    if not rank_cols: continue
    test_copy['score'] = test_copy[rank_cols].mean(axis=1)

    yr_long = []; yr_ls = []
    for m, grp in test_copy.groupby('month'):
        if len(grp) < 10: continue
        n = max(1, len(grp)//4)
        top = grp.nlargest(n, 'score'); bot = grp.nsmallest(n, 'score')
        yr_long.append(top['fwd_ret'].mean() - 0.003)
        yr_ls.append(top['fwd_ret'].mean() - bot['fwd_ret'].mean())

    if yr_long:
        long_yr = np.prod(1+np.array(yr_long)) - 1
        ls_yr = np.prod(1+np.array(yr_ls)) - 1
        ls_avg = np.mean(yr_ls) * 100
        print(f"  {test_yr:>4d}  {long_yr*100:+7.1f}% {ls_yr*100:+7.1f}% {ls_avg:+7.2f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
