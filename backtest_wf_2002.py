# -*- coding: utf-8 -*-
"""
小众战法 滚动Walk-Forward回测 (2002-2026)
==========================================
方法: 纯乘法交互对, 零ML, 零参数
- 训练窗口(5年): 10对交互中选IR最高的4对
- 测试窗口(1年): OOS验证
- 滚动: 每年向前滚1年
- 调仓: 月末选股, 次月首日开盘价买入
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

# ============ 参数 ============
TOP_N = 15          # 每月选股数
COST = 0.0033       # 单边成本 (印花税0.1%+佣金0.03%+滑点0.2%)
TRAIN_YEARS = 5     # 训练窗口(年)
TEST_YEARS = 1      # 测试窗口(年)
MCAP_FLOOR = 0.20   # 市值后20%排除
LIMIT_UP = 0.095    # 涨停过滤

FEATS = ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr']
# 10对全交互
ALL_PAIRS = [
    ('amihud','max_rev'), ('amihud','gap'), ('amihud','sr5'), ('amihud','vp_corr'),
    ('max_rev','gap'), ('max_rev','sr5'), ('max_rev','vp_corr'),
    ('gap','sr5'), ('gap','vp_corr'),
    ('sr5','vp_corr')
]

print("=" * 70)
print("小众战法 · 滚动Walk-Forward回测 (2002-2026)")
print(f"方法: 纯乘法 | 训练{TRAIN_YEARS}年 | 测试{TEST_YEARS}年 | Top{TOP_N}")
print("=" * 70)

# ============ 加载数据 ============
print("\n[1] 加载因子+价格数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
print(f"因子: {len(fn):,}行 {fn['trade_date'].min().date()}~{fn['trade_date'].max().date()}")

# 获取每月第一个交易日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print(f"月度调仓日: {len(monthly_dates)}个 ({monthly_dates[0].date()}~{monthly_dates[-1].date()})")

# 构建调仓日→下个调仓日价格映射
print("[2] 构建持有期价格映射...")
import duckdb
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close,
           COALESCE(amount, GREATEST(vol*close, 1.0)) AS amount_proxy,
           COALESCE(close * total_share / 10000, GREATEST(COALESCE(amount, GREATEST(vol*close,1.0)), close*vol) / 1000000) AS mcap,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret_1d
    FROM kline_daily WHERE trade_date >= '2002-01-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
con.close()

# 构建: 当前调仓日 → (收盘价, 市值, 前日收益, 下月首日开盘价, 下月收益)
rd_map = {}
for i in range(len(monthly_dates) - 1):
    cur = monthly_dates[i]
    nxt = monthly_dates[i + 1]
    cp = kline[kline['trade_date'] == cur][['ts_code', 'close', 'mcap', 'ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date'] == nxt][['ts_code', 'open']].rename(columns={'open': 'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner')
    m['fwd_ret'] = m['next_open'] / m['close'] - 1
    rd_map[cur] = m
del kline; gc.collect()
print(f"有效调仓日: {len(rd_map)}")

# ============ Walk-Forward 主循环 ============
print("\n[3] 滚动Walk-Forward...\n")

YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS  # 2007
LAST_TEST_YR = YEARS[-1]                 # 2026

all_results = []      # 每月收益
fold_details = []     # 每折详情
pair_history = []     # 每折选了什么对

for test_yr in range(FIRST_TEST_YR, LAST_TEST_YR + 1):
    train_start_yr = test_yr - TRAIN_YEARS
    train_end_yr = test_yr - 1

    # 训练期月度日期
    train_mds = [d for d in monthly_dates if train_start_yr <= d.year <= train_end_yr]
    # 测试期月度日期
    test_mds = [d for d in monthly_dates if d.year == test_yr]

    if len(train_mds) < 24 or len(test_mds) < 3:
        continue

    # ---- 训练: 选出最佳4对 ----
    pair_ir = {}
    for (fa, fb) in ALL_PAIRS:
        monthly_irs = []
        for rd in train_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date'] == rd].copy()
            px = rd_map[rd]
            valid = set(px.index)
            day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            # 因子排名
            for f in [fa, fb]:
                if f in day.columns:
                    day[f'{f}_r'] = day[f].rank(pct=True)

            if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns:
                continue

            day['score'] = day[f'{fa}_r'] * day[f'{fb}_r']
            day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values

            # 计算该月IR (top vs bottom quintile spread / std)
            day_valid = day.dropna(subset=['score', 'fwd_ret'])
            if len(day_valid) < 50: continue
            top_q = day_valid.nlargest(int(len(day_valid) * 0.2), 'score')['fwd_ret'].mean()
            bot_q = day_valid.nsmallest(int(len(day_valid) * 0.2), 'score')['fwd_ret'].mean()
            monthly_irs.append(top_q - bot_q)

        if len(monthly_irs) >= 12:
            mean_spread = np.mean(monthly_irs)
            std_spread = np.std(monthly_irs)
            pair_ir[(fa, fb)] = mean_spread / std_spread if std_spread > 0 else 0

    # 选IR最高的4对
    sorted_pairs = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)
    top4 = [p for p, ir in sorted_pairs[:4]]

    # ---- 测试: OOS运行 ----
    fold_rets = []
    for rd in test_mds:
        if rd not in rd_map: continue
        day = fn[fn['trade_date'] == rd].copy()
        px = rd_map[rd]
        valid = set(px.index)
        day = day[day['ts_code'].isin(valid)]
        if len(day) < 100: continue

        # 所有涉及因子排名
        all_f = list(set([x for p in top4 for x in p]))
        for f in all_f:
            if f in day.columns:
                day[f'{f}_r'] = day[f].rank(pct=True)

        # 乘法得分
        day['score'] = 0
        ok = True
        for fa, fb in top4:
            if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                day['score'] += day[f'{fa}_r'] * day[f'{fb}_r']
            else:
                ok = False
        if not ok: continue

        # 风控过滤
        px_match = px.loc[day['ts_code'].values]
        day['mcap'] = px_match['mcap'].values
        day['ret_1d'] = px_match['ret_1d'].values
        day['fwd_ret'] = px_match['fwd_ret'].values

        day['mcap_r'] = day['mcap'].rank(pct=True)
        day = day[day['mcap_r'] >= MCAP_FLOOR]       # 市值后20%排除
        day = day[day['ret_1d'] < LIMIT_UP]          # 涨停不可买入
        day = day[day['fwd_ret'].notna()]
        if len(day) < 50: continue

        top = day.nlargest(TOP_N, 'score')
        if len(top) < 5: continue

        month_ret = top['fwd_ret'].mean() - COST
        fold_rets.append({
            'date': str(rd)[:7], 'ret': month_ret, 'n': len(top),
            'yr': rd.year, 'test_yr': test_yr, 'train': f'{train_start_yr}-{train_end_yr}',
            'mcap_med': top['mcap'].median()
        })

    if fold_rets:
        r_arr = np.array([x['ret'] for x in fold_rets])
        ann = np.mean(r_arr) * 12
        vol = np.std(r_arr) * np.sqrt(12)
        sh = ann / vol if vol > 0 else 0
        cum = np.cumprod(1 + r_arr)
        mdd = np.min(cum / np.maximum.accumulate(cum) - 1)
        win = (r_arr > 0).mean() * 100

        pair_history.append({
            'test_yr': test_yr, 'train': f'{train_start_yr}-{train_end_yr}',
            'pairs': top4, 'ir': [pair_ir[p] for p in top4],
            'n_months': len(fold_rets), 'ann': ann, 'sharpe': sh, 'mdd': mdd, 'win': win
        })
        all_results.extend(fold_rets)

        # 打印该折
        pairs_str = ' | '.join([f'{a[:4]}×{b[:4]}' for a,b in top4])
        print(f"  {test_yr} | 训练{train_start_yr}-{train_end_yr} | {pairs_str}")
        print(f"        {len(fold_rets)}月 | 年化{ann*100:+.1f}% | Sharpe{sh:+.2f} | MDD{mdd*100:.1f}% | Win{win:.0f}%")

# ============ 汇总 ============
print("\n" + "=" * 70)
print("全期汇总 (纯OOS)")
print("=" * 70)

r_all = np.array([x['ret'] for x in all_results])
ann_all = np.mean(r_all) * 12
vol_all = np.std(r_all) * np.sqrt(12)
sh_all = ann_all / vol_all if vol_all > 0 else 0
cum_all = np.cumprod(1 + r_all)
mdd_all = np.min(cum_all / np.maximum.accumulate(cum_all) - 1)
win_all = (r_all > 0).mean() * 100
final_val = np.prod(1 + r_all)

print(f"OOS期间: {all_results[0]['date']} ~ {all_results[-1]['date']}")
print(f"总月数: {len(r_all)}")
print(f"年化收益: {ann_all*100:+.2f}%")
print(f"年化波动: {vol_all*100:.1f}%")
print(f"Sharpe: {sh_all:+.2f}")
print(f"最大回撤: {mdd_all*100:.1f}%")
print(f"月胜率: {win_all:.1f}%")
print(f"累计收益: {(final_val-1)*100:+.1f}%")
print(f"100万终值: {1e6*final_val:,.0f}")

# 分年
print(f"\n{'年份':<6s} {'训练窗':<13s} {'月':>4s} {'年化':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win':>5s} {'选对'}")
print("-" * 85)
for yr in range(FIRST_TEST_YR, LAST_TEST_YR + 1):
    dr = [x for x in all_results if x['yr'] == yr]
    fold_info = [p for p in pair_history if p['test_yr'] == yr]
    if len(dr) >= 3:
        r = np.array([x['ret'] for x in dr])
        a = np.mean(r) * 12; v = np.std(r) * np.sqrt(12)
        s = a / v if v > 0 else 0
        cum_y = np.cumprod(1 + r)
        m = np.min(cum_y / np.maximum.accumulate(cum_y) - 1) if len(cum_y) > 0 else 0
        w = (r > 0).mean() * 100
        train_str = fold_info[0]['train'] if fold_info else '?'
        pairs_short = ' '.join([f'{a[:3]}×{b[:3]}' for a,b in fold_info[0]['pairs']]) if fold_info else '?'
        print(f"{yr:<6d} {train_str:<13s} {len(dr):>4d} {a*100:>+7.1f}% {s:>+7.2f} {m*100:>7.1f}% {w:>4.0f}% {pairs_short}")

# 胜率统计
print(f"\n--- 年度统计 ---")
yr_rets = []
for yr in range(FIRST_TEST_YR, LAST_TEST_YR + 1):
    dr = [x['ret'] for x in all_results if x['yr'] == yr]
    if len(dr) >= 6:
        yr_ret = np.prod(1 + np.array(dr)) - 1
        yr_rets.append((yr, yr_ret))
pos_yrs = sum(1 for _, r in yr_rets if r > 0)
print(f"盈利年: {pos_yrs}/{len(yr_rets)} ({pos_yrs/len(yr_rets)*100:.0f}%)")
print(f"平均盈利年: {np.mean([r for _,r in yr_rets if r>0])*100:+.1f}%")
print(f"平均亏损年: {np.mean([r for _,r in yr_rets if r<0])*100:+.1f}%")

# 回撤分析
print(f"\n--- 回撤事件 ---")
running_max = 1.0; drawdowns = []
for x in all_results:
    running_max = max(running_max, running_max * (1 + x['ret']))
    dd = running_max * (1 + x['ret']) / running_max - 1
    if dd < -0.10:
        drawdowns.append((x['date'], dd, running_max))
# 合并连续回撤
if drawdowns:
    events = []; cur_start = drawdowns[0][0]; cur_worst = drawdowns[0][1]
    for i in range(1, len(drawdowns)):
        if drawdowns[i][0] != drawdowns[i-1][0]:  # same month check
            prev_m, prev_y = drawdowns[i-1][0].split('-')
            cur_m, cur_y = drawdowns[i][0].split('-')
            if int(cur_y) == int(prev_y) and int(cur_m) == int(prev_m) + 1:
                cur_worst = min(cur_worst, drawdowns[i][1])
            else:
                events.append((cur_start, cur_worst))
                cur_start = drawdowns[i][0]; cur_worst = drawdowns[i][1]
    events.append((cur_start, cur_worst))
    for start, worst in sorted(events, key=lambda x: x[1])[:5]:
        print(f"  {start}: {worst*100:.1f}%")

# 滚动5年Sharpe
print(f"\n--- 滚动5年Sharpe ---")
for yr_start in range(FIRST_TEST_YR, LAST_TEST_YR - 3):
    yr_end = yr_start + 4
    dr = [x['ret'] for x in all_results if yr_start <= x['yr'] <= yr_end]
    if len(dr) >= 30:
        r = np.array(dr)
        a5 = np.mean(r) * 12; v5 = np.std(r) * np.sqrt(12)
        s5 = a5 / v5 if v5 > 0 else 0
        print(f"  {yr_start}-{yr_end}: Sharpe {s5:+.2f} 年化{a5*100:+.1f}%")

# 选对稳定性
print(f"\n--- 交互对选择频率 (共{len(pair_history)}折) ---")
pair_freq = {}
for ph in pair_history:
    for p in ph['pairs']:
        k = f'{p[0][:4]}×{p[1][:4]}'
        pair_freq[k] = pair_freq.get(k, 0) + 1
for k, v in sorted(pair_freq.items(), key=lambda x: x[1], reverse=True):
    pct = v / len(pair_history) * 100
    bar = '█' * int(pct / 5)
    print(f"  {k}: {v}/{len(pair_history)} ({pct:.0f}%) {bar}")

# 输出版本
print(f"\n耗时: {time.time()-t0:.0f}s")
print(f"\n写结果文件...")

# 保存monthly明细
df_detail = pd.DataFrame(all_results)
df_detail.to_csv('D:/AgentQuant/our/cache/wf_monthly_2002.csv', index=False)

# 保存折汇总
df_folds = pd.DataFrame(pair_history)
df_folds.to_csv('D:/AgentQuant/our/cache/wf_folds_2002.csv', index=False)

print("完成.")
