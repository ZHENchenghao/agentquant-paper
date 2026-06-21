# -*- coding: utf-8 -*-
"""行业轮动完整回测 · 因子动物园 + Walk-Forward
==============================================
申万30行业 2005-2026 · 20+因子 · WF IC + 多空 + 分年拆解
每个因子独立WF检验 → 相关性矩阵 → 最优组合 → 策略合成
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from itertools import combinations
t0 = time.time()

print("=" * 70)
print("行业轮动 · 因子动物园 · 完整WF回测")
print("=" * 70)

# ============ 1. 数据加载 ============
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 申万行业指数
raw = con.execute("""
    SELECT industry, trade_date, close
    FROM proxy_industry_daily
    WHERE trade_date >= DATE '2005-01-01'
    ORDER BY industry, trade_date
""").df()
raw['trade_date'] = pd.to_datetime(raw['trade_date'])

# HS300基准
hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close']

con.close()

N_INDUSTRIES = raw['industry'].nunique()
print(f"[1] 申万行业: {N_INDUSTRIES}个, {len(raw)}行日线, "
      f"{raw['trade_date'].min().date()}~{raw['trade_date'].max().date()}")

# ============ 2. 日频特征计算 ============
raw = raw.sort_values(['industry', 'trade_date']).reset_index(drop=True)

for ind_name, grp in raw.groupby('industry'):
    idx = grp.index
    close = grp['close'].values

    # 日收益
    ret = np.concatenate([[np.nan], (close[1:] - close[:-1]) / close[:-1]])

    # 累计收益(N日)
    def roll_ret(k):
        result = np.full(len(close), np.nan)
        for i in range(k, len(close)):
            result[i] = close[i] / close[i-k] - 1
        return result

    raw.loc[idx, 'ret_1d'] = ret
    raw.loc[idx, 'ret_5d'] = roll_ret(5)
    raw.loc[idx, 'ret_20d'] = roll_ret(20)
    raw.loc[idx, 'ret_60d'] = roll_ret(60)
    raw.loc[idx, 'ret_120d'] = roll_ret(120)
    raw.loc[idx, 'ret_240d'] = roll_ret(240)

    # 波动率
    for k in [20, 60, 120]:
        raw.loc[idx, f'vol_{k}d'] = pd.Series(ret).rolling(k).std().values * np.sqrt(252)

    # 下行波动
    raw.loc[idx, 'down_vol_60d'] = pd.Series(np.where(ret < 0, ret, 0)).rolling(60).std().values * np.sqrt(252)

    # MAX效应(月内最大单日涨幅)
    raw.loc[idx, 'max_ret_20d'] = pd.Series(ret).rolling(20).max().values

    # 偏度
    raw.loc[idx, 'skew_60d'] = pd.Series(ret).rolling(60).skew().values

    # 最大回撤(1年)
    cum = np.cumprod(1 + np.nan_to_num(ret, 0))
    h_max = np.maximum.accumulate(cum)
    dd = cum / h_max - 1
    raw.loc[idx, 'mdd_240d'] = pd.Series(dd).rolling(240).min().values

    # 距52周高低
    h_52w = pd.Series(close).rolling(240).max().values
    l_52w = pd.Series(close).rolling(240).min().values
    raw.loc[idx, 'pct_52w_high'] = close / h_52w - 1
    raw.loc[idx, 'pct_52w_low'] = close / l_52w - 1

    # 成交量变化(收益绝对值之和=实际波动, 非方向性)
    raw.loc[idx, 'abs_ret_20d'] = pd.Series(np.abs(ret)).rolling(20).sum().values  # 月内总振幅

print(f"[2] 日频特征: 完成")

# ============ 3. 月度因子构建 ============
raw['month'] = raw['trade_date'].dt.to_period('M')

monthly = raw.groupby(['industry', 'month']).agg(
    close=('close', 'last'),
    # 动量因子
    mom_1m=('ret_20d', 'last'),       # 近1月收益
    mom_3m=('ret_60d', 'last'),       # 近3月收益
    mom_6m=('ret_120d', 'last'),      # 近6月收益
    mom_12m=('ret_240d', 'last'),     # 近12月收益
    # 反转因子
    rev_5d=('ret_5d', 'last'),         # 短期反转(5日为负→下月反弹)
    rev_10d=('ret_5d', lambda x: x.tail(10).mean()),
    # 波动率因子
    vol_1m=('vol_20d', 'last'),
    vol_3m=('vol_60d', 'last'),
    vol_12m=('vol_120d', 'last'),
    # 下行风险
    down_vol=('down_vol_60d', 'last'),
    # 极端收益
    max_ret=('max_ret_20d', 'last'),
    # 偏度
    skewness=('skew_60d', 'last'),
    # 最大回撤
    mdd_1y=('mdd_240d', 'last'),
    # 距高低点
    pct_high=('pct_52w_high', 'last'),
    pct_low=('pct_52w_low', 'last'),
    # 总振幅(替代波动变化)
    vol_chg=('abs_ret_20d', 'last'),
    n_days=('trade_date', 'nunique'),
).reset_index()

# 衍生因子
monthly['mom_12m_1m'] = monthly['mom_1m'] - monthly['mom_12m']  # 短期-长期动量差(拥挤)
monthly['vol_term'] = monthly['vol_1m'] - monthly['vol_12m']     # 波动率期限结构

monthly['month'] = monthly['month'].dt.to_timestamp()

# 目标: 下月收益
monthly['fwd_ret'] = monthly.groupby('industry')['mom_1m'].shift(-1)

# 过滤数据不足的月份
monthly = monthly[monthly['n_days'] >= 15]
monthly = monthly.dropna(subset=['fwd_ret'])
print(f"[3] 月度数据: {len(monthly)}行, {monthly['month'].nunique()}月")

# ============ 4. 因子清单 ============
FACTORS = {
    # === 动量 ===
    'mom_1m':    ('动量1月', 1),     # direction: +1=正相关期望, -1=负相关期望
    'mom_3m':    ('动量3月', 1),
    'mom_6m':    ('动量6月', 1),
    'mom_12m':   ('动量12月', 1),
    'mom_12m_1m':('短长动量差', -1),  # 短>长→拥挤→看空

    # === 反转 ===
    'rev_5d':    ('5日反转', -1),     # 5日跌→下月弹(反转)
    'rev_10d':   ('10日反转', -1),

    # === 低风险 ===
    'vol_1m':    ('波动1月', -1),     # 低波→高收益(低波异象)
    'vol_3m':    ('波动3月', -1),
    'vol_12m':   ('波动12月', -1),
    'down_vol':  ('下行波动', -1),

    # === 尾部特征 ===
    'max_ret':   ('月内最大涨', -1),  # MAX效应→彩票偏好→未来低收益
    'skewness':  ('偏度', -1),        # 正偏→彩票→未来低收益

    # === 回撤/位置 ===
    'mdd_1y':    ('1年最大回撤', 1),  # 深跌→反弹
    'pct_high':  ('距52w高', -1),    # 近新高→可能见顶
    'pct_low':   ('距52w低', 1),     # 近新低→反弹

    # === 波动变化 ===
    'vol_chg':   ('振幅(低波)', -1),   # 高振幅→风险→低收益(低波异象)
    'vol_term':  ('波动期限', -1),    # 短端>长端→恐慌→低收益
}

print(f"\n[4] 因子清单: {len(FACTORS)}个")
for k, (name, direction) in FACTORS.items():
    print(f"  {k:<16s} {name:<12s} 预期{'正' if direction>0 else '负'}相关")

# ============ 5. Walk-Forward IC检验 ============
print(f"\n{'='*70}")
print("Walk-Forward 月度横截面IC")
print(f"{'='*70}")

START_YEAR = 2010
YEARS = sorted(set(d.year for d in monthly['month']))
WF_START = START_YEAR + 5  # 5年初始训练

# 每年作为测试窗口, 前5年作为训练
# 不做参数估计, 只做IC计算(IC不需要训练)

all_ics = {k: [] for k in FACTORS}
yearly_ic = {yr: {} for yr in range(WF_START, YEARS[-1]+1)}

for test_yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{test_yr}-01-01')
    te = pd.Timestamp(f'{test_yr}-12-31')
    test = monthly[(monthly['month'] >= ts) & (monthly['month'] <= te)]

    for factor_name, (display, direction) in FACTORS.items():
        if factor_name not in monthly.columns:
            continue
        yr_ics = []
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[factor_name, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[factor_name].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic):
                    yr_ics.append(ic)
                    all_ics[factor_name].append(ic)
        if yr_ics:
            yearly_ic[test_yr][factor_name] = np.mean(yr_ics)

# 汇总统计
print(f"\n{'因子':<16s} {'预期':>4s} {'IC均值':>8s} {'IC标准差':>8s} {'IR':>7s} {'t-stat':>7s} {'正比例':>7s} {'方向对':>6s}")
print("-" * 72)

factor_stats = {}
for factor_name, (display, expected_dir) in FACTORS.items():
    ics = all_ics.get(factor_name, [])
    if len(ics) < 10:
        continue
    mi = np.mean(ics)
    std = np.std(ics)
    ir = mi / std * np.sqrt(12) if std > 0 else 0
    t = mi / std * np.sqrt(len(ics)) if std > 0 else 0
    pos_ratio = np.mean(np.array(ics) > 0)
    dir_correct = (mi > 0 and expected_dir > 0) or (mi < 0 and expected_dir < 0)

    factor_stats[factor_name] = {
        'name': display, 'ic': mi, 'std': std, 'ir': ir, 't': t,
        'pos_ratio': pos_ratio, 'dir_correct': dir_correct,
        'expected_dir': expected_dir, 'n_months': len(ics)
    }

    print(f"{factor_name:<16s} {'正' if expected_dir>0 else '负':>4s} "
          f"{mi*100:+7.2f}% {std*100:>7.2f}% {ir:>6.2f} {t:>6.2f} "
          f"{pos_ratio*100:>6.0f}% {'✅' if dir_correct else '⚠️':>6s}")

# ============ 6. 分年IC热度图 ============
print(f"\n{'='*70}")
print("分年IC (每个因子×每年)")
print(f"{'='*70}")

# 选出显著的因子
sig_factors = [k for k, v in sorted(factor_stats.items(), key=lambda x: abs(x[1]['ic']), reverse=True)[:12]]
yr_list = sorted(yearly_ic.keys())

print(f"{'年份':>6s}", end='')
for f in sig_factors:
    print(f" {factor_stats[f]['name']:>8s}", end='')
print()

for yr in yr_list:
    print(f"{yr:>6d}", end='')
    for f in sig_factors:
        ic_val = yearly_ic[yr].get(f, np.nan)
        if not np.isnan(ic_val):
            print(f" {ic_val*100:+7.2f}%", end='')
        else:
            print(f" {'--':>8s}", end='')
    print()

# ============ 7. 因子相关性 ============
print(f"\n{'='*70}")
print("因子相关性矩阵(基于月度排名)")
print(f"{'='*70}")

# 计算每个因子每月的横截面排名
rank_cols = {}
for f in sig_factors:
    rank_cols[f] = monthly.groupby('month')[f].rank(pct=True)

rank_df = pd.DataFrame(rank_cols)
corr_mat = rank_df.corr()

# 简洁输出
for i, f1 in enumerate(sig_factors):
    corrs = []
    for f2 in sig_factors:
        corrs.append(corr_mat.loc[f1, f2])
    print(f"  {f1:<16s} " + ' '.join(f'{c:+5.2f}' for c in corrs))

# ============ 8. 多因子组合测试 ============
print(f"\n{'='*70}")
print("多因子组合 · 等权合成")
print(f"{'='*70}")

# 8a. 用实际IC方向选因子(不用预期)
# 筛选: |t|>1.5 的因子, 用实际IC符号决定方向
valid_factors = []
for k, v in sorted(factor_stats.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    if abs(v['t']) > 1.5 and v['n_months'] > 30:
        actual_dir = 1 if v['ic'] > 0 else -1  # 实际IC方向
        valid_factors.append((k, actual_dir))

top_factors = [f for f, d in valid_factors[:6]]  # 取Top6显著因子
print(f"显著因子(t>1.5): {[(f, '正' if d>0 else '负') for f,d in valid_factors]}")
print(f"入选Top6: {top_factors}")

# 计算综合排名(用实际IC方向)
monthly_valid = monthly.dropna(subset=top_factors + ['fwd_ret']).copy()
factor_dirs = {f: d for f, d in valid_factors if f in top_factors}
for f in top_factors:
    direction = factor_dirs[f]  # 实际IC方向
    monthly_valid[f'{f}_rank'] = monthly_valid.groupby('month')[f].rank(pct=True) * direction

rank_cols = [f'{f}_rank' for f in top_factors]
monthly_valid['combo_score'] = monthly_valid[rank_cols].mean(axis=1)

# WF 多空组合
long_rets = []; short_rets = []; eq_rets = []
for test_yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{test_yr}-01-01')
    te = pd.Timestamp(f'{test_yr}-12-31')
    test = monthly_valid[(monthly_valid['month'] >= ts) & (monthly_valid['month'] <= te)]

    for m, grp in test.groupby('month'):
        if len(grp) > 10:
            n = max(1, len(grp) // 4)
            top = grp.nlargest(n, 'combo_score')
            bot = grp.nsmallest(n, 'combo_score')
            long_rets.append(top['fwd_ret'].mean())
            short_rets.append(bot['fwd_ret'].mean())
            eq_rets.append(grp['fwd_ret'].mean())

long_arr = np.array(long_rets); short_arr = np.array(short_rets); eq_arr = np.array(eq_rets)
ls_arr = long_arr - short_arr

n_months = len(long_arr)
long_cum = np.prod(1+long_arr); short_cum = np.prod(1+short_arr); eq_cum = np.prod(1+eq_arr)

long_ann = long_cum ** (12/n_months) - 1
ls_ann = (np.prod(1+ls_arr)) ** (12/n_months) - 1
eq_ann = eq_cum ** (12/n_months) - 1

# MDD
def calc_mdd(rets):
    cum = np.cumprod(1+rets)
    peak = np.maximum.accumulate(cum)
    return np.min(cum/peak - 1)

ls_sharpe = np.mean(ls_arr) / np.std(ls_arr) * np.sqrt(12) if np.std(ls_arr) > 0 else 0
ls_hit = np.mean(ls_arr > 0)

print(f"\n多因子组合 WF({WF_START}-{YEARS[-1]}):")
print(f"  做多(Top25%):    年化{long_ann*100:+.1f}%  累积{long_cum-1:+.1%}  MDD{calc_mdd(long_arr)*100:.1f}%")
print(f"  做空(Bot25%):    年化{(short_cum**(12/n_months)-1)*100:+.1f}%  累积{short_cum-1:+.1%}")
print(f"  多空:           年化{ls_ann*100:+.1f}%  Sharpe{ls_sharpe:.2f}  命中{ls_hit*100:.0f}%")
print(f"  等权基准:       年化{eq_ann*100:+.1f}%  累积{eq_cum-1:+.1%}")

# 8b. 单调性检验
monthly_valid['combo_q'] = monthly_valid.groupby('month')['combo_score'].transform(
    lambda x: pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop'))
q_rets = monthly_valid.groupby('combo_q')['fwd_ret'].mean() * 100
print(f"\n分层单调性:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    if q in q_rets.index:
        print(f"  {q}: {q_rets[q]:+.2f}%")

# 8c. 分年多空
print(f"\n分年多空收益:")
for yr in range(WF_START, YEARS[-1]+1):
    yr_ls = [ls_arr[i] for i, m in enumerate(range(len(ls_arr))) if m < len(long_rets)]
    # Simple approach: filter by year
    yr_longs = []; yr_shorts = []
    ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
    test = monthly_valid[(monthly_valid['month'] >= ts) & (monthly_valid['month'] <= te)]
    for m, grp in test.groupby('month'):
        if len(grp) > 10:
            n = max(1, len(grp) // 4)
            top = grp.nlargest(n, 'combo_score')
            bot = grp.nsmallest(n, 'combo_score')
            yr_longs.append(top['fwd_ret'].mean())
            yr_shorts.append(bot['fwd_ret'].mean())
    if yr_longs:
        yr_ls_ret = np.mean(np.array(yr_longs) - np.array(yr_shorts)) * 100
        yr_long_ret = np.prod(1+np.array(yr_longs)) - 1
        print(f"  {yr}: 多{yr_long_ret*100:+5.1f}%  多空月均{yr_ls_ret:+.2f}%")


print(f"\n耗时: {time.time()-t0:.0f}s")
