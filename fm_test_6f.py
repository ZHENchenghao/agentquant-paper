# -*- coding: utf-8 -*-
"""
Fama-MacBeth两步回归检验 6因子
================================
第一步(时序): 个股收益 ~ 6因子 → 因子暴露β (rolling 60月)
第二步(截面): 每月个股收益 ~ β → 因子溢价λ + Newey-West t检验

如果|t|>2.0 → 因子溢价统计显著
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, warnings
from scipy import stats
warnings.filterwarnings('ignore')
t0 = time.time()

FEATS = ['amihud', 'max_rev', 'price_rev', 'turnover_rev', 'sr5', 'vp_corr']
print("=" * 60)
print("Fama-MacBeth两步回归 · 6因子显著性检验")
print("=" * 60)

# ============ 加载 ============
print("[1] 加载...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
# 月度收益 (下一个月的持有收益)
kline_monthly = con.execute("""
    SELECT ts_code, trade_date, close,
           LEAD(close, 20) OVER(PARTITION BY ts_code ORDER BY trade_date)/close-1 AS fwd_ret_20d
    FROM kline_daily WHERE trade_date >= '2002-01-01'
""").df()
kline_monthly['trade_date'] = pd.to_datetime(kline_monthly['trade_date'])
con.close()

# 月度调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print(f"月度: {len(monthly_dates)}个 ({monthly_dates[0].date()}~{monthly_dates[-1].date()})")

# ============ 第一步: 时序回归 (滚动60月) ============
print("[2] 第一步: 时序回归 → 因子暴露β...")

# 对每个调仓日, 用过去60个月数据回归
betas = {}  # {date: DataFrame(index=ts_code, columns=FEATS)}
ROLLING_MONTHS = 60

for i, rd in enumerate(monthly_dates):
    if i < ROLLING_MONTHS + 1: continue
    past_start = monthly_dates[i - ROLLING_MONTHS]

    # 获取过去60个月的因子+收益数据
    past_data = fn[(fn['trade_date'] >= past_start) & (fn['trade_date'] < rd)].copy()
    past_ret = kline_monthly[(kline_monthly['trade_date'] >= past_start) & (kline_monthly['trade_date'] < rd)].copy()

    merged = past_data.merge(past_ret[['ts_code','trade_date','fwd_ret_20d']], on=['ts_code','trade_date'], how='inner')
    merged = merged.dropna(subset=FEATS + ['fwd_ret_20d'])

    # 对每只个股做时序回归: ret ~ factors
    stock_betas = {}
    for code, grp in merged.groupby('ts_code'):
        if len(grp) < 30: continue  # 至少30个观测
        X = grp[FEATS].values
        y = grp['fwd_ret_20d'].values
        # winsorize extreme values
        y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            stock_betas[code] = beta
        except:
            continue

    if len(stock_betas) >= 100:
        betas[rd] = pd.DataFrame(stock_betas, index=FEATS).T

    if i % 50 == 0:
        print(f"  进度: {i}/{len(monthly_dates)} ({rd.date()}), {len(stock_betas)}只")

print(f"  完成: {len(betas)}个月有β估计")

# ============ 第二步: 截面回归 + Newey-West ============
print("[3] 第二步: 截面回归 → 因子溢价λ...")

lambda_ts = {f: [] for f in FEATS}  # 因子溢价时间序列

for rd in monthly_dates:
    if rd not in betas: continue
    beta_df = betas[rd]

    # 获取当月收益
    ret_data = kline_monthly[kline_monthly['trade_date'] == rd].copy()
    ret_data = ret_data.dropna(subset=['fwd_ret_20d'])

    # 合并β和收益
    reg_df = beta_df.join(ret_data.set_index('ts_code')[['fwd_ret_20d']], how='inner')
    if len(reg_df) < 100: continue

    # winsorize收益
    y = np.clip(reg_df['fwd_ret_20d'].values,
                np.percentile(reg_df['fwd_ret_20d'], 1),
                np.percentile(reg_df['fwd_ret_20d'], 99))
    X = reg_df[FEATS].values

    try:
        lam = np.linalg.lstsq(X, y, rcond=None)[0]
        for j, f in enumerate(FEATS):
            lambda_ts[f].append(lam[j])
    except:
        for f in FEATS:
            lambda_ts[f].append(np.nan)

# Newey-West t检验 (滞后12期, 对应月度数据的1年自相关)
def newey_west_t(series, max_lags=12):
    """Newey-West t-statistic"""
    s = np.array(series)
    s = s[~np.isnan(s)]
    if len(s) < 30: return 0, 0, 0
    T = len(s)
    mean = np.mean(s)
    # 计算自协方差
    gamma0 = np.sum((s - mean) ** 2) / T
    nw_var = gamma0
    for lag in range(1, min(max_lags + 1, T - 1)):
        gamma_lag = np.sum((s[:T-lag] - mean) * (s[lag:] - mean)) / T
        weight = 1 - lag / (max_lags + 1)  # Bartlett kernel
        nw_var += 2 * weight * gamma_lag
    se = np.sqrt(nw_var / T)
    t_stat = mean / se if se > 0 else 0
    # 双侧p值
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), T - 1))
    return mean, t_stat, p_val

print(f"\n{'因子':<16s} {'λ均值':>8s} {'NW-t':>8s} {'p值':>8s} {'显著性':<10s} {'月数':>6s}")
print("-" * 65)
for f in FEATS:
    mean_lam, t_stat, p_val = newey_west_t(lambda_ts[f])
    n = sum(~np.isnan(lambda_ts[f]))
    sig = '✅ 显著' if abs(t_stat) > 2.0 and p_val < 0.05 else \
          ('⚠ 边缘' if abs(t_stat) > 1.65 and p_val < 0.10 else '❌ 不显著')
    print(f"{f:<16s} {mean_lam*100:>+7.2f}% {t_stat:>+7.2f} {p_val:>7.3f} {sig:<10s} {n:>5d}")

# 联合检验
print(f"\n[4] 联合显著性...")
all_lams = np.column_stack([lambda_ts[f] for f in FEATS])
valid_rows = ~np.any(np.isnan(all_lams), axis=1)
n_valid = valid_rows.sum()
print(f"  有效月数: {n_valid}/{len(monthly_dates)}")

# 简单: 检查每个因子λ的均值是否显著>0
for f in FEATS:
    vals = np.array(lambda_ts[f])
    vals = vals[~np.isnan(vals)]
    # 单样本t检验(简化)
    t_simple = np.mean(vals) / (np.std(vals) / np.sqrt(len(vals))) if len(vals) > 0 else 0
    sharpe_lam = np.mean(vals) / np.std(vals) * np.sqrt(12) if np.std(vals) > 0 else 0
    print(f"  {f}: λ夏普={sharpe_lam:+.2f} 简单t={t_simple:+.2f} 均值={np.mean(vals)*100:+.3f}%/月")

print(f"\n耗时: {time.time()-t0:.0f}s")
