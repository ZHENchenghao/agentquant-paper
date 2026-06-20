# -*- coding: utf-8 -*-
"""
Fama-MacBeth 交互项检验
=======================
H0: 单因子无截面溢价 (已验证)
H1: 乘法交互对(冷门×触发器)有显著截面溢价
测试: Amihud×Turnover, Amihud×MaxRev, Amihud×SR5, Turnover×SR5
      以及最稳定的 Price×Turnover
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, warnings
from scipy import stats
warnings.filterwarnings('ignore')
t0 = time.time()

# 测试的交互对
INTERACTION_PAIRS = [
    ('amihud', 'turnover_rev', 'Amihud×Turnover'),
    ('amihud', 'max_rev', 'Amihud×MaxRev'),
    ('amihud', 'sr5', 'Amihud×SR5'),
    ('turnover_rev', 'sr5', 'Turnover×SR5'),
    ('price_rev', 'turnover_rev', 'Price×Turnover'),
]

print("=" * 60)
print("Fama-MacBeth 交互项截面溢价检验")
print("=" * 60)

# 加载
print("[1] 加载...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline_monthly = con.execute("""
    SELECT ts_code, trade_date, close,
           LEAD(close, 20) OVER(PARTITION BY ts_code ORDER BY trade_date)/close-1 AS fwd_ret_20d
    FROM kline_daily WHERE trade_date >= '2002-01-01'
""").df()
kline_monthly['trade_date'] = pd.to_datetime(kline_monthly['trade_date'])
con.close()

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 预计算交互项 (截面排名乘法, 每月独立)
print("[2] 预计算交互项得分...")
for (fa, fb, label) in INTERACTION_PAIRS:
    fn[fa+'_r'] = fn.groupby('trade_date')[fa].rank(pct=True)
    fn[fb+'_r'] = fn.groupby('trade_date')[fb].rank(pct=True)
    pair_name = fa[:4] + '_x_' + fb[:4]
    fn[pair_name] = fn[fa+'_r'] * fn[fb+'_r']
    fn.drop(columns=[fa+'_r', fb+'_r'], inplace=True)

# FM第二步: 每月截面回归 (收益 ~ 交互项得分)
print("[3] 截面回归: 月收益 ~ 交互项得分...")

pair_names = [fa[:4]+'_x_'+fb[:4] for fa, fb, _ in INTERACTION_PAIRS]
lambda_ts = {p: [] for p in pair_names}

for i, rd in enumerate(monthly_dates):
    if i < 12: continue  # 跳过前12个月(需要交互项稳定)

    day = fn[fn['trade_date'] == rd].copy()
    ret_data = kline_monthly[kline_monthly['trade_date'] == rd].copy()

    merged = day.merge(ret_data[['ts_code', 'fwd_ret_20d']], on='ts_code', how='inner')
    merged = merged.dropna(subset=pair_names + ['fwd_ret_20d'])
    if len(merged) < 200: continue

    # 截面回归
    X = merged[pair_names].values
    y = np.clip(merged['fwd_ret_20d'].values,
                np.percentile(merged['fwd_ret_20d'], 1),
                np.percentile(merged['fwd_ret_20d'], 99))

    try:
        lam = np.linalg.lstsq(X, y, rcond=None)[0]
        for j, p in enumerate(pair_names):
            lambda_ts[p].append(lam[j] * 100)  # 转为百分比
    except:
        for p in pair_names:
            lambda_ts[p].append(np.nan)

# Newey-West t检验
def newey_west(series, max_lags=12):
    s = np.array(series)
    s = s[~np.isnan(s)]
    if len(s) < 30: return 0, 0, 0
    T = len(s); mean = np.mean(s)
    nw_var = np.sum((s-mean)**2)/T
    for lag in range(1, min(max_lags+1, T-1)):
        gamma = np.sum((s[:T-lag]-mean)*(s[lag:]-mean))/T
        nw_var += 2*(1-lag/(max_lags+1))*gamma
    se = np.sqrt(nw_var/T)
    t_stat = mean/se if se > 0 else 0
    p_val = 2*(1-stats.t.cdf(abs(t_stat), T-1))
    return mean, t_stat, p_val

print('\n' + '=' * 70)
print('%s %10s %10s %10s %12s %8s' % ('交互对', '月均溢价', 'NW-t', 'p值', 'λ夏普(年)', '月数'))
print('-' * 70)
for (fa, fb, label) in INTERACTION_PAIRS:
    p = fa[:4]+'_x_'+fb[:4]
    mean_lam, t_stat, p_val = newey_west(lambda_ts[p])
    n = sum(~np.isnan(lambda_ts[p]))
    sharpe_lam = np.mean([x for x in lambda_ts[p] if not np.isnan(x)]) / \
                 np.std([x for x in lambda_ts[p] if not np.isnan(x)]) * np.sqrt(12) \
                 if n > 0 and np.std([x for x in lambda_ts[p] if not np.isnan(x)]) > 0 else 0
    sig = 'SIG' if abs(t_stat) > 2.0 and p_val < 0.05 else ('WEAK' if abs(t_stat) > 1.65 else 'NO')
    print('%s %+9.4f%% %+9.2f %9.3f %+11.2f %7d  [%s]' % (label, mean_lam, t_stat, p_val, sharpe_lam, n, sig))

# 对比单因子结果
print('\n=== 对比: 单因子 vs 交互项 ===')
print('因子/交互对     NW-t    显著性')
single_t = {'amihud':1.77, 'max_rev':-1.34, 'price_rev':0.64, 'turnover_rev':-1.05, 'sr5':0.18, 'vp_corr':-1.86}
for (fa, fb, label) in INTERACTION_PAIRS:
    p = fa[:4]+'_x_'+fb[:4]
    _, t_stat, _ = newey_west(lambda_ts[p])
    t_a = abs(single_t.get(fa, 0)); t_b = abs(single_t.get(fb, 0))
    print('%s: t=%+.2f (vs 单因子 %s:%.2f %s:%.2f)' % (label, t_stat, fa, t_a, fb, t_b))

print('\n耗时: %.0fs' % (time.time()-t0))
