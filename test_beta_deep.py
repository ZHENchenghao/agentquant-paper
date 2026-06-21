# -*- coding: utf-8 -*-
"""β剥离深挖 · 多窗口+残差因子重构
====================================
改进: ①36/60/120月多窗口β ②Bayesian shrinkage ③残差波动率结构
测试: 残差空间里, 哪些因子有截面预测力?
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats
t0 = time.time()

print("=" * 70)
print("β剥离深挖")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

ind = con.execute("""SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2005-01-01' ORDER BY industry, trade_date""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])

market = con.execute("""SELECT trade_date, AVG(close) as mkt_close FROM kline_daily
    WHERE trade_date >= DATE '2005-01-01' GROUP BY trade_date ORDER BY trade_date""").df()
market['trade_date'] = pd.to_datetime(market['trade_date'])
con.close()

ind['month'] = ind['trade_date'].dt.to_period('M')
ind_m = ind.groupby(['industry','month'])['close'].last().reset_index()
ind_m['month'] = ind_m['month'].dt.to_timestamp()
ind_m['ret'] = ind_m.groupby('industry')['close'].pct_change()

market['month'] = market['trade_date'].dt.to_period('M')
mkt_m = market.groupby('month')['mkt_close'].last().reset_index()
mkt_m['month'] = mkt_m['month'].dt.to_timestamp()
mkt_m['mkt_ret'] = mkt_m['mkt_close'].pct_change()

merged = ind_m.merge(mkt_m[['month','mkt_ret']], on='month', how='inner')
merged = merged.dropna(subset=['ret','mkt_ret']).sort_values(['industry','month'])

# === β估计: 多窗口 ===
WINDOWS = [36, 60, 120]
for w in WINDOWS:
    print(f'  β估计 {w}月窗口...')
    for ind_name, grp in merged.groupby('industry'):
        idx = grp.index
        rets = grp['ret'].values; mkt_rets = grp['mkt_ret'].values
        beta = np.full(len(grp), np.nan); resid = np.full(len(grp), np.nan)
        for i in range(w, len(grp)):
            y = rets[i-w:i]; x = mkt_rets[i-w:i]
            valid = ~(np.isnan(y) | np.isnan(x))
            if valid.sum() > max(15, w*0.5):
                slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])
                beta[i] = slope
                if not np.isnan(rets[i]) and not np.isnan(mkt_rets[i]):
                    resid[i] = rets[i] - (intercept + slope * mkt_rets[i])
        merged.loc[idx, f'beta_{w}m'] = beta
        merged.loc[idx, f'resid_{w}m'] = resid

# Bayesian shrinkage (toward 1.0)
for w in WINDOWS:
    col = f'beta_{w}m'
    merged[f'beta_{w}m_shrink'] = merged[col] * 0.7 + 1.0 * 0.3
    # 重算残差
    for ind_name, grp in merged.groupby('industry'):
        idx = grp.index
        for i in range(w, len(grp)):
            if not np.isnan(grp.iloc[i]['ret']) and not np.isnan(grp.iloc[i]['mkt_ret']):
                b = grp.iloc[i][f'beta_{w}m_shrink']
                if not np.isnan(b):
                    merged.loc[idx[i], f'resid_shrink_{w}m'] = (
                        grp.iloc[i]['ret'] - b * grp.iloc[i]['mkt_ret'])

# === 残差上的因子 ===
merged = merged.dropna(subset=['ret'])
for w in WINDOWS:
    for suffix in ['', '_shrink']:
        rcol = f'resid{suffix}_{w}m'
        if rcol not in merged.columns: continue

        # 残差动量
        merged[f'{rcol}_mom1'] = merged.groupby('industry')[rcol].shift(1)
        merged[f'{rcol}_mom3'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(3).sum())
        merged[f'{rcol}_mom12'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(12).sum())
        # 残差波动
        merged[f'{rcol}_vol3'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(3).std())
        merged[f'{rcol}_vol12'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(12).std())
        # 残差极端值
        merged[f'{rcol}_max'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(6).max())
        merged[f'{rcol}_min'] = merged.groupby('industry')[rcol].transform(lambda x: x.rolling(6).min())

# 目标
merged['fwd_resid_60m'] = merged.groupby('industry')['resid_60m'].shift(-1)
merged['fwd_ret'] = merged.groupby('industry')['ret'].shift(-1)
merged = merged.dropna(subset=['fwd_ret'])

print(f'\n[1] 数据: {len(merged)}行, {merged["month"].nunique()}月')

# === WF IC: 残差因子 vs 原始因子 ===
YEARS = sorted(set(d.year for d in merged['month']))
TRAIN = 5; WF_START = YEARS[0]+TRAIN+1

# 测试因子
TEST_FACTORS = [
    ('resid_60m_mom1', '残差动量1月(60mβ)', 'fwd_resid_60m'),
    ('resid_60m_mom3', '残差动量3月(60mβ)', 'fwd_resid_60m'),
    ('resid_60m_vol3', '残差波动(60mβ)', 'fwd_resid_60m'),
    ('resid_shrink_60m_mom1', '收缩残差动量', 'fwd_resid_60m'),
    ('resid_36m_mom1', '残差动量(36mβ)', 'fwd_resid_60m'),
    ('resid_120m_mom1', '残差动量(120mβ)', 'fwd_resid_60m'),
    ('ret', '原始动量(上月收益)', 'fwd_ret'),
]

print(f'\n[2] WF IC ({WF_START}-{YEARS[-1]})')
print(f'{"因子":<24s} {"预测目标":<14s} {"IC":>8s} {"IR":>7s} {"t":>7s}')

for f, name, target in TEST_FACTORS:
    if f not in merged.columns: continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month']>=ts)&(merged['month']<=te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[f, target])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid[target].rank())
                if not np.isnan(ic): ics.append(ic)
    if len(ics) > 10:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std>0 else 0
        ir = mi/std*np.sqrt(12) if std>0 else 0
        print(f'{name:<24s} {target:<14s} {mi*100:+7.2f}% {ir:+6.2f} {t:+6.2f}')

# === 残差预测 vs 原始 → 策略对比 ===
print(f'\n[3] 策略WF: 残差预测选行业')
# 策略: 用残差动量选行业 → 实际总收益
for label, factor in [('原始动量','ret'),('残差动量60m','resid_60m_mom1'),('残差波动60m','resid_60m_vol3')]:
    long_r = []
    for yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01'); train_e = pd.Timestamp(f'{yr-1}-12-31')
        train = merged[(merged['month']>=train_s)&(merged['month']<=train_e)]
        test = merged[(merged['month']>=ts)&(merged['month']<=te)]
        if len(test)<20 or len(train)<40: continue

        # 训练窗定方向
        dir_ics = []
        for m, grp in train.groupby('month'):
            v = grp.dropna(subset=[factor,'fwd_ret'])
            if len(v)>5:
                ic = v[factor].rank().corr(v['fwd_ret'].rank())
                if not np.isnan(ic): dir_ics.append(ic)
        direction = 1 if (len(dir_ics)>6 and np.mean(dir_ics)>0) else -1

        for m, grp in test.groupby('month'):
            grp = grp.dropna(subset=[factor,'fwd_ret'])
            if len(grp)<5: continue
            grp = grp.copy()
            grp['score'] = grp[factor].rank(pct=True)*direction
            top5 = grp.nlargest(5,'score')
            long_r.append(top5['fwd_ret'].mean()-0.003)

    if long_r:
        la = np.array(long_r); n = len(la)
        lac = np.prod(1+la); la_ann = lac**(12/n)-1
        c = np.cumprod(1+la); mdd = np.min(c/np.maximum.accumulate(c)-1)
        vol = np.std(la)*np.sqrt(12); sh = la_ann/vol if vol>0 else 0
        print(f'  {label:<16s} 年化{la_ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.0f}% ({n}月)')

# 残差结构特征
print(f'\n[4] 残差统计特征:')
for w in [36,60,120]:
    rcol = f'resid_{w}m'
    if rcol in merged.columns:
        vals = merged[rcol].dropna()
        ac = merged.groupby('industry')[rcol].apply(lambda x: x.autocorr(1)).mean()
        var_ratio = vals.var() / merged['ret'].var()
        print(f'  {w}月β残差: 自相关={ac:+.4f} 方差占比={var_ratio*100:.0f}% '
              f'均值={vals.mean()*100:+.3f}% 标准差={vals.std()*100:.2f}%')

print(f'\n耗时: {time.time()-t0:.0f}s')
