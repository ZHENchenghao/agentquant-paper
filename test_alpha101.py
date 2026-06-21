# -*- coding: utf-8 -*-
"""
Alpha101 novel factors: implement + IC test
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE = 'cache/factors_all.parquet'

print('=' * 70)
print('Alpha101 Novel Factors Test')
print('=' * 70)

con = duckdb.connect(DB, read_only=True)

# Compute novel Alpha101 factors directly in SQL
print('\n[1] Computing Alpha101 factors...')

alpha_sql = """
WITH base AS (
    SELECT ts_code, trade_date,
           close, open, high, low, vol AS volume,
           LAG(close, 1) OVER w AS close_l1,
           LAG(vol, 1) OVER w AS vol_l1,
           LAG(open, 1) OVER w AS open_l1,
           LAG(high, 1) OVER w AS high_l1,
           LAG(low, 1) OVER w AS low_l1
    FROM kline_daily
    WHERE trade_date >= '2016-01-01'
    WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
),
-- Daily metrics
daily AS (
    SELECT *,
        -- alpha_101: intraday position (close-open)/(high-low)
        (close - open) / NULLIF(high - low, 0) AS a101_intraday,
        -- alpha_012: volume direction * price direction
        SIGN(volume - vol_l1) * (-(close - close_l1) / NULLIF(close_l1, 0)) AS a012_volprice,
        -- alpha_053: 12-day win rate
        CASE WHEN close > close_l1 THEN 1 ELSE 0 END AS up_day,
        -- alpha_002 support: rank of delta log volume
        LN(volume / NULLIF(vol_l1, 0)) AS delta_logvol,
        -- close change
        (close - close_l1) / NULLIF(close_l1, 0) AS ret_1d
    FROM base
    WHERE close_l1 IS NOT NULL
),
-- Rolling computations
rolling AS (
    SELECT *,
        -- alpha_053: rolling 12-day up-day ratio
        AVG(up_day) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 11 PRECEDING) AS a053_upratio,
        -- alpha_002: -corr(rank(delta_logvol,2), rank(ret_1d),6)
        -- Simplified: rolling rank correlation of vol change vs return
        -- alpha_012 smoothed: 5d avg of a012_volprice
        AVG(a012_volprice) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 4 PRECEDING) AS a012_smooth
    FROM daily
)
SELECT ts_code, trade_date,
       a101_intraday,
       a012_volprice,
       a012_smooth,
       a053_upratio
FROM rolling
WHERE trade_date >= '2016-02-01'
"""

alpha_df = con.execute(alpha_sql).df()
print('  Computed: %d rows, %d stocks' % (len(alpha_df), alpha_df['ts_code'].nunique()))
print('  Date range: %s ~ %s' % (alpha_df['trade_date'].min(), alpha_df['trade_date'].max()))

# Merge with factor cache + target
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1) - (x.fc/x.close-1) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
con.close()

factors = pd.read_parquet(CACHE)
for d in [factors, target, alpha_df]:
    d['trade_date'] = d['trade_date'].astype(str)

# Merge
df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(alpha_df, on=['ts_code', 'trade_date'], how='left')

alpha_cols = ['a101_intraday', 'a012_volprice', 'a012_smooth', 'a053_upratio']
for c in alpha_cols:
    df[c] = df[c].fillna(0)

print('  Merged: %d rows, alpha coverage: %d' % (len(df), (df['a053_upratio'] != 0).sum()))

# ============================================================
# IC analysis
# ============================================================
print('\n[2] Individual factor IC vs excess_ret...')

for col in alpha_cols:
    valid = df[[col, 'excess_ret']].dropna()
    if len(valid) > 1000:
        ic, p = stats.spearmanr(valid[col], valid['excess_ret'])
        print('  %-20s IC=%+.4f  p=%.4f  n=%d' % (col, ic, p, len(valid)))

# ============================================================
# Backtest: baseline vs baseline + alpha101
# ============================================================
print('\n[3] Backtest...')

exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date', 'excess_ret']
base_feats = [c for c in df.columns if c not in exclude and c not in alpha_cols
              and df[c].dtype in ('float64','float32','int64','int32')]
upgrade_feats = base_feats + alpha_cols
upgrade_feats = [f for f in upgrade_feats if f in df.columns]

print('  Base: %d, +Alpha101: %d' % (len(base_feats), len(upgrade_feats)))

def bt(train_df, test_df, feats):
    flist = [f for f in feats if f in train_df.columns]
    X_tr = train_df[flist].fillna(train_df[flist].median())
    y_tr = train_df['excess_ret'].fillna(0)
    X_te = test_df[flist].fillna(train_df[flist].median())

    m = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                       subsample=0.8, colsample_bytree=0.8,
                       n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    mask = ~np.isnan(pred) & ~np.isnan(test_df['excess_ret'].values)
    ic, _ = stats.spearmanr(pred[mask], test_df['excess_ret'].values[mask])

    te2 = test_df.copy(); te2['pred'] = pred
    te2['ym'] = pd.to_datetime(te2['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in te2.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        mrets.append(top['excess_ret'].mean())
    if len(mrets) < 3: return {'ic': ic, 'sh': 0, 'mdd': 0}
    rets = np.array(mrets)
    ann = np.mean(rets)*12
    vol = np.std(rets, ddof=1)*np.sqrt(12) if len(rets)>2 else 0.01
    sh = ann/vol if vol>0 else 0
    mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    return {'ic': ic, 'sh': sh, 'mdd': mdd}

windows = [
    ('2020->2021', '2017-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2021->2022', '2018-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2022->2023', '2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2023->2024', '2020-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

print('  %-14s | %8s %8s %8s | %8s %8s | %s' % ('Window', 'BL_IC', 'A1_IC', 'dIC', 'BL_Sh', 'A1_Sh', 'Win?'))
print('  ' + '-' * 75)

all_res = []
for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date']>=tr_s)&(df['trade_date']<=tr_e)]
    te = df[(df['trade_date']>=te_s)&(df['trade_date']<=te_e)]
    if len(tr) < 5000: continue
    bl = bt(tr, te, base_feats)
    a1 = bt(tr, te, upgrade_feats)
    all_res.append((label, bl, a1))
    dic = a1['ic'] - bl['ic']
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f | %s' % (
        label, bl['ic'], a1['ic'], dic, bl['sh'], a1['sh'],
        'A101' if a1['ic']>bl['ic'] else 'BASE'))

if all_res:
    avg_bl = np.mean([r[1]['ic'] for r in all_res])
    avg_a1 = np.mean([r[2]['ic'] for r in all_res])
    wins = sum(1 for r in all_res if r[2]['ic']>r[1]['ic'])
    print('  %-14s | %+.4f %+.4f %+.4f | Wins: %d/%d' % (
        'AVERAGE', avg_bl, avg_a1, avg_a1-avg_bl, wins, len(all_res)))

print('\nDone.')
