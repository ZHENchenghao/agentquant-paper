# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('Neutralization: Linear(Alpha) vs Quadratic(vFinal)')
print('=' * 70)

# Load
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
industry = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn = 1
""").df()
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1) - (x.fc/x.close-1) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
mcap_raw = con.execute("""
    SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
           close * total_share / 10000 AS mcap
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df()
con.close()

factors = pd.read_parquet('cache/factors_all.parquet')
for d in [factors, target, mcap_raw]:
    d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(mcap_raw, on=['ts_code', 'trade_date'], how='left')

exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret', 'ind_name', 'mcap']
raw_feats = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64','float32','int64','int32')]
print('  Data: %d rows, %d raw factors' % (len(df), len(raw_feats)))

# Prepare neutralization inputs
df['ln_mcap'] = np.log(df['mcap'].clip(lower=1e6))
df['ln_mcap_sq'] = df['ln_mcap'] ** 2
ind_dummies = pd.get_dummies(df['ind_name'], prefix='ind', dummy_na=False)
ind_dummies = ind_dummies.fillna(0).astype(float)

# Common factor columns (drop those with too many NaN)
valid_cols = [f for f in raw_feats if df[f].notna().sum() > len(df)*0.5]
print('  Valid factors: %d' % len(valid_cols))

# Fill NaN in factor values with median
y_df = df[valid_cols].copy()
for c in valid_cols:
    y_df[c] = y_df[c].fillna(y_df[c].median())

y = y_df.values.astype(float)

# ============================================================
# Method A: Linear (alphasickle)
# ============================================================
X_a = pd.concat([df['ln_mcap'], ind_dummies], axis=1).fillna(0).values.astype(float)
model_a = LinearRegression(fit_intercept=False)
model_a.fit(X_a, y)
resid_a = y - X_a @ model_a.coef_.T

for i, col in enumerate(valid_cols):
    df[col + '_a'] = resid_a[:, i]
feats_a = [c + '_a' for c in valid_cols]

# ============================================================
# Method V: Quadratic (vFinal)
# ============================================================
X_v = pd.concat([df['ln_mcap'], df['ln_mcap_sq'], ind_dummies], axis=1).fillna(0).values.astype(float)
model_v = LinearRegression(fit_intercept=False)
model_v.fit(X_v, y)
resid_v = y - X_v @ model_v.coef_.T

for i, col in enumerate(valid_cols):
    df[col + '_v'] = resid_v[:, i]
feats_v = [c + '_v' for c in valid_cols]

# Residual mcap correlation check
df_check = df[df['trade_date'] >= '2024-01-01']
for label, fcols in [('Linear(Alpha)', feats_a[:6]), ('Quadratic(vFinal)', feats_v[:6])]:
    corrs = []
    for fc in fcols:
        if fc in df_check.columns:
            v = df_check[[fc, 'ln_mcap']].dropna()
            if len(v) > 100:
                c, _ = stats.spearmanr(v[fc], v['ln_mcap'])
                corrs.append(abs(c))
    avg = np.mean(corrs) if corrs else 0
    print('  %s avg |IC| with market cap: %.6f' % (label, avg))

# ============================================================
# Rolling backtest
# ============================================================
print('\n  %-14s | %8s %8s %8s | %8s %8s' % (
    'Window', 'A_IC', 'V_IC', 'Win?', 'A_Sh', 'V_Sh'))
print('  ' + '-' * 70)

def backtest(train_df, test_df, feat_cols):
    feats = [f for f in feat_cols if f in train_df.columns]
    X_tr = train_df[feats].fillna(train_df[feats].median())
    y_tr = train_df['excess_ret'].fillna(0)
    X_te = test_df[feats].fillna(train_df[feats].median())

    m = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                       subsample=0.8, colsample_bytree=0.8,
                       n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    mask = ~np.isnan(pred) & ~np.isnan(test_df['excess_ret'].values)
    ic, _ = stats.spearmanr(pred[mask], test_df['excess_ret'].values[mask])

    te2 = test_df.copy()
    te2['pred'] = pred
    te2['ym'] = pd.to_datetime(te2['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in te2.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        mrets.append(top['excess_ret'].mean())

    if len(mrets) < 3: return {'ic': ic, 'sh': 0, 'mdd': 0}
    rets = np.array(mrets)
    ann = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets)>2 else 0.01
    sh = ann/vol if vol>0 else 0
    mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    return {'ic': ic, 'sh': sh, 'mdd': mdd}

windows = [
    ('2019->2020', '2016-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2020->2021', '2017-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2021->2022', '2018-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2022->2023', '2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2023->2024', '2020-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

all_res = []
for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date']>=tr_s) & (df['trade_date']<=tr_e)]
    te = df[(df['trade_date']>=te_s) & (df['trade_date']<=te_e)]
    if len(tr) < 5000 or len(te) < 1000: continue

    a = backtest(tr, te, feats_a)
    v = backtest(tr, te, feats_v)
    all_res.append((label, a, v))
    winner = 'Alpha' if a['ic'] > v['ic'] else 'vFinal'
    print('  %-14s | %+.4f %+.4f %6s | %8.3f %8.3f' % (
        label, a['ic'], v['ic'], winner, a['sh'], v['sh']))

if all_res:
    avg_a_ic = np.mean([r[1]['ic'] for r in all_res])
    avg_v_ic = np.mean([r[2]['ic'] for r in all_res])
    avg_a_sh = np.mean([r[1]['sh'] for r in all_res])
    avg_v_sh = np.mean([r[2]['sh'] for r in all_res])
    a_wins = sum(1 for r in all_res if r[1]['ic'] > r[2]['ic'])
    v_wins = sum(1 for r in all_res if r[2]['ic'] > r[1]['ic'])
    print('  %-14s | %+.4f %+.4f | %8.3f %8.3f | Wins: A=%d V=%d' % (
        'AVERAGE', avg_a_ic, avg_v_ic, avg_a_sh, avg_v_sh, a_wins, v_wins))

print('\nDone.')
