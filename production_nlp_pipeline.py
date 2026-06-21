# -*- coding: utf-8 -*-
"""
vFinal+ Production Pipeline
24 base factors + NLP_sent_20d
-> Industry + Quadratic Market Cap Neutralization
-> Standardization
-> LightGBM (depth=10, leaves=63)
-> Production cutoffs: micro-cap exclusion + ChiNext 40% cap
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from lightgbm import LGBMRegressor
import warnings, json, os
from datetime import datetime
warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE = 'cache/factors_all.parquet'
ASTOCK = 'D:/AgentQuant/Astock-main/astock_mapped.parquet'
OUTPUT_DIR = 'D:/AgentQuant/our/production'

os.makedirs(OUTPUT_DIR, exist_ok=True)

print('=' * 80)
print('vFinal+ Production Pipeline: 24F + NLP + Quadratic OLS')
print('=' * 80)

# ============================================================
# 0. Load all data
# ============================================================
print('\n[0] Loading data...')
con = duckdb.connect(DB, read_only=True)

# Industry mapping
industry = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map
    ) WHERE rn = 1
""").df()

# Target: 20-day excess return
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close - 1.0) - (x.fc/x.close - 1.0) AS excess_ret
    FROM (
        SELECT ts_code, trade_date, close,
               LEAD(close, 20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
        FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16'
    ) s
    JOIN (
        SELECT trade_date, close,
               LEAD(close, 20) OVER(ORDER BY trade_date) AS fc
        FROM kline_daily WHERE ts_code = 'sh000300'
    ) x ON s.trade_date = x.trade_date
    WHERE s.fc IS NOT NULL
""").df()

# Market cap data for neutralization
mcap = con.execute("""
    SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
           close * total_share / 10000.0 AS mcap
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df()

# Board classification (for ChiNext cap)
board = con.execute("""
    SELECT DISTINCT ts_code,
           CASE WHEN ts_code LIKE 'sz300%' OR ts_code LIKE 'sz301%' THEN 'ChiNext'
                WHEN ts_code LIKE 'sh688%' THEN 'STAR'
                ELSE 'MainBoard' END AS board
    FROM kline_daily WHERE trade_date >= '2024-01-01'
""").df()

con.close()

# ============================================================
# 1. Build NLP sentiment factor from Astock
# ============================================================
print('\n[1] Building NLP sent_20d from Astock...')
news = pd.read_parquet(ASTOCK)
news['trade_date_clean'] = pd.to_datetime(news['trade_date'], errors='coerce')
news['sentiment'] = news['label'].map({0: 0, 1: 1, 2: -1})
news = news.sort_values(['ts_code', 'trade_date_clean'])

# 20-day rolling sentiment per stock
news['sent_20d'] = news.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(20, min_periods=3).mean().shift(1))

# Aggregate to daily per-stock
daily_sent = news.groupby(['ts_code', 'trade_date_clean']).agg(
    sent_20d=('sent_20d', 'last'),
    sent_5d=('sentiment', lambda x: x.rolling(5, min_periods=1).mean().iloc[-1] if len(x) >= 1 else 0),
    n_articles=('sentiment', 'count'),
).reset_index()

daily_sent['trade_date_str'] = daily_sent['trade_date_clean'].dt.strftime('%Y-%m-%d')
print('  NLP coverage: %d stocks, %d days, %s ~ %s' % (
    daily_sent['ts_code'].nunique(),
    daily_sent['trade_date_clean'].nunique(),
    daily_sent['trade_date_clean'].min().date(),
    daily_sent['trade_date_clean'].max().date()))

# ============================================================
# 2. Merge all data
# ============================================================
print('\n[2] Merging: factors + target + NLP + mcap + board...')

factors = pd.read_parquet(CACHE)
for d in [factors, target, mcap]:
    d['trade_date'] = d['trade_date'].astype(str)

daily_sent['trade_date_str'] = daily_sent['trade_date_str'].astype(str)

df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code', 'trade_date'], how='left')
df = df.merge(board, on='ts_code', how='left')
df['board'] = df['board'].fillna('MainBoard')

# Merge NLP
df = df.merge(daily_sent[['ts_code', 'trade_date_str', 'sent_20d']],
              left_on=['ts_code', 'trade_date'],
              right_on=['ts_code', 'trade_date_str'], how='left')
df['sent_20d'] = df['sent_20d'].fillna(0.0)  # no news = neutral
df = df.drop(columns=['trade_date_str'])

print('  Merged: %d rows, %d stocks, %d industries' % (
    len(df), df.ts_code.nunique(), df.ind_name.nunique()))

# ============================================================
# 3. Quadratic Market Cap + Industry Neutralization (ALL 25 factors)
# ============================================================
print('\n[3] Quadratic OLS Neutralization on 25 factors...')

# Select factor columns
exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret', 'ind_name', 'mcap', 'board']
raw_feats = [c for c in df.columns if c not in exclude
             and c not in ('sent_20d',)
             and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]

all_feats = raw_feats + ['sent_20d']
all_feats = [f for f in all_feats if f in df.columns]

print('  Factors to neutralize: %d (24 base + NLP)' % len(all_feats))

# Prepare neutralization inputs
df['ln_mcap'] = np.log(df['mcap'].clip(lower=1e6))
df['ln_mcap_sq'] = df['ln_mcap'] ** 2
ind_dummies = pd.get_dummies(df['ind_name'], prefix='ind', dummy_na=False).fillna(0).astype(float)

X_neu = pd.concat([df['ln_mcap'], df['ln_mcap_sq'], ind_dummies], axis=1).fillna(0).values.astype(float)

# Neutralize
y_raw = df[all_feats].copy()
for c in all_feats:
    y_raw[c] = y_raw[c].fillna(y_raw[c].median())
y_vals = y_raw.values.astype(float)

model_neu = LinearRegression(fit_intercept=False)
model_neu.fit(X_neu, y_vals)
residuals = y_vals - X_neu @ model_neu.coef_.T

# Replace factors with neutralized residuals
neu_feats = []
for i, col in enumerate(all_feats):
    neu_name = col + '_neu'
    df[neu_name] = residuals[:, i]
    neu_feats.append(neu_name)

# Standardize (Z-score)
for col in neu_feats:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val > 0:
        df[col] = (df[col] - mean_val) / std_val

print('  Neutralized + Standardized: %d factors' % len(neu_feats))
print('  Residual mcap IC (sent_20d_neu): %.6f' % (
    stats.spearmanr(df['sent_20d_neu'].dropna(), df['ln_mcap'].loc[df['sent_20d_neu'].notna()])[0]))

# ============================================================
# 4. Production cutoffs
# ============================================================
print('\n[4] Production cutoffs...')

# Micro-cap exclusion: bottom 20% by market cap
df['mcap_pct'] = df.groupby('trade_date')['mcap'].rank(pct=True)
df['micro_cap'] = df['mcap_pct'] < 0.20

# ChiNext cap: max 40% of portfolio can be ChiNext
df['is_chinext'] = df['board'] == 'ChiNext'

n_micro = df['micro_cap'].sum()
n_total = len(df)
print('  Micro-cap excluded: %d/%d (%.1f%%)' % (n_micro, n_total, 100*n_micro/n_total))
print('  ChiNext capped: max 40%% of portfolio')

# ============================================================
# 5. LightGBM backtest (rolling 3-year train, 1-year test)
# ============================================================
print('\n[5] LightGBM backtest...')

def portfolio_backtest(train_df, test_df, feat_list):
    """Full pipeline: neutralized factors -> LightGBM -> cutoffs -> portfolio"""
    feats = [f for f in feat_list if f in train_df.columns]

    # Training data: exclude micro-caps (model learns from investable universe)
    tr = train_df[~train_df['micro_cap']].copy()
    X_tr = tr[feats].fillna(tr[feats].median())
    y_tr = tr['excess_ret'].fillna(0)

    # Test data
    te = test_df[~test_df['micro_cap']].copy()
    X_te = te[feats].fillna(tr[feats].median())

    # Train
    model = LGBMRegressor(
        learning_rate=0.05, num_leaves=63, max_depth=10,
        subsample=0.8, colsample_bytree=0.8,
        n_estimators=200, verbose=-1, random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    te['pred'] = model.predict(X_te)

    # Feature importance
    importance = dict(zip(feats, model.feature_importances_))

    # IC
    mask = ~np.isnan(te['pred']) & ~np.isnan(te['excess_ret'])
    ic, _ = stats.spearmanr(te.loc[mask, 'pred'], te.loc[mask, 'excess_ret'])

    # Monthly portfolio with ChiNext cap
    te['ym'] = pd.to_datetime(te['trade_date']).dt.to_period('M')
    monthly_rets = []
    monthly_composition = {}

    for mo, g in te.groupby('ym'):
        if len(g) < 30:
            continue

        # Top 30 by prediction
        top30 = g.nlargest(30, 'pred')

        # ChiNext cap: max 40% = 12 stocks
        chinext_count = top30['is_chinext'].sum()
        if chinext_count > 12:
            # Drop lowest-predicted ChiNext stocks
            chinext_stocks = top30[top30['is_chinext']].nsmallest(chinext_count - 12, 'pred')
            top30 = top30.drop(chinext_stocks.index)
            # Backfill with next best non-ChiNext
            remaining = g[~g.index.isin(top30.index)].nlargest(chinext_count - 12, 'pred')
            top30 = pd.concat([top30, remaining])

        monthly_rets.append(top30['excess_ret'].mean())
        monthly_composition[str(mo)] = {
            'n_stocks': len(top30),
            'chinext_count': int(top30['is_chinext'].sum()),
            'top3_pred': top30['pred'].head(3).tolist(),
        }

    if len(monthly_rets) < 3:
        return {'ic': ic, 'sh': 0, 'mdd': 0, 'mr': 0, 'importance': importance, 'months': len(monthly_rets)}

    rets = np.array(monthly_rets)
    ann_ret = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets) > 2 else 0.01
    sharpe = ann_ret / vol if vol > 0 else 0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    mdd = np.min(cum / peak - 1)
    win_rate = np.mean(rets > 0)

    return {
        'ic': ic, 'sharpe': sharpe, 'mdd': mdd,
        'ann_ret': ann_ret, 'win_rate': win_rate, 'mr': np.mean(rets),
        'importance': importance, 'months': len(rets),
        'composition': monthly_composition,
    }

# Baseline: 24 neutralized factors (no NLP)
base_neu = [f for f in neu_feats if not f.startswith('sent_')]
# Upgrade: 25 neutralized factors (with NLP)
all_neu = neu_feats

print('  Baseline: %d factors (24 base, neutralized)' % len(base_neu))
print('  Upgrade:  %d factors (24 base + NLP, neutralized)' % len(all_neu))

# Test windows
windows = [
    ('2020', '2017-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2021', '2018-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2022', '2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2023', '2020-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2024', '2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

print('\n  %-6s | %8s %8s %8s | %8s %8s %8s | %6s %6s | %s' % (
    'Year', 'BL_IC', 'NL_IC', 'dIC', 'BL_Sh', 'NL_Sh', 'dSh', 'BL_WR', 'NL_WR', 'Win?'))
print('  ' + '-' * 95)

all_baseline = []
all_upgrade = []
nlp_importance_all = []

for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date'] >= tr_s) & (df['trade_date'] <= tr_e)]
    te = df[(df['trade_date'] >= te_s) & (df['trade_date'] <= te_e)]
    if len(tr) < 5000: continue

    bl = portfolio_backtest(tr, te, base_neu)
    nl = portfolio_backtest(tr, te, all_neu)

    all_baseline.append(bl)
    all_upgrade.append(nl)

    if 'sent_20d_neu' in nl['importance']:
        nlp_importance_all.append(nl['importance']['sent_20d_neu'])

    dic = nl['ic'] - bl['ic']
    dsh = nl['sharpe'] - bl['sharpe']
    dwr = nl['win_rate'] - bl['win_rate']
    winner = 'NLP' if nl['ic'] > bl['ic'] else 'BASE'

    print('  %-6s | %+.4f %+.4f %+.4f | %8.3f %8.3f %+.3f | %5.1f%% %5.1f%% | %s' % (
        label, bl['ic'], nl['ic'], dic,
        bl['sharpe'], nl['sharpe'], dsh,
        bl['win_rate']*100, nl['win_rate']*100, winner))

# ============================================================
# 6. Summary
# ============================================================
print('\n' + '=' * 80)
print('FINAL SUMMARY')
print('=' * 80)

avg_bl_ic = np.mean([r['ic'] for r in all_baseline])
avg_nl_ic = np.mean([r['ic'] for r in all_upgrade])
avg_bl_sh = np.mean([r['sharpe'] for r in all_baseline])
avg_nl_sh = np.mean([r['sharpe'] for r in all_upgrade])
avg_bl_mdd = np.mean([r['mdd'] for r in all_baseline])
avg_nl_mdd = np.mean([r['mdd'] for r in all_upgrade])
avg_bl_wr = np.mean([r['win_rate'] for r in all_baseline])
avg_nl_wr = np.mean([r['win_rate'] for r in all_upgrade])
wins = sum(1 for i in range(len(all_baseline)) if all_upgrade[i]['ic'] > all_baseline[i]['ic'])

print('  %-12s %8s %8s %8s' % ('Metric', 'Baseline', '+NLP', 'Delta'))
print('  %-12s %8s %8s %8s' % ('-'*12, '-'*8, '-'*8, '-'*8))
print('  %-12s %+.4f %+.4f %+.4f' % ('IC', avg_bl_ic, avg_nl_ic, avg_nl_ic - avg_bl_ic))
print('  %-12s %8.3f %8.3f %+.3f' % ('Sharpe', avg_bl_sh, avg_nl_sh, avg_nl_sh - avg_bl_sh))
print('  %-12s %+7.1f%% %+7.1f%% %+7.1f%%' % ('MDD', avg_bl_mdd*100, avg_nl_mdd*100, (avg_nl_mdd - avg_bl_mdd)*100))
print('  %-12s %6.1f%% %6.1f%% %+6.1f%%' % ('WinRate', avg_bl_wr*100, avg_nl_wr*100, (avg_nl_wr - avg_bl_wr)*100))
print('  %-12s %8d %8d' % ('Wins', wins, len(all_baseline)))

if nlp_importance_all:
    print('\n  NLP sent_20d avg feature importance: %.4f (range %.4f-%.4f)' % (
        np.mean(nlp_importance_all), np.min(nlp_importance_all), np.max(nlp_importance_all)))

# ============================================================
# 7. Save production model
# ============================================================
print('\n[7] Saving production artifacts...')

# Train final model on all data through 2025, for 2026 production
train_final = df[(df['trade_date'] >= '2019-01-01') & (df['trade_date'] <= '2025-12-31')]
train_final = train_final[~train_final['micro_cap']]

X_final = train_final[all_neu].fillna(train_final[all_neu].median())
y_final = train_final['excess_ret'].fillna(0)

final_model = LGBMRegressor(
    learning_rate=0.05, num_leaves=63, max_depth=10,
    subsample=0.8, colsample_bytree=0.8,
    n_estimators=200, verbose=-1, random_state=42, n_jobs=-1
)
final_model.fit(X_final, y_final)

# Save
import joblib
joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'lgbm_model.pkl'))

# Save config
config = {
    'pipeline': 'vFinal+',
    'factors': all_neu,
    'n_factors': len(all_neu),
    'neutralization': 'quadratic_ols_industry',
    'model': 'LightGBM_depth10_leaves63',
    'cutoffs': {'micro_cap_pct': 0.20, 'chinext_max_pct': 0.40},
    'built': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'nlp_source': 'Astock_25493_labeled_articles',
    'nlp_factor': 'sent_20d',
}
with open(os.path.join(OUTPUT_DIR, 'pipeline_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print('  Model: %s/lgbm_model.pkl' % OUTPUT_DIR)
print('  Config: %s/pipeline_config.json' % OUTPUT_DIR)
print('  Factors: %d (24 base + NLP sent_20d, all neutralized)' % len(all_neu))

print('\nDone. vFinal+ production pipeline ready.')
