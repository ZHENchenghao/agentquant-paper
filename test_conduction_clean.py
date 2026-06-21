# -*- coding: utf-8 -*-
"""
Phase 2: Two-stage conduction test (clean rewrite)
Stage 1: macro -> predict which sectors will outperform
Stage 2: within selected sectors, use stock factors to pick best stocks
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

CACHE = 'cache/factors_all.parquet'

print('=' * 70)
print('Phase 2: Two-Stage Conduction Test')
print('=' * 70)

# --- Load ---
print('\n[0] Loading data...')
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

industry = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name,
               ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn = 1
""").df()

target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1) - (x.fc/x.close-1) AS excess_ret
    FROM (
        SELECT ts_code, trade_date, close,
               LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
        FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16'
    ) s
    JOIN (
        SELECT trade_date, close,
               LEAD(close,20) OVER(ORDER BY trade_date) AS fc
        FROM kline_daily WHERE ts_code='sh000300'
    ) x ON s.trade_date=x.trade_date
    WHERE s.fc IS NOT NULL
""").df()

macro = con.execute("""
    SELECT CAST(trade_date AS VARCHAR) AS trade_date, wti, copper, gold, vix, us10y
    FROM macro_indicators WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
con.close()

# Forward fill macro
for c in ['wti', 'copper', 'gold', 'vix', 'us10y']:
    macro[c] = macro[c].ffill().bfill()

MACRO_COLS = ['wti', 'copper', 'gold', 'vix', 'us10y']

# Merge
factors = pd.read_parquet(CACHE)
factors['trade_date'] = factors['trade_date'].astype(str)
target['trade_date'] = target['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(macro, on='trade_date', how='left')

print('  %d rows, %d stocks, %d industries, dates %s ~ %s' % (
    len(df), df.ts_code.nunique(), df.ind_name.nunique(),
    str(df.trade_date.min()), str(df.trade_date.max())))

# --- Industry daily returns (from kline, NOT factor cache) ---
con2 = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 每日行业等权收益 (直接用K线)
ind_daily_raw = con2.execute("""
    WITH stock_ret AS (
        SELECT k.ts_code, k.trade_date,
               k.close / LAG(k.close) OVER(PARTITION BY k.ts_code ORDER BY k.trade_date) - 1 AS ret_1d,
               x.close / LAG(x.close) OVER(ORDER BY k.trade_date) - 1 AS idx_ret
        FROM kline_daily k
        JOIN kline_daily x ON k.trade_date = x.trade_date AND x.ts_code = 'sh000300'
        WHERE k.trade_date >= '2016-01-01'
    ),
    stock_ind AS (
        SELECT sr.*, COALESCE(si.ind_name, 'Other') AS ind_name
        FROM stock_ret sr
        LEFT JOIN (
            SELECT ts_code, ind_name FROM (
                SELECT ts_code, ind_name,
                       ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
                FROM stock_industry_map
            ) WHERE rn = 1
        ) si ON sr.ts_code = si.ts_code
    )
    SELECT CAST(trade_date AS VARCHAR) AS trade_date, ind_name,
           AVG(ret_1d - idx_ret) AS excess_ret
    FROM stock_ind
    WHERE ret_1d IS NOT NULL
    GROUP BY trade_date, ind_name
    ORDER BY trade_date, ind_name
""").df()
con2.close()

# Merge macro
ind_daily = ind_daily_raw.merge(macro, on='trade_date', how='inner')

# Forward 10d return
ind_daily['fwd_10d'] = ind_daily.groupby('ind_name')['excess_ret'].transform(
    lambda x: x.rolling(10, min_periods=5).sum().shift(-10))

print('  Industry daily: %d rows' % len(ind_daily))

# --- Stage 1: Macro sensitivity per industry ---
print('\n[1] Stage 1: Macro -> Industry sensitivity...')

top_inds = df.groupby('ind_name').size().nlargest(12).index.tolist()
print('  Top12: %s' % ', '.join(top_inds[:6]) + '...')

ind_models = {}
ind_ic = {}

for ind in top_inds:
    sub = ind_daily[ind_daily['ind_name'] == ind].copy()
    valid = sub[MACRO_COLS + ['fwd_10d']].dropna()
    if len(valid) < 100:
        print('  %s: only %d valid rows, skip' % (ind, len(valid)))
        continue

    # Per-macro IC
    best_ic = 0
    best_m = ''
    for mc in MACRO_COLS:
        ic, _ = stats.spearmanr(valid[mc], valid['fwd_10d'])
        if abs(ic) > abs(best_ic):
            best_ic = ic
            best_m = mc
    ind_ic[ind] = {'macro': best_m, 'ic': best_ic}

    # Train model
    X = valid[MACRO_COLS].values
    y = valid['fwd_10d'].values
    m = LGBMRegressor(n_estimators=50, max_depth=3, num_leaves=7, verbose=-1, random_state=42)
    m.fit(X, y)
    ind_models[ind] = m

print('  Models trained: %d/%d' % (len(ind_models), len(top_inds)))
print('  Sector sensitivity:')
for ind, info in sorted(ind_ic.items(), key=lambda x: -abs(x[1]['ic']))[:6]:
    direction = 'pos' if info['ic'] > 0 else 'neg'
    print('    %-12s -> %-8s IC=%+.4f (%s)' % (ind, info['macro'], info['ic'], direction))

# --- Stock features ---
exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret', 'ind_name'] + MACRO_COLS
stock_feats = [c for c in df.columns if c not in exclude
               and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
print('\n  Stock features: %d' % len(stock_feats))

# --- Evaluation ---
print('\n[2] Two-stage vs Baseline...')

def run_test(train_start, train_end, test_start, test_end, label):
    tr = df[(df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)]
    te = df[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)]
    if len(tr) < 5000 or len(te) < 1000:
        return None

    feats = [f for f in stock_feats if f in tr.columns]

    # === Baseline: single-stage ===
    X_tr = tr[feats].fillna(tr[feats].median())
    y_tr = tr['excess_ret'].fillna(0)
    X_te = te[feats].fillna(tr[feats].median())

    m_bl = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                          subsample=0.8, colsample_bytree=0.8,
                          n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m_bl.fit(X_tr, y_tr)
    bl_pred = m_bl.predict(X_te)

    # Monthly portfolio
    te_bl = te.copy()
    te_bl['pred'] = bl_pred
    te_bl['ym'] = pd.to_datetime(te_bl['trade_date']).dt.to_period('M')
    bl_rets = []
    for mo, g in te_bl.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        bl_rets.append(top['excess_ret'].mean())

    # === Two-stage ===
    # Refit industry models on train period
    train_ind = ind_daily[(ind_daily['trade_date'] >= train_start) & (ind_daily['trade_date'] <= train_end)]
    stage1_models = {}
    for ind in top_inds:
        sub = train_ind[train_ind['ind_name'] == ind]
        valid = sub[MACRO_COLS + ['fwd_10d']].dropna()
        if len(valid) < 50: continue
        m = LGBMRegressor(n_estimators=50, max_depth=3, num_leaves=7, verbose=-1, random_state=42)
        m.fit(valid[MACRO_COLS].values, valid['fwd_10d'].values)
        stage1_models[ind] = m

    te_2s = te.copy()
    te_2s['pred'] = 0.0
    te_2s['ym'] = pd.to_datetime(te_2s['trade_date']).dt.to_period('M')

    for mo, g in te_2s.groupby('ym'):
        if len(g) < 500: continue
        mo_macro = g[MACRO_COLS].mean().values.reshape(1, -1)

        # Predict sector returns
        scores = {}
        for ind, m in stage1_models.items():
            try:
                scores[ind] = m.predict(mo_macro)[0]
            except:
                pass
        if not scores: continue

        # Top 5 sectors
        top5 = sorted(scores, key=scores.get, reverse=True)[:5]
        g_sel = g[g['ind_name'].isin(top5)]
        if len(g_sel) < 30: g_sel = g

        # Train stock model on selected sectors
        tr_sel = tr[tr['ind_name'].isin(top5)]
        if len(tr_sel) < 1000: tr_sel = tr
        X_tr_s = tr_sel[feats].fillna(tr_sel[feats].median())
        y_tr_s = tr_sel['excess_ret'].fillna(0)
        X_te_s = g_sel[feats].fillna(tr_sel[feats].median())

        m_s = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                             subsample=0.8, colsample_bytree=0.8,
                             n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
        m_s.fit(X_tr_s, y_tr_s)
        preds = m_s.predict(X_te_s)
        for j, idx in enumerate(g_sel.index):
            if j < len(preds):
                te_2s.loc[idx, 'pred'] = preds[j]

    # Monthly returns
    ts_rets = []
    for mo, g in te_2s.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        ts_rets.append(top['excess_ret'].mean())

    def metric(rets):
        if len(rets) < 3: return (0, 0, 0, 0)
        a = np.array(rets)
        ann = np.mean(a) * 12
        vol = np.std(a, ddof=1) * np.sqrt(12) if len(a) > 2 else 0.01
        sh = ann / vol if vol > 0 else 0
        mdd = np.min(np.cumprod(1+a) / np.maximum.accumulate(np.cumprod(1+a)) - 1)
        return (np.mean(a), sh, mdd, len(a))

    # IC
    mask_bl = ~np.isnan(te_bl['pred']) & ~np.isnan(te_bl['excess_ret'])
    ic_bl, _ = stats.spearmanr(te_bl.loc[mask_bl, 'pred'], te_bl.loc[mask_bl, 'excess_ret'])
    mask_2s = ~np.isnan(te_2s['pred']) & ~np.isnan(te_2s['excess_ret'])
    ic_2s, _ = stats.spearmanr(te_2s.loc[mask_2s, 'pred'], te_2s.loc[mask_2s, 'excess_ret'])

    bl_mr, bl_sh, bl_mdd, bl_n = metric(bl_rets)
    ts_mr, ts_sh, ts_mdd, ts_n = metric(ts_rets)

    return (label, ic_bl, ic_2s, bl_sh, ts_sh, bl_mdd, ts_mdd)

windows = [
    ('22-23->24', '2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('21-22->23', '2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('23-24->25', '2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
]

print('  %-14s %8s %8s %8s | %8s %8s | %8s %8s' % (
    'Window', 'BL_IC', '2S_IC', 'dIC', 'BL_Sh', '2S_Sh', 'BL_MDD', '2S_MDD'))
print('  ' + '-' * 80)

all_res = []
for label, tr_s, tr_e, te_s, te_e in windows:
    r = run_test(tr_s, tr_e, te_s, te_e, label)
    if r is None: continue
    label, ic_bl, ic_2s, bl_sh, ts_sh, bl_mdd, ts_mdd = r
    all_res.append(r)
    winner = '2S' if ic_2s > ic_bl else 'BL'
    print('  %-14s %+.4f %+.4f %+.4f | %8.3f %8.3f | %+7.1f%% %+7.1f%%  %s' % (
        label, ic_bl, ic_2s, ic_2s-ic_bl, bl_sh, ts_sh, bl_mdd*100, ts_mdd*100, winner))

if all_res:
    avg_bl_ic = np.mean([r[1] for r in all_res])
    avg_2s_ic = np.mean([r[2] for r in all_res])
    avg_bl_sh = np.mean([r[3] for r in all_res])
    avg_2s_sh = np.mean([r[4] for r in all_res])
    wins = sum(1 for r in all_res if r[2] > r[1])
    print('  %-14s %+.4f %+.4f %+.4f | %8.3f %8.3f | Wins: %d/%d' % (
        'AVERAGE', avg_bl_ic, avg_2s_ic, avg_2s_ic-avg_bl_ic,
        avg_bl_sh, avg_2s_sh, wins, len(all_res)))

print('\nDone.')
