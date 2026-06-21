# -*- coding: utf-8 -*-
"""
vFinal+ Full Rolling Backtest 2019-2024
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print('=' * 90)
print('vFinal+ Rolling Backtest 2019-2024')
print('=' * 90)

# Load
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
mcap = con.execute("""SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
    close * total_share / 10000.0 AS mcap FROM kline_daily WHERE trade_date >= '2002-01-01'""").df()
board = con.execute("""SELECT DISTINCT ts_code,
    CASE WHEN ts_code LIKE 'sz300%' OR ts_code LIKE 'sz301%' THEN 'ChiNext'
         WHEN ts_code LIKE 'sh688%' THEN 'STAR' ELSE 'MainBoard' END AS board
    FROM kline_daily WHERE trade_date >= '2019-01-01'""").df()
con.close()

# NLP
news = pd.read_parquet('D:/AgentQuant/Astock-main/astock_mapped.parquet')
news['trade_date_clean'] = pd.to_datetime(news['trade_date'], errors='coerce')
news['sentiment'] = news['label'].map({0:0, 1:1, 2:-1})
news = news.sort_values(['ts_code', 'trade_date_clean'])
news['sent_20d'] = news.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(20, min_periods=3).mean().shift(1))
daily_sent = news.groupby(['ts_code', 'trade_date_clean']).agg(
    sent_20d=('sent_20d', 'last')).reset_index()
daily_sent['trade_date'] = daily_sent['trade_date_clean'].dt.strftime('%Y-%m-%d')

# Merge
factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')
print('  Factors: %d rows, %d stocks, %s ~ %s' % (len(factors), factors['ts_code'].nunique(), factors['trade_date'].min(), factors['trade_date'].max()))
for d in [target, mcap, daily_sent]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(board, on='ts_code', how='left')
df['board'] = df['board'].fillna('MainBoard')
df = df.merge(daily_sent[['ts_code','trade_date','sent_20d']], on=['ts_code','trade_date'], how='left')
df['sent_20d'] = df['sent_20d'].fillna(0.0)

# Factor list
exclude = ['ts_code','trade_date','close','factor_group','_k','report_date','excess_ret','ind_name','mcap','board']
raw_feats = [c for c in df.columns if c not in exclude and c != 'sent_20d'
             and df[c].dtype in ('float64','float32','int64','int32')]
ALL_RAW = [f for f in raw_feats + ['sent_20d'] if f in df.columns]
print(' Factors: %d (including sent_20d)' % len(ALL_RAW))

# ============================================================
# Neutralize (sampled fit to avoid OOM)
# ============================================================
def neutralize(data, factor_list):
    data = data.copy()
    data['ln_mcap'] = np.log(data['mcap'].fillna(1e6).clip(lower=1e6))
    data['ln_mcap_sq'] = data['ln_mcap'] ** 2
    ind_dum = pd.get_dummies(data['ind_name'], prefix='ind').fillna(0).astype(float)
    X_full = pd.concat([data['ln_mcap'], data['ln_mcap_sq'], ind_dum], axis=1).fillna(0).values.astype(float)

    feats = [f for f in factor_list if f in data.columns]
    y_df = data[feats].copy()
    for c in feats: y_df[c] = y_df[c].fillna(y_df[c].median()).fillna(0)
    y = np.nan_to_num(y_df.values.astype(float), 0)

    # Sample fit to avoid OOM
    if X_full.shape[0] > 50000:
        idx = np.random.choice(X_full.shape[0], 50000, replace=False)
        X_fit, y_fit = X_full[idx], y[idx]
    else:
        X_fit, y_fit = X_full, y

    m = LinearRegression(fit_intercept=False)
    m.fit(X_fit, y_fit)
    resid = y - X_full @ m.coef_.T

    neu_names = []
    for i, col in enumerate(feats):
        name = col + '_n'
        data[name] = resid[:, i]
        mn, st = data[name].mean(), data[name].std()
        if st > 0: data[name] = (data[name] - mn) / st
        neu_names.append(name)
    return data, neu_names

# ============================================================
# Roll
# ============================================================
print('\n%-20s %8s %8s %8s | %8s %8s' % ('Window', 'BL_IC', 'NL_IC', 'dIC', 'BL_Sh', 'NL_Sh'))
print('-' * 70)

records = []

for test_yr in range(2008, 2025):
    tr_s = '%d-01-01' % (test_yr-3)
    tr_e = '%d-12-31' % (test_yr-1)
    te_s = '%d-01-01' % test_yr
    te_e = '%d-12-31' % test_yr

    tr = df[(df['trade_date']>=tr_s) & (df['trade_date']<=tr_e)].copy()
    te = df[(df['trade_date']>=te_s) & (df['trade_date']<=te_e)].copy()

    tr = tr.dropna(subset=['excess_ret'])
    te = te.dropna(subset=['excess_ret'])
    if len(tr) < 10000 or len(te) < 2000: continue

    # Neutralize
    tr, neu_feats = neutralize(tr, ALL_RAW)

    # Apply to test
    te['ln_mcap'] = np.log(te['mcap'].fillna(1e6).clip(lower=1e6))
    te['ln_mcap_sq'] = te['ln_mcap'] ** 2
    ind_te = pd.get_dummies(te['ind_name'], prefix='ind').fillna(0).astype(float)
    X_te = np.nan_to_num(pd.concat([te['ln_mcap'].fillna(0), te['ln_mcap_sq'].fillna(0), ind_te], axis=1).fillna(0).values.astype(float), 0)

    for raw_col in ALL_RAW:
        if raw_col not in te.columns or raw_col not in tr.columns: continue
        neu_name = raw_col + '_n'
        if neu_name not in tr.columns: continue

        tr_y = np.nan_to_num(tr[raw_col].fillna(tr[raw_col].median()).fillna(0).values.astype(float), 0)
        te_y = np.nan_to_num(te[raw_col].fillna(te[raw_col].median()).fillna(0).values.astype(float), 0)

        ind_tr = pd.get_dummies(tr['ind_name'], prefix='ind').fillna(0).astype(float)
        X_tr = np.nan_to_num(pd.concat([tr['ln_mcap'].fillna(0), tr['ln_mcap_sq'].fillna(0), ind_tr], axis=1).fillna(0).values.astype(float), 0)

        if X_tr.shape[0] > 50000:
            idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
            X_f, y_f = X_tr[idx], tr_y[idx]
        else:
            X_f, y_f = X_tr, tr_y

        m = LinearRegression(fit_intercept=False)
        m.fit(X_f, y_f)
        te[neu_name] = te_y - X_te @ m.coef_.T
        if tr[neu_name].std() > 0:
            te[neu_name] = (te[neu_name] - tr[neu_name].mean()) / tr[neu_name].std()

    # Sample large datasets BEFORE filtering (avoid OOM on copy)
    if len(tr) > 500000:
        tr = tr.sample(500000, random_state=42)
    if len(te) > 300000:
        te = te.sample(300000, random_state=42)

    # Micro-cap filter
    for d in [tr, te]:
        d['mcap'] = d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['mcap_pct'] = d.groupby('trade_date')['mcap'].rank(pct=True)

    tr_f = tr[tr['mcap_pct'] >= 0.20]
    te_f = te[te['mcap_pct'] >= 0.20]

    base_neu = [f for f in neu_feats if not f.startswith('sent_')]
    all_neu = neu_feats

    # Train
    for feats, label in [(base_neu, 'bl'), (all_neu, 'nl')]:
        flist = [f for f in feats if f in tr_f.columns]
        X_tr = tr_f[flist].fillna(tr_f[flist].median())
        y_tr = tr_f['excess_ret'].fillna(0)
        X_te = te_f[flist].fillna(tr_f[flist].median())

        m = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                           subsample=0.8, colsample_bytree=0.8,
                           n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        te_f['pred_'+label] = m.predict(X_te)
        ic = np.corrcoef(te_f['pred_'+label].dropna(), te_f.loc[te_f['pred_'+label].notna(), 'excess_ret'])[0,1]

        te_f['ym'] = pd.to_datetime(te_f['trade_date']).dt.to_period('M')
        for mo, g in te_f.groupby('ym'):
            if len(g) < 30: continue
            top30 = g.nlargest(30, 'pred_'+label)
            cn = top30['board'].isin(['ChiNext']).sum()
            if cn > 12:
                drop = top30[top30['board']=='ChiNext'].nsmallest(cn-12, 'pred_'+label).index
                top30 = top30.drop(drop)
                fill = g[~g.index.isin(top30.index)].nlargest(cn-12, 'pred_'+label)
                top30 = pd.concat([top30, fill])
            records.append({
                'year': test_yr, 'month': str(mo), 'model': label,
                'ret': top30['excess_ret'].mean(), 'ic': ic,
            })

    bl_ic = np.mean([r['ic'] for r in records if r['year']==test_yr and r['model']=='bl'])
    nl_ic = np.mean([r['ic'] for r in records if r['year']==test_yr and r['model']=='nl'])
    print('%-20s %+.4f %+.4f %+.4f' % ('%s-%s->%d'%(tr_s[:4],tr_e[:4],test_yr), bl_ic, nl_ic, nl_ic-bl_ic))

# ============================================================
# Summary
# ============================================================
recs = pd.DataFrame(records)
print('\n' + '=' * 90)
print('TEAR SHEET')
print('=' * 90)

for label, name in [('bl', 'Baseline'), ('nl', '+NLP')]:
    rets = recs[recs['model']==label].set_index('month')['ret'].sort_index()
    rv = rets.values
    ann = np.mean(rv)*12
    vol = np.std(rv, ddof=1)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+rv)
    mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    wr = np.mean(rv>0)
    print('%s: AnnRet=%+.1f%% Vol=%.1f%% Sharpe=%.3f MDD=%+.1f%% WinRate=%.1f%% Cum=%+.1f%%' % (
        name, ann*100, vol*100, sh, mdd*100, wr*100, (cum[-1]-1)*100))

# Year by year
print('\n%-6s | %10s %10s | %8s %8s | %s' % ('Year', 'BL_Ret', 'NL_Ret', 'BL_Sh', 'NL_Sh', 'Winner'))
for yr in sorted(recs['year'].unique()):
    bl = recs[(recs['model']=='bl')&(recs['year']==yr)]['ret']
    nl = recs[(recs['model']=='nl')&(recs['year']==yr)]['ret']
    if len(bl)<3: continue
    bl_ann = bl.mean()*12; nl_ann = nl.mean()*12
    bl_vol = bl.std()*np.sqrt(12); nl_vol = nl.std()*np.sqrt(12)
    bl_sh = bl_ann/bl_vol if bl_vol>0 else 0; nl_sh = nl_ann/nl_vol if nl_vol>0 else 0
    w = 'NLP' if nl_sh>bl_sh else 'BASE'
    print('%-6d | %+9.1f%% %+9.1f%% | %8.3f %8.3f | %s' % (yr, bl_ann*100, nl_ann*100, bl_sh, nl_sh, w))

recs.to_parquet('cache/rolling_final_2019_2024.parquet')
print('\nDone.')
