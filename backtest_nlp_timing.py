# -*- coding: utf-8 -*-
"""
NLP Sentiment as Position-Sizing Overlay
Extreme sentiment -> reduce exposure. Normal -> full position.
Compare: Baseline(24F) vs NLP-Factor(25F) vs NLP-Timing(24F+position overlay)
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('NLP Timing Overlay: Sentiment-driven Position Sizing')
print('=' * 80)

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
con.close()

# NLP sentiment
news = pd.read_parquet('D:/AgentQuant/Astock-main/astock_mapped.parquet')
news['trade_date_clean'] = pd.to_datetime(news['trade_date'], errors='coerce')
news['sentiment'] = news['label'].map({0:0, 1:1, 2:-1})
news = news.sort_values(['ts_code', 'trade_date_clean'])
news['sent_20d'] = news.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(20, min_periods=3).mean().shift(1))
daily_sent = news.groupby(['ts_code', 'trade_date_clean']).agg(
    sent_20d=('sent_20d', 'last')).reset_index()
daily_sent['trade_date'] = daily_sent['trade_date_clean'].dt.strftime('%Y-%m-%d')

# Also build MARKET-LEVEL daily sentiment (for timing)
mkt_sent = news.groupby('trade_date_clean')['sentiment'].agg(['mean','count']).reset_index()
mkt_sent.columns = ['trade_date', 'mkt_sentiment', 'n_articles']
mkt_sent['trade_date'] = mkt_sent['trade_date'].dt.strftime('%Y-%m-%d')
# Rolling 5-day market sentiment
mkt_sent_ts = mkt_sent.set_index('trade_date')['mkt_sentiment'].sort_index()
mkt_sent_ts.index = pd.to_datetime(mkt_sent_ts.index)
mkt_sent_rolling = mkt_sent_ts.rolling(5).mean().shift(1)
mkt_sent_rolling.name = 'mkt_sent_5d'
mkt_sent_df = mkt_sent_rolling.reset_index()
mkt_sent_df['trade_date'] = mkt_sent_df['trade_date'].dt.strftime('%Y-%m-%d')

# Merge
factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')
for d in [target, mcap, daily_sent, mkt_sent_df]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(daily_sent[['ts_code','trade_date','sent_20d']], on=['ts_code','trade_date'], how='left')
df['sent_20d'] = df['sent_20d'].fillna(0.0)
df = df.merge(mkt_sent_df[['trade_date','mkt_sent_5d']], on='trade_date', how='left')
df['mkt_sent_5d'] = df['mkt_sent_5d'].fillna(0.0)

exclude = ['ts_code','trade_date','close','factor_group','_k','report_date','excess_ret','ind_name','mcap','mkt_sent_5d']
feats = [c for c in df.columns if c not in exclude and c!='sent_20d' and df[c].dtype in ('float64','float32','int64','int32')]
all_feats = feats + ['sent_20d']
print('  Factors: %d base + NLP = %d' % (len(feats), len(all_feats)))

# ============================================================
def neutralize_and_train(tr, te, feat_list):
    """Neutralize, train LightGBM, predict. Returns te with 'pred' column."""
    tr, te = tr.copy(), te.copy()
    for d in [tr, te]:
        d['ln_mcap'] = np.log(d['mcap'].fillna(1e6).clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap']**2

    ind_tr = pd.get_dummies(tr['ind_name'], prefix='ind').fillna(0).astype(float)
    X_tr = np.nan_to_num(pd.concat([tr['ln_mcap'], tr['ln_mcap_sq'], ind_tr], axis=1).fillna(0).values, 0)
    y_tr = np.nan_to_num(tr[feat_list].fillna(tr[feat_list].median()).fillna(0).values, 0)
    if X_tr.shape[0] > 50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        X_f, y_f = X_tr[idx], y_tr[idx]
    else:
        X_f, y_f = X_tr, y_tr

    m = LinearRegression(fit_intercept=False)
    m.fit(X_f, y_f)
    resid = y_tr - X_tr @ m.coef_.T
    neu_names = []
    for i, col in enumerate(feat_list):
        name = col+'_n'
        tr[name] = resid[:,i]
        tr[name] = (tr[name]-tr[name].mean())/tr[name].std()
        neu_names.append(name)

    # Apply to test
    ind_te = pd.get_dummies(te['ind_name'], prefix='ind').fillna(0).astype(float)
    X_te = np.nan_to_num(pd.concat([te['ln_mcap'], te['ln_mcap_sq'], ind_te], axis=1).fillna(0).values, 0)
    te_y = np.nan_to_num(te[feat_list].fillna(te[feat_list].median()).fillna(0).values, 0)
    te_resid = te_y - X_te @ m.coef_.T
    for i, col in enumerate(feat_list):
        name = col+'_n'
        te[name] = te_resid[:,i]
        te[name] = (te[name]-tr[name].mean())/tr[name].std()

    flist = [f for f in neu_names if f in tr.columns]
    model = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                           subsample=0.8, colsample_bytree=0.8,
                           n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    model.fit(tr[flist].fillna(tr[flist].median()), tr['excess_ret'].fillna(0))
    te['pred'] = model.predict(te[flist].fillna(tr[flist].median()))
    return te

# ============================================================
# Position sizing function
def position_size(mkt_sent_val):
    """NLP Timing: adjust position count based on market sentiment"""
    if pd.isna(mkt_sent_val) or mkt_sent_val == 0:
        return 30, 1.0  # no data -> full position
    abs_sent = abs(mkt_sent_val)
    if abs_sent > 0.5:   # extreme sentiment -> reduce to 15 stocks (50%)
        return 15, 0.5
    elif abs_sent > 0.3:  # elevated -> reduce to 22 stocks (73%)
        return 22, 0.73
    else:                 # normal -> 30 stocks
        return 30, 1.0

# ============================================================
# Rolling backtest
results = []

for test_yr in range(2008, 2025):
    tr_s = '%d-01-01'%(test_yr-3); tr_e = '%d-12-31'%(test_yr-1)
    te_s = '%d-01-01'%test_yr; te_e = '%d-12-31'%test_yr
    tr = df[(df['trade_date']>=tr_s)&(df['trade_date']<=tr_e)].dropna(subset=['excess_ret'])
    te = df[(df['trade_date']>=te_s)&(df['trade_date']<=te_e)].dropna(subset=['excess_ret'])
    if len(tr)<5000: continue
    # OOM guard: sample large training sets
    if len(tr)>500000: tr = tr.sample(500000, random_state=42)
    if len(te)>300000: te = te.sample(300000, random_state=42)

    # Three strategies:
    # 1. Baseline: 24F, always 30 stocks
    te_bl = neutralize_and_train(tr, te, feats)
    # 2. NLP-Factor: 25F (sent_20d as feature), always 30 stocks
    te_nl = neutralize_and_train(tr, te, all_feats)

    for te_df, label in [(te_bl,'BL'),(te_nl,'NLP-Factor')]:
        te_df['ym'] = pd.to_datetime(te_df['trade_date']).dt.to_period('M')
        for mo, g in te_df.groupby('ym'):
            if len(g)<30: continue
            top = g.nlargest(30, 'pred')
            results.append({
                'year': test_yr, 'month': str(mo), 'model': label,
                'ret': top['excess_ret'].mean(),
            })

    # 3. NLP-Timing: 24F + position sizing
    te_tm = neutralize_and_train(tr, te, feats)
    te_tm['ym'] = pd.to_datetime(te_tm['trade_date']).dt.to_period('M')
    for mo, g in te_tm.groupby('ym'):
        if len(g)<30: continue
        n_stocks, sizing = position_size(g['mkt_sent_5d'].mean())
        top = g.nlargest(n_stocks, 'pred')
        results.append({
            'year': test_yr, 'month': str(mo), 'model': 'NLP-Timing',
            'ret': top['excess_ret'].mean(),
        })

    # 4. NLP-Both: 25F (sent_20d as factor) + position sizing
    te_both = neutralize_and_train(tr, te, all_feats)
    te_both['ym'] = pd.to_datetime(te_both['trade_date']).dt.to_period('M')
    for mo, g in te_both.groupby('ym'):
        if len(g)<30: continue
        n_stocks, sizing = position_size(g['mkt_sent_5d'].mean())
        top = g.nlargest(n_stocks, 'pred')
        results.append({
            'year': test_yr, 'month': str(mo), 'model': 'NLP-Both',
            'ret': top['excess_ret'].mean(),
        })

    ic_bl = np.corrcoef(te_bl['pred'].dropna(), te_bl.loc[te_bl['pred'].notna(),'excess_ret'])[0,1]
    ic_nl = np.corrcoef(te_nl['pred'].dropna(), te_nl.loc[te_nl['pred'].notna(),'excess_ret'])[0,1]
    print('  %d: BL=%.4f NL=%.4f' % (test_yr, ic_bl, ic_nl))

# ============================================================
# Summary
recs = pd.DataFrame(results)

print('\n%-15s %10s %10s %10s %10s %10s %10s' % ('Model','AnnRet','Vol','Sharpe','MDD','WinRate','Cum'))
print('-' * 80)
for label, name in [('BL','Baseline'),('NLP-Factor','NLP-Factor'),('NLP-Timing','NLP-Timing'),('NLP-Both','NLP-Both')]:
    rets = recs[recs['model']==label].set_index('month')['ret'].sort_index()
    if len(rets)==0: continue
    rv = rets.values
    ann=np.mean(rv)*12; vol=np.std(rv,ddof=1)*np.sqrt(12)
    sh=ann/vol if vol>0 else 0
    cum=np.cumprod(1+rv); mdd=np.min(cum/np.maximum.accumulate(cum)-1)
    wr=np.mean(rv>0)
    print('%-15s %+9.1f%% %9.1f%% %10.3f %+9.1f%% %9.1f%% %+9.1f%%' % (
        name, ann*100, vol*100, sh, mdd*100, wr*100, (cum[-1]-1)*100))

print('\nDone.')
