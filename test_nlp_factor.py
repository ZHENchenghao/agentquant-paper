# -*- coding: utf-8 -*-
"""
NLP因子回测: 用Astock标注新闻构建情感因子
每个股票: 过去N天新闻标签均值 → 预测未来20日超额收益
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('NLP Sentiment Factor from Astock Labeled News')
print('=' * 70)

# Load Astock mapped data
news = pd.read_parquet('D:/AgentQuant/Astock-main/astock_mapped.parquet')
news['trade_date_clean'] = pd.to_datetime(news['trade_date_clean'])
news['sentiment'] = news['label'].map({0: 0, 1: 1, 2: -1})  # hold=0, long=+1, short=-1

print('  News: %d articles, %d stocks, %s ~ %s' % (
    len(news), news['ts_code'].nunique(),
    news['trade_date_clean'].min().date(), news['trade_date_clean'].max().date()))

# Build stock-level sentiment features
# For each (ts_code, trade_date), compute rolling sentiment over past N days
print('\n[1] Building stock-level sentiment features...')

news_sorted = news.sort_values(['ts_code', 'trade_date_clean'])

# Rolling 5-day and 20-day sentiment per stock
news_sorted['sent_5d'] = news_sorted.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(5, min_periods=1).mean().shift(1))
news_sorted['sent_20d'] = news_sorted.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(20, min_periods=3).mean().shift(1))
news_sorted['sent_count_20d'] = news_sorted.groupby('ts_code')['sentiment'].transform(
    lambda x: x.rolling(20, min_periods=1).count().shift(1))
# Sentiment change (momentum of sentiment)
news_sorted['sent_chg'] = news_sorted['sent_5d'] - news_sorted['sent_20d']

# Aggregate to daily per-stock (take latest sentiment values for each day)
daily_sent = news_sorted.groupby(['ts_code', 'trade_date_clean']).agg(
    sent_5d=('sent_5d', 'last'),
    sent_20d=('sent_20d', 'last'),
    sent_chg=('sent_chg', 'last'),
    sent_count=('sent_count_20d', 'last'),
    n_articles_today=('sentiment', 'count'),
    sentiment_today=('sentiment', 'mean'),
).reset_index()

daily_sent['trade_date_str'] = daily_sent['trade_date_clean'].dt.strftime('%Y-%m-%d')
print('  Daily stock sentiment: %d rows' % len(daily_sent))
print('  Date range: %s ~ %s' % (
    daily_sent['trade_date_clean'].min().date(), daily_sent['trade_date_clean'].max().date()))

# ============================================================
# Merge with factor cache + target
# ============================================================
print('\n[2] Merging with factor cache...')

factors = pd.read_parquet('cache/factors_all.parquet')
factors['trade_date'] = factors['trade_date'].astype(str)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1) - (x.fc/x.close-1) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
con.close()
target['trade_date'] = target['trade_date'].astype(str)

# Merge factors + target
df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')

# Merge sentiment
daily_sent['trade_date_str'] = daily_sent['trade_date_str'].astype(str)
df_w_sent = df.merge(daily_sent, left_on=['ts_code', 'trade_date'],
                      right_on=['ts_code', 'trade_date_str'], how='left')

# Fill missing sentiment with 0 (no news = neutral)
for col in ['sent_5d', 'sent_20d', 'sent_chg', 'sentiment_today']:
    df_w_sent[col] = df_w_sent[col].fillna(0)
df_w_sent['sent_count'] = df_w_sent['sent_count'].fillna(0)
df_w_sent['n_articles_today'] = df_w_sent['n_articles_today'].fillna(0)

print('  Merged: %d rows, sentiment coverage: %.1f%%' % (
    len(df_w_sent),
    100 * (df_w_sent['sent_count'] > 0).sum() / len(df_w_sent)))

# ============================================================
# Test NLP factors vs excess returns
# ============================================================
print('\n[3] NLP Factor IC Analysis...')

# Only use dates where we have sufficient news coverage
df_test = df_w_sent[df_w_sent['trade_date'] >= '2019-01-01']  # After first 6 months of news

# Per-stock sentiment IC
sent_cols = ['sent_5d', 'sent_20d', 'sent_chg', 'sentiment_today']
for col in sent_cols:
    valid = df_test[[col, 'excess_ret']].dropna()
    if len(valid) > 1000:
        ic, p = stats.spearmanr(valid[col], valid['excess_ret'])
        print('  %-20s IC=%+.4f  p=%.4f  n=%d' % (col, ic, p, len(valid)))

# ============================================================
# LightGBM backtest: baseline vs +NLP
# ============================================================
print('\n[4] Backtest: 24 factors vs 24+NLP factors...')

exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret', 'trade_date_clean', 'trade_date_str', 'sent_count']
base_feats = [c for c in df_w_sent.columns if c not in exclude
              and df_w_sent[c].dtype in ('float64', 'float32', 'int64', 'int32')
              and c not in sent_cols + ['n_articles_today']]
nlp_feats = base_feats + sent_cols + ['n_articles_today']
nlp_feats = [f for f in nlp_feats if f in df_w_sent.columns]

print('  Baseline features: %d, +NLP: %d' % (len(base_feats), len(nlp_feats)))

def backtest(train_df, test_df, feat_list):
    feats = [f for f in feat_list if f in train_df.columns]
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
    ann = np.mean(rets)*12
    vol = np.std(rets, ddof=1)*np.sqrt(12) if len(rets)>2 else 0.01
    sh = ann/vol if vol>0 else 0
    mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    return {'ic': ic, 'sh': sh, 'mdd': mdd}

# Test on 2020 (trained on 2019 news + factors)
windows = [
    ('2019->2020', '2019-01-01', '2019-12-31', '2020-01-01', '2020-12-31'),
    ('2020->2021', '2020-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
]

print('  %-14s | %8s %8s %8s | %8s %8s' % (
    'Window', 'BL_IC', 'NLP_IC', 'dIC', 'BL_Sh', 'NLP_Sh'))
print('  ' + '-' * 70)

for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df_w_sent[(df_w_sent['trade_date']>=tr_s) & (df_w_sent['trade_date']<=tr_e)]
    te = df_w_sent[(df_w_sent['trade_date']>=te_s) & (df_w_sent['trade_date']<=te_e)]
    if len(tr) < 5000: continue

    bl = backtest(tr, te, base_feats)
    nl = backtest(tr, te, nlp_feats)
    dic = nl['ic'] - bl['ic']
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f' % (
        label, bl['ic'], nl['ic'], dic, bl['sh'], nl['sh']))

print('\nDone.')
