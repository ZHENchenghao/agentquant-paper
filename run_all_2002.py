# -*- coding: utf-8 -*-
"""
One script: Backfill 2002-2015 -> Rebuild factors -> Rolling backtest + NLP -> Shutdown
Self-contained, saves checkpoints, error-resilient.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['TQDM_DISABLE'] = '1'

from datetime import datetime
def log(msg):
    print('[%s] %s' % (datetime.now().strftime('%H:%M:%S'), msg), flush=True)

log('=== Pipeline START ===')

# ============================================================
# PHASE 1: Backfill K-line 2002-2015
# ============================================================
import duckdb, pandas as pd, numpy as np
import akshare as ak

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE_DIR = 'cache'

log('Phase 1: Backfill K-line 2002-2015')

# Check if already done
con = duckdb.connect(DB, read_only=True)
existing = con.execute("SELECT count(*) FROM kline_daily WHERE trade_date < '2016-01-01' AND data_source IN ('sina','baostock','akshare_backfill')").fetchone()[0]
con.close()
if existing > 1000000:
    log('  Already backfilled: %d rows, skipping' % existing)
else:
    con = duckdb.connect(DB, read_only=True)
    codes = con.execute("""
        SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date >= '2016-01-01'
        AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%' AND ts_code NOT LIKE 'sh688%'
    """).fetchall()
    con.close()
    tasks = [r[0] for r in codes]
    log('  %d stocks to download' % len(tasks))

    all_dfs = []
    done = errors = 0
    t0 = time.time()

    for ts in tasks:
        try:
            df = ak.stock_zh_a_daily(symbol=ts, adjust='qfq')
            if df is not None and len(df) > 0:
                df = df.rename(columns={'date':'trade_date','volume':'vol'})
                df['ts_code'] = ts
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[(df['trade_date']>='2002-01-01')&(df['trade_date']<='2015-12-31')]
                if len(df)>0:
                    all_dfs.append(df[['ts_code','trade_date','open','high','low','close','vol']])
        except:
            errors += 1
        done += 1
        if done % 500 == 0:
            elapsed = time.time()-t0
            log('  %d/%d %d dfs %d err %.0f/s' % (done, len(tasks), len(all_dfs), errors, done/elapsed))

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['ts_code','trade_date'])
        log('  Total: %d rows, inserting...' % len(df_all))

        con = duckdb.connect(DB)
        con.execute("DELETE FROM kline_daily WHERE trade_date < '2016-01-01'")
        bsize = 50000
        for i in range(0, len(df_all), bsize):
            batch = df_all.iloc[i:i+bsize]
            con.execute("BEGIN")
            for _, row in batch.iterrows():
                con.execute("INSERT INTO kline_daily (ts_code,trade_date,open,close,high,low,vol,amount,total_share,turnover_rate,is_st,data_source) VALUES (?,?,?,?,?,?,?,0,0,0,0,'sina')",
                    [row['ts_code'], row['trade_date'], float(row['open']), float(row['close']),
                     float(row['high']), float(row['low']), float(row['vol'])])
            con.execute("COMMIT")
        n = con.execute("SELECT count(*), min(trade_date), max(trade_date) FROM kline_daily WHERE data_source='sina'").fetchone()
        log('  Inserted: %d rows, %s ~ %s' % (n[0], n[1], n[2]))
        con.close()
        # Save checkpoint
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, 'phase1_done.txt'), 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M'))

# ============================================================
# PHASE 2: Rebuild Factor Cache from 2002
# ============================================================
log('Phase 2: Rebuild Factor Cache')

# Check if cached
factor_cache_file = os.path.join(CACHE_DIR, 'factors_2002_full.parquet')
if os.path.exists(factor_cache_file):
    log('  Factor cache exists, skipping rebuild')
else:
    log('  Computing I-factors (valuation)...')
    con = duckdb.connect(DB, read_only=True)
    # Simplified: use log_mcap from close*total_share, PE/PB/PS from financials
    i_factors = con.execute("""
        WITH fin AS (
            SELECT CASE WHEN ts_code LIKE '%.SH' THEN 'sh'||SUBSTR(ts_code,1,6)
                        WHEN ts_code LIKE '%.SZ' THEN 'sz'||SUBSTR(ts_code,1,6) END AS ts_code,
                   report_date, eps, shareholders_equity, revenue, total_assets
            FROM financial_statements WHERE report_date >= '2001-01-01'
        ),
        kline AS (SELECT ts_code, trade_date, close, total_share FROM kline_daily WHERE trade_date >= '2002-01-01'),
        joined AS (
            SELECT k.*, f.eps, f.shareholders_equity, f.revenue, f.total_assets,
                   ROW_NUMBER() OVER(PARTITION BY k.ts_code, k.trade_date ORDER BY f.report_date DESC) AS rn
            FROM kline k LEFT JOIN fin f ON k.ts_code=f.ts_code
                AND f.report_date <= k.trade_date AND f.report_date >= k.trade_date - INTERVAL 270 DAY
        )
        SELECT ts_code, trade_date, close, eps, shareholders_equity, revenue, total_assets, total_share
        FROM joined WHERE rn=1
    """).df()
    con.close()

    i_factors['pe'] = i_factors['close'] / i_factors['eps'].clip(lower=0.01)
    i_factors['pb'] = i_factors['close'] / (i_factors['shareholders_equity'] / i_factors['total_share']).clip(lower=0.01)
    i_factors['ps'] = i_factors['close'] / ((i_factors['revenue'] / i_factors['total_share']).clip(lower=0.01))
    i_factors['log_mcap'] = np.log((i_factors['close'] * i_factors['total_share'] / 10000).clip(lower=1))
    i_factors = i_factors[['ts_code','trade_date','pe','pb','ps','log_mcap']]
    log('  I-factors: %d rows' % len(i_factors))

    log('  Computing H-factors (technical)...')
    con = duckdb.connect(DB, read_only=True)
    h_factors = con.execute("""
        WITH p AS (
            SELECT ts_code, trade_date, close, vol,
                   LAG(close,1) OVER w AS c1, LAG(close,5) OVER w AS c5,
                   LAG(close,20) OVER w AS c20, LAG(close,60) OVER w AS c60,
                   LAG(close,120) OVER w AS c120,
                   AVG(close) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS ma20,
                   AVG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS vol_ma20,
                   STDDEV(close) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS std20,
                   AVG(CASE WHEN close>LAG(close,1) OVER w THEN close-LAG(close,1) OVER w ELSE 0 END)
                       OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 5 PRECEDING) AS gain6,
                   AVG(ABS(close-LAG(close,1) OVER w))
                       OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 5 PRECEDING) AS abs6,
                   AVG(CASE WHEN close>LAG(close,1) OVER w THEN close-LAG(close,1) OVER w ELSE 0 END)
                       OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS gain14,
                   AVG(ABS(close-LAG(close,1) OVER w))
                       OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING) AS abs14
            FROM kline_daily WHERE trade_date >= '2002-01-01'
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        )
        SELECT ts_code, trade_date,
               100*gain6/NULLIF(abs6,0) AS rsi6,
               100*gain14/NULLIF(abs14,0) AS rsi14,
               (close-ma20)/NULLIF(std20,0) AS boll_pos,
               std20/NULLIF(ma20,0) AS boll_width,
               close/NULLIF(c20,0)-1 AS div_ma20,
               close/NULLIF(c60,0)-1 AS div_ma60,
               close/NULLIF(c120,0)-1 AS div_ma120,
               vol/NULLIF(vol_ma20,0) AS vol_ratio
        FROM p WHERE c1 IS NOT NULL
    """).df()
    con.close()
    h_factors['rsi_extreme'] = np.where(h_factors['rsi6']>70, 1, np.where(h_factors['rsi6']<30, -1, 0))
    h_factors['ma_score'] = ((h_factors['div_ma20']>0).astype(int)+(h_factors['div_ma60']>0).astype(int)+(h_factors['div_ma120']>0).astype(int))
    log('  H-factors: %d rows' % len(h_factors))

    log('  Merging...')
    df = i_factors.merge(h_factors, on=['ts_code','trade_date'], how='outer')
    # Add NLP placeholder (will be filled by later step)
    df['sent_20d'] = 0.0
    # Add missing C-factors with zeros
    for c in ['margin_panic','streak5_dn','nb_bull','nb_diverge','vix_stress',
              'roe','gross_margin','net_margin','profit_margin','log_eps']:
        df[c] = 0.0
    df = df.drop(columns=['close'], errors='ignore')
    df.to_parquet(factor_cache_file)
    log('  Saved: %d rows, %d stocks' % (len(df), df['ts_code'].nunique()))

# ============================================================
# PHASE 3: Rolling Backtest
# ============================================================
log('Phase 3: Rolling Backtest 2019-2024 + NLP')

from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# Load factor cache
df = pd.read_parquet(os.path.join(CACHE_DIR, 'factors_2002_full.parquet'))
df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
log('  Loaded: %d rows' % len(df))

# Quick target
con = duckdb.connect(DB, read_only=True)
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0)-(x.fc/x.close-1.0) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
target['trade_date'] = target['trade_date'].astype(str)

industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn=1""").df()
mcap = con.execute("""SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
    close*total_share/10000 AS mcap FROM kline_daily WHERE trade_date>='2016-01-01'""").df()
con.close()

# NLP sentiment from Astock
news = pd.read_parquet('D:/AgentQuant/Astock-main/astock_mapped.parquet')
news['trade_date_clean'] = pd.to_datetime(news['trade_date'], errors='coerce')
news['sentiment'] = news['label'].map({0:0, 1:1, 2:-1})
news = news.sort_values(['ts_code','trade_date_clean'])
news['sent_20d'] = news.groupby('ts_code')['sentiment'].transform(lambda x: x.rolling(20, min_periods=3).mean().shift(1))
daily_sent = news.groupby(['ts_code','trade_date_clean']).agg(sent_20d=('sent_20d','last')).reset_index()
daily_sent['trade_date'] = daily_sent['trade_date_clean'].dt.strftime('%Y-%m-%d')

# Merge all
for d in [target, mcap, daily_sent]: d['trade_date'] = d['trade_date'].astype(str)
df = df.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(daily_sent[['ts_code','trade_date','sent_20d']], on=['ts_code','trade_date'], how='left')
df['sent_20d'] = df['sent_20d'].fillna(0.0)
log('  Merged: %d rows' % len(df))

# Factor list
exclude = ['ts_code','trade_date','close','factor_group','_k','report_date','excess_ret','ind_name','mcap']
feats = [c for c in df.columns if c not in exclude and c!='sent_20d' and df[c].dtype in ('float64','float32','int64','int32')]
all_feats = feats + ['sent_20d']
log('  Factors: %d' % len(all_feats))

# Rolling backtest
results = []
for test_yr in range(2019, 2025):
    tr_s = '%d-01-01'%(test_yr-3); tr_e = '%d-12-31'%(test_yr-1)
    te_s = '%d-01-01'%test_yr; te_e = '%d-12-31'%test_yr
    tr = df[(df['trade_date']>=tr_s)&(df['trade_date']<=tr_e)].dropna(subset=['excess_ret'])
    te = df[(df['trade_date']>=te_s)&(df['trade_date']<=te_e)].dropna(subset=['excess_ret'])
    if len(tr)<10000: continue

    # Simple neutralization: sample-based
    for d in [tr, te]:
        d['ln_mcap'] = np.log(d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6).clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap']**2

    ind_tr = pd.get_dummies(tr['ind_name'], prefix='ind').fillna(0).astype(float)
    X_tr = np.nan_to_num(pd.concat([tr['ln_mcap'], tr['ln_mcap_sq'], ind_tr], axis=1).fillna(0).values, 0)
    y_df = tr[all_feats].fillna(tr[all_feats].median()).fillna(0)
    y_tr = np.nan_to_num(y_df.values, 0)
    if X_tr.shape[0]>50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        X_f, y_f = X_tr[idx], y_tr[idx]
    else:
        X_f, y_f = X_tr, y_tr

    m = LinearRegression(fit_intercept=False)
    m.fit(X_f, y_f)
    resid = y_tr - X_tr @ m.coef_.T
    neu_names = []
    for i, col in enumerate(all_feats):
        name = col+'_n'
        tr[name] = resid[:,i]
        tr[name] = (tr[name]-tr[name].mean())/tr[name].std()
        neu_names.append(name)

    # Apply to test
    ind_te = pd.get_dummies(te['ind_name'], prefix='ind').fillna(0).astype(float)
    X_te = np.nan_to_num(pd.concat([te['ln_mcap'], te['ln_mcap_sq'], ind_te], axis=1).fillna(0).values, 0)
    te_y = np.nan_to_num(te[all_feats].fillna(te[all_feats].median()).fillna(0).values, 0)
    te_resid = te_y - X_te @ m.coef_.T
    for i, col in enumerate(all_feats):
        name = col+'_n'
        te[name] = te_resid[:,i]
        te[name] = (te[name]-tr[name].mean())/tr[name].std()

    base_neu = [n for n in neu_names if not n.startswith('sent_')]
    all_neu = neu_names

    for feats_use, label in [(base_neu,'BL'),(all_neu,'NLP')]:
        flist = [f for f in feats_use if f in tr.columns]
        model = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                               subsample=0.8, colsample_bytree=0.8,
                               n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
        model.fit(tr[flist].fillna(tr[flist].median()), tr['excess_ret'].fillna(0))
        te['pred'] = model.predict(te[flist].fillna(tr[flist].median()))
        ic = np.corrcoef(te['pred'].dropna(), te.loc[te['pred'].notna(),'excess_ret'])[0,1]
        te['ym'] = pd.to_datetime(te['trade_date']).dt.to_period('M')
        mrets = []
        for mo, g in te.groupby('ym'):
            if len(g)<30: continue
            mrets.append(g.nlargest(30,'pred')['excess_ret'].mean())
        rets = np.array(mrets) if mrets else np.array([0])
        sh = np.mean(rets)*12/(np.std(rets,ddof=1)*np.sqrt(12)) if len(rets)>2 and np.std(rets)>0 else 0
        results.append({'year':test_yr,'model':label,'ic':ic,'sharpe':sh,'mr':np.mean(rets)})
        log('  %d %s: IC=%.4f Sharpe=%.3f' % (test_yr, label, ic, sh))

# Save results
res_df = pd.DataFrame(results)
res_df.to_parquet(os.path.join(CACHE_DIR, 'backtest_2002_2024.parquet'))
log('Results saved: %d records' % len(res_df))

# Summary
for label in ['BL','NLP']:
    sub = res_df[res_df['model']==label]
    log('%s: avg IC=%.4f avg Sharpe=%.3f' % (label, sub['ic'].mean(), sub['sharpe'].mean()))

log('=== Pipeline COMPLETE ===')
log('Total time: %.0f min' % ((time.time()-t0)/60))
