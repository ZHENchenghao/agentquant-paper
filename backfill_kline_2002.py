# -*- coding: utf-8 -*-
"""Backfill kline_daily 2002-2015 from akshare"""
import sys, io, os, time, ssl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'

import akshare as ak
import duckdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
START = '20020101'
END = '20151231'

print('=' * 70)
print('Backfill kline_daily: %s -> %s' % (START[:4], END[:4]))
print('=' * 70)

# Get all stock codes
con = duckdb.connect(DB, read_only=True)
all_stocks = con.execute("""
    SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date >= '2016-01-01'
    AND ts_code NOT LIKE 'sh000%%' AND ts_code NOT LIKE 'sz399%%'
    AND ts_code NOT LIKE 'sh688%%'
""").fetchall()
con.close()

stocks_ak = []
for r in all_stocks:
    ts = r[0]
    code = ts[2:]  # 'sh600015' -> '600015', 'sz000001' -> '000001'
    stocks_ak.append((ts, code))

print('%d stocks to backfill' % len(stocks_ak))

# Check existing coverage
con = duckdb.connect(DB, read_only=True)
existing = con.execute("SELECT count(*) FROM kline_daily WHERE trade_date < '2016-01-01' AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%'").fetchone()[0]
con.close()
print('Existing pre-2016 rows: %d' % existing)

if existing > 100000:
    print('Already have substantial pre-2016 data, skipping.')
    sys.exit(0)

# ============================================================
# Multi-threaded download
# ============================================================
def download_one(args):
    ts_code, ak_code = args
    try:
        df = ak.stock_zh_a_hist(symbol=ak_code, period='daily',
                                start_date=START, end_date=END, adjust='qfq')
        if df is None or len(df) == 0:
            return None

        # Rename columns
        col_map = {
            '日期': 'trade_date', '开盘': 'open', '收盘': 'close',
            '最高': 'high', '最低': 'low', '成交量': 'vol',
            '成交额': 'amount', '换手率': 'turnover_rate',
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # Keep only needed columns
        keep = ['ts_code', 'trade_date', 'open', 'close', 'high', 'low', 'vol']
        df = df[[c for c in keep if c in df.columns]]
        return df
    except Exception as e:
        return None

print('Downloading %d stocks (%d threads)...' % (len(stocks_ak), 8))
dfs = []
done = 0
errors = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(download_one, s): s for s in stocks_ak}
    for f in as_completed(futures):
        done += 1
        result = f.result()
        if result is not None and len(result) > 0:
            dfs.append(result)
        else:
            errors += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (len(stocks_ak) - done)
            print('  %d/%d (%.0f%%) %d rows, %d err, ETA %.0fs' % (
                done, len(stocks_ak), 100*done/len(stocks_ak), sum(len(d) for d in dfs), errors, eta))

elapsed = time.time() - t0
print('Done in %.0fs: %d stocks downloaded, %d errors' % (elapsed, len(dfs), errors))

if not dfs:
    print('No data downloaded!')
    sys.exit(1)

# Merge all
df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=['ts_code', 'trade_date'])
print('Total: %d rows, %d stocks' % (len(df_all), df_all['ts_code'].nunique()))
print('Date range: %s ~ %s' % (df_all['trade_date'].min(), df_all['trade_date'].max()))

# Fill total_share and other missing columns (use most recent value from 2016+)
con = duckdb.connect(DB)
shares = con.execute("""
    SELECT ts_code, MAX(total_share) AS total_share
    FROM kline_daily WHERE trade_date >= '2016-01-01' AND total_share > 0
    GROUP BY ts_code
""").df()
con.close()

df_all = df_all.merge(shares, on='ts_code', how='left')
df_all['total_share'] = df_all['total_share'].fillna(0)
df_all['amount'] = 0
df_all['turnover_rate'] = 0
df_all['is_st'] = 0
df_all['data_source'] = 'akshare_backfill'

# Insert into DuckDB
print('Inserting into DuckDB...')
con = duckdb.connect(DB)

# Delete any existing pre-2016 data to avoid duplicates
con.execute("DELETE FROM kline_daily WHERE trade_date < '2016-01-01'")

# Batch insert
batch_size = 50000
total_rows = len(df_all)
inserted = 0
for i in range(0, total_rows, batch_size):
    batch = df_all.iloc[i:i+batch_size]
    con.execute("BEGIN")
    for _, row in batch.iterrows():
        try:
            con.execute("""
                INSERT INTO kline_daily (ts_code, trade_date, open, close, high, low, vol, amount, total_share, turnover_rate, is_st, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                row['ts_code'], row['trade_date'],
                float(row.get('open', 0) or 0), float(row.get('close', 0) or 0),
                float(row.get('high', 0) or 0), float(row.get('low', 0) or 0),
                float(row.get('vol', 0) or 0), float(row.get('amount', 0) or 0),
                float(row.get('total_share', 0) or 0), float(row.get('turnover_rate', 0) or 0),
                0, 'akshare_backfill'
            ])
        except:
            pass
    con.execute("COMMIT")
    inserted += len(batch)
    print('  %d/%d (%.0f%%)' % (inserted, total_rows, 100*inserted/total_rows))

# Verify
n = con.execute("SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT ts_code) FROM kline_daily WHERE data_source = 'akshare_backfill'").fetchone()
print('Verified: %d rows, %d stocks, %s ~ %s' % (n[0], n[3], n[1], n[2]))

con.close()
print('Done.')
