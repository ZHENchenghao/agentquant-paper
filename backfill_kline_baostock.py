# -*- coding: utf-8 -*-
"""Backfill kline_daily 2002-2015 via BaoStock"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import baostock as bs
import duckdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
START = '2002-01-01'
END = '2015-12-31'

print('=' * 70)
print('BaoStock Backfill: %s -> %s' % (START[:4], END[:4]))
print('=' * 70)

# Login
bs.login()

# Get all stock codes from DB
con = duckdb.connect(DB, read_only=True)
all_codes = con.execute("""
    SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date >= '2016-01-01'
    AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%'
    AND ts_code NOT LIKE 'sh688%'
""").fetchall()
con.close()

# Convert to BaoStock format
bs_codes = []
for r in all_codes:
    ts = r[0]
    if ts.startswith('sh'):
        bs_codes.append((ts, 'sh.' + ts[2:]))
    elif ts.startswith('sz'):
        bs_codes.append((ts, 'sz.' + ts[2:]))

print('%d stocks to download' % len(bs_codes))

# ============================================================
def download_bs(args):
    ts_code, bs_code = args
    try:
        rs = bs.query_history_k_data_plus(bs_code,
            'date,open,high,low,close,volume',
            start_date=START, end_date=END,
            frequency='d', adjustflag='2')  # forward-adjusted
        if rs.error_code != '0':
            return None

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        if not data:
            return None

        df = pd.DataFrame(data, columns=rs.fields)
        df.columns = ['trade_date', 'open', 'high', 'low', 'close', 'vol']
        df['ts_code'] = ts_code
        for c in ['open', 'high', 'low', 'close', 'vol']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.dropna(subset=['close'])
        return df
    except:
        return None

# Sequential download (BaoStock doesn't like concurrency)
print('Downloading...')
all_dfs = []
done = 0
errors = 0
t0 = time.time()

for args in bs_codes:
    result = download_bs(args)
    done += 1
    if result is not None and len(result) > 0:
        all_dfs.append(result)
    else:
        errors += 1
    if done % 500 == 0:
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (len(bs_codes) - done) / rate
        rows = sum(len(d) for d in all_dfs)
        print('  %d/%d (%.0f%%) %d rows, %d err, %.0f/s, ETA %.0fs' % (
            done, len(bs_codes), 100*done/len(bs_codes), rows, errors, rate, eta))

elapsed = time.time() - t0
print('Done in %.0fs: %d stocks, %d errors' % (elapsed, len(all_dfs), errors))

if not all_dfs:
    print('No data!')
    bs.logout()
    sys.exit(1)

# Merge
df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=['ts_code', 'trade_date'])
print('Total: %d rows, %d stocks, %s ~ %s' % (
    len(df_all), df_all['ts_code'].nunique(),
    df_all['trade_date'].min().date(), df_all['trade_date'].max().date()))

# Add missing columns
con = duckdb.connect(DB)
shares = con.execute("""
    SELECT ts_code, MAX(total_share) AS total_share
    FROM kline_daily WHERE trade_date >= '2016-01-01' AND total_share > 0
    GROUP BY ts_code
""").df()
con.close()

df_all = df_all.merge(shares, on='ts_code', how='left')
df_all['total_share'] = df_all['total_share'].fillna(0)
df_all['amount'] = 0.0
df_all['turnover_rate'] = 0.0
df_all['is_st'] = 0
df_all['data_source'] = 'baostock_backfill'

# Insert
print('Inserting into DuckDB...')
con = duckdb.connect(DB)
con.execute("DELETE FROM kline_daily WHERE trade_date < '2016-01-01'")

batch_size = 50000
total = len(df_all)
for i in range(0, total, batch_size):
    batch = df_all.iloc[i:i+batch_size]
    con.execute("BEGIN")
    for _, row in batch.iterrows():
        try:
            con.execute("""
                INSERT INTO kline_daily (ts_code, trade_date, open, close, high, low, vol, amount, total_share, turnover_rate, is_st, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [row['ts_code'], row['trade_date'],
                  float(row['open']), float(row['close']), float(row['high']),
                  float(row['low']), float(row['vol']), float(row['amount']),
                  float(row['total_share']), float(row['turnover_rate']), 0, 'baostock_backfill'])
        except:
            pass
    con.execute("COMMIT")
    if (i // batch_size) % 10 == 0:
        print('  %d/%d' % (i+batch_size, total))

# Verify
n = con.execute("SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT ts_code) FROM kline_daily WHERE data_source = 'baostock_backfill'").fetchone()
print('Verified: %d rows, %d stocks, %s ~ %s' % (n[0], n[3], n[1], n[2]))
con.close()

bs.logout()
print('Done.')
