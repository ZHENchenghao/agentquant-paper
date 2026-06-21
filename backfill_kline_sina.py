# -*- coding: utf-8 -*-
"""Backfill kline_daily 2002-2015 via Sina API (akshare)"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import akshare as ak

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
START, END = '2002-01-01', '2015-12-31'
THREADS = 4

print('Sina Backfill: %s -> %s (%d threads)' % (START[:4], END[:4], THREADS))

# Get stock list (only stocks that exist in 2016+)
con = duckdb.connect(DB, read_only=True)
codes = con.execute("""
    SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date >= '2016-01-01'
    AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%' AND ts_code NOT LIKE 'sh688%'
""").fetchall()
con.close()

tasks = [r[0] for r in codes if r[0].startswith(('sh','sz'))]
print('%d stocks to download' % len(tasks))

def download(ts_code):
    try:
        df = ak.stock_zh_a_daily(symbol=ts_code, adjust='qfq')
        if df is None or len(df) == 0: return None
        df = df.rename(columns={'date':'trade_date','volume':'vol'})
        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        # Filter to 2002-2015
        df = df[(df['trade_date'] >= START) & (df['trade_date'] <= END)]
        if len(df) == 0: return None
        keep = ['ts_code','trade_date','open','high','low','close','vol']
        return df[[c for c in keep if c in df.columns]]
    except:
        return None

all_dfs = []
done = errors = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=THREADS) as pool:
    futures = {pool.submit(download, t): t for t in tasks}
    for f in as_completed(futures):
        done += 1
        r = f.result()
        if r is not None: all_dfs.append(r)
        else: errors += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(tasks) - done) / rate
            rows = sum(len(d) for d in all_dfs)
            print('  %d/%d %d rows %derr %.0f/s ETA %.0fs' % (done, len(tasks), rows, errors, rate, eta))

elapsed = time.time() - t0
print('Done: %.0fs, %d stocks, %d errors' % (elapsed, len(all_dfs), errors))

if not all_dfs:
    print('No data!'); sys.exit(1)

df_all = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['ts_code','trade_date'])
print('Total: %d rows, %d stocks, %s ~ %s' % (len(df_all), df_all['ts_code'].nunique(),
    df_all['trade_date'].min().date(), df_all['trade_date'].max().date()))

# Add missing columns, insert
con = duckdb.connect(DB)
shares = con.execute("SELECT ts_code, MAX(total_share) AS ts FROM kline_daily WHERE trade_date >= '2016-01-01' AND total_share > 0 GROUP BY ts_code").df()
con.close()
df_all = df_all.merge(shares, on='ts_code', how='left')
df_all['total_share'] = df_all['ts'].fillna(0)
df_all['amount'] = 0.0
df_all['turnover_rate'] = 0.0

con = duckdb.connect(DB)
con.execute("DELETE FROM kline_daily WHERE trade_date < '2016-01-01'")
bsize = 50000
for i in range(0, len(df_all), bsize):
    batch = df_all.iloc[i:i+bsize]
    con.execute("BEGIN")
    for _, row in batch.iterrows():
        con.execute("INSERT INTO kline_daily (ts_code,trade_date,open,close,high,low,vol,amount,total_share,turnover_rate,is_st,data_source) VALUES (?,?,?,?,?,?,?,0,?,0,0,'sina')",
            [row['ts_code'], row['trade_date'], float(row['open']), float(row['close']),
             float(row['high']), float(row['low']), float(row['vol']), float(row['total_share'])])
    con.execute("COMMIT")
    if i % 500000 == 0: print('  insert %d/%d' % (i+bsize, len(df_all)))

n = con.execute("SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT ts_code) FROM kline_daily WHERE data_source='sina'").fetchone()
print('Done: %d rows, %d stocks, %s ~ %s' % (n[0], n[3], n[1], n[2]))
con.close()
