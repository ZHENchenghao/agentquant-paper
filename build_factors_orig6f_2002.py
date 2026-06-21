# -*- coding: utf-8 -*-
"""
从2002年起重建原始6因子: Turnover(-), Amihud, Price(-), Max_Ret(-), VP_Corr, Short_Rev
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
OUT = 'D:/AgentQuant/our/cache/factors_orig6f_2002.parquet'

con = duckdb.connect(DB, read_only=True)
print("=" * 60)
print("重建原始6因子 (2002起)")
print("=" * 60)

con.execute("""
CREATE TEMP TABLE base_daily AS
SELECT
    ts_code, trade_date, open, high, low, close, vol,
    COALESCE(amount, GREATEST(vol * close, 1.0)) AS amount_proxy,
    close / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret_1d,
    LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_close,
    COALESCE(turnover_rate, vol / NULLIF(total_share, 0) * 100) AS turnover
FROM kline_daily WHERE trade_date >= '2002-01-01'
""")

# 1. Amihud (log)
print("[1] Amihud...")
con.execute("""
CREATE TEMP TABLE amihud_20d AS
SELECT ts_code, trade_date,
    LN(1.0 + AVG(ABS(ret_1d) / NULLIF(GREATEST(amount_proxy, 1.0), 0) * 1e10) OVER (
        PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    )) AS amihud
FROM base_daily WHERE ret_1d IS NOT NULL
""")

# 2. Max_Ret(-)
print("[2] Max_Rev...")
con.execute("""
CREATE TEMP TABLE maxrev_20d AS
SELECT ts_code, trade_date,
    -MAX(ret_1d) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS max_rev
FROM base_daily WHERE ret_1d IS NOT NULL
""")

# 3. Price(-) — 股价负向: 低价股有散户偏好
print("[3] Price_Rev...")
con.execute("""
CREATE TEMP TABLE price_daily AS
SELECT ts_code, trade_date,
    -close AS price_rev
FROM base_daily
""")

# 4. Turnover(-) — 换手率负向
print("[4] Turnover_Rev...")
con.execute("""
CREATE TEMP TABLE turnover_rev_20d AS
SELECT ts_code, trade_date,
    -AVG(turnover) OVER (
        PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS turnover_rev
FROM base_daily WHERE turnover IS NOT NULL
""")

# 5. VP_Corr
print("[5] VP_Corr...")
con.execute("""
CREATE TEMP TABLE vpcorr_10d AS
SELECT ts_code, trade_date,
    CASE
        WHEN STDDEV_POP(ret_1d) OVER w * STDDEV_POP(vol_change) OVER w > 0
        THEN (AVG(ret_1d * vol_change) OVER w - AVG(ret_1d) OVER w * AVG(vol_change) OVER w)
             / NULLIF(STDDEV_POP(ret_1d) OVER w * STDDEV_POP(vol_change) OVER w, 0)
        ELSE 0
    END AS vp_corr
FROM (
    SELECT ts_code, trade_date, ret_1d,
        vol / NULLIF(LAG(vol) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) - 1 AS vol_change
    FROM base_daily WHERE ret_1d IS NOT NULL
) sub
WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
""")

# 6. Short_Rev (5日反转)
print("[6] Short_Rev...")
con.execute("""
CREATE TEMP TABLE sr5_daily AS
SELECT ts_code, trade_date,
    -(close / LAG(close, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1) AS sr5
FROM base_daily
""")

# 合并
print("[7] 合并6因子...")
con.execute("""
CREATE TEMP TABLE factors_merged AS
SELECT
    b.ts_code, b.trade_date, b.close, b.amount_proxy,
    a.amihud, m.max_rev, p.price_rev,
    COALESCE(s.sr5, 0) AS sr5,
    COALESCE(v.vp_corr, 0) AS vp_corr,
    COALESCE(t.turnover_rev, 0) AS turnover_rev
FROM base_daily b
LEFT JOIN amihud_20d a ON b.ts_code=a.ts_code AND b.trade_date=a.trade_date
LEFT JOIN maxrev_20d m ON b.ts_code=m.ts_code AND b.trade_date=m.trade_date
LEFT JOIN price_daily p ON b.ts_code=p.ts_code AND b.trade_date=p.trade_date
LEFT JOIN sr5_daily s ON b.ts_code=s.ts_code AND b.trade_date=s.trade_date
LEFT JOIN vpcorr_10d v ON b.ts_code=v.ts_code AND b.trade_date=v.trade_date
LEFT JOIN turnover_rev_20d t ON b.ts_code=t.ts_code AND b.trade_date=t.trade_date
WHERE a.amihud IS NOT NULL AND m.max_rev IS NOT NULL
""")

stats = con.execute("""
SELECT MIN(trade_date), MAX(trade_date), COUNT(*), COUNT(DISTINCT ts_code), COUNT(DISTINCT trade_date)
FROM factors_merged
""").fetchone()
print(f"日期: {stats[0]}~{stats[1]}, 行数: {stats[2]:,}, 股票: {stats[3]}, 交易日: {stats[4]}")

print(f"[8] 写出 {OUT} ...")
con.execute(f"COPY factors_merged TO '{OUT}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 500000)")
con.close()
print(f"完成! {time.time()-t0:.0f}s")
