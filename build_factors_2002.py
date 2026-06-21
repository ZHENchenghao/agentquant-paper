# -*- coding: utf-8 -*-
"""
从2002年起重建5因子: Amihud, Max_Rev, Gap, Short_Rev, VP_Corr
使用DuckDB窗口函数，高效处理全A股24年数据
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
OUT = 'D:/AgentQuant/our/cache/factors_5f_2002.parquet'

con = duckdb.connect(DB, read_only=True)

print("=" * 60)
print("从2002年重建5因子")
print("=" * 60)

# Step 1: 构建基础日频数据 (带成交额代理)
print("[1] 构建基础OHLCV+成交额代理...")
sql_base = """
CREATE TEMP TABLE base_daily AS
SELECT
    ts_code, trade_date,
    open, high, low, close, vol,
    -- 成交额代理: 优先amount, 否则vol*close
    COALESCE(amount, GREATEST(vol * close, 1.0)) AS amount_proxy,
    -- 日收益率
    close / LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret_1d,
    -- 前收盘
    LAG(close) OVER (PARTITION BY ts_code ORDER BY trade_date) AS prev_close,
    -- 换手率(如果无则用vol/total_share代理)
    COALESCE(turnover_rate, vol / NULLIF(total_share, 0) * 100) AS turnover
FROM kline_daily
WHERE trade_date >= '2002-01-01'
"""
con.execute(sql_base)

# Step 2: Amihud日频illiq
print("[2] 计算Amihud日频illiq...")
sql_illiq = """
CREATE TEMP TABLE illiq_daily AS
SELECT
    ts_code, trade_date,
    ABS(ret_1d) / NULLIF(GREATEST(amount_proxy, 1.0), 0) * 1e10 AS illiq_raw
FROM base_daily
WHERE ret_1d IS NOT NULL
"""
con.execute(sql_illiq)

# Step 3: Amihud 20日均值 + log
print("[3] 计算Amihud 20日滚动均值+log...")
sql_amihud = """
CREATE TEMP TABLE amihud_20d AS
SELECT
    ts_code, trade_date,
    LN(1.0 + AVG(illiq_raw) OVER (
        PARTITION BY ts_code ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    )) AS amihud
FROM illiq_daily
WHERE illiq_raw IS NOT NULL
"""
con.execute(sql_amihud)

# Step 4: Max_Rev (20日内最大日收益的负值)
print("[4] 计算Max_Rev (20日最大收益的负值)...")
sql_maxrev = """
CREATE TEMP TABLE maxrev_20d AS
SELECT
    ts_code, trade_date,
    -MAX(ret_1d) OVER (
        PARTITION BY ts_code ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ) AS max_rev
FROM base_daily
WHERE ret_1d IS NOT NULL
"""
con.execute(sql_maxrev)

# Step 5: Gap (当日跳空缺口反转)
print("[5] 计算Gap (跳空缺口)...")
sql_gap = """
CREATE TEMP TABLE gap_daily AS
SELECT
    ts_code, trade_date,
    -- gap = open/prev_close - 1, 反转方向: -gap
    -(open / NULLIF(prev_close, 0) - 1) AS gap
FROM base_daily
WHERE prev_close IS NOT NULL AND prev_close > 0
"""
con.execute(sql_gap)

# Step 6: Short_Rev (5日反转)
print("[6] 计算Short_Rev (5日反转)...")
sql_sr5 = """
CREATE TEMP TABLE sr5_daily AS
SELECT
    ts_code, trade_date,
    -(close / LAG(close, 5) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1) AS sr5
FROM base_daily
"""
con.execute(sql_sr5)

# Step 7: VP_Corr (量价相关性, 10日滚动)
print("[7] 计算VP_Corr (10日量价相关性)...")
sql_vpcorr = """
CREATE TEMP TABLE vpcorr_10d AS
SELECT
    ts_code, trade_date,
    -- DuckDB的CORR不支持窗口函数，改用公式: COV/(STDDEV_POP(v)*STDDEV_POP(p))
    CASE
        WHEN STDDEV_POP(ret_1d) OVER w * STDDEV_POP(vol_change) OVER w > 0
        THEN (AVG(ret_1d * vol_change) OVER w - AVG(ret_1d) OVER w * AVG(vol_change) OVER w)
             / NULLIF(STDDEV_POP(ret_1d) OVER w * STDDEV_POP(vol_change) OVER w, 0)
        ELSE 0
    END AS vp_corr
FROM (
    SELECT
        ts_code, trade_date, ret_1d,
        vol / NULLIF(LAG(vol) OVER (PARTITION BY ts_code ORDER BY trade_date), 0) - 1 AS vol_change
    FROM base_daily
    WHERE ret_1d IS NOT NULL
) sub
WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)
"""
con.execute(sql_vpcorr)

# Step 8: 合并所有因子
print("[8] 合并5因子...")
sql_merge = """
CREATE TEMP TABLE factors_merged AS
SELECT
    b.ts_code, b.trade_date,
    b.close, b.amount_proxy,
    a.amihud,
    m.max_rev,
    g.gap,
    COALESCE(s.sr5, 0) AS sr5,
    COALESCE(v.vp_corr, 0) AS vp_corr
FROM base_daily b
LEFT JOIN amihud_20d a ON b.ts_code = a.ts_code AND b.trade_date = a.trade_date
LEFT JOIN maxrev_20d m ON b.ts_code = m.ts_code AND b.trade_date = m.trade_date
LEFT JOIN gap_daily g ON b.ts_code = g.ts_code AND b.trade_date = g.trade_date
LEFT JOIN sr5_daily s ON b.ts_code = s.ts_code AND b.trade_date = s.trade_date
LEFT JOIN vpcorr_10d v ON b.ts_code = v.ts_code AND b.trade_date = v.trade_date
WHERE a.amihud IS NOT NULL
  AND m.max_rev IS NOT NULL
  AND g.gap IS NOT NULL
"""
con.execute(sql_merge)

# Step 9: 统计
print("[9] 统计...")
stats = con.execute("""
SELECT
    MIN(trade_date) AS start_dt, MAX(trade_date) AS end_dt,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT trade_date) AS trading_days,
    COUNT(DISTINCT ts_code) AS stocks,
    AVG(amihud) AS avg_amihud, STDDEV(amihud) AS std_amihud,
    AVG(max_rev) AS avg_maxrev, AVG(gap) AS avg_gap,
    AVG(sr5) AS avg_sr5, AVG(vp_corr) AS avg_vpcorr
FROM factors_merged
""").fetchone()

print(f"日期范围: {stats[0]} ~ {stats[1]}")
print(f"总行数: {stats[2]:,}")
print(f"交易日: {stats[3]}")
print(f"股票数: {stats[4]}")
print(f"Amihud: mean={stats[5]:.4f} std={stats[6]:.4f}")
print(f"Max_Rev: mean={stats[7]:.4f}")
print(f"Gap: mean={stats[8]:.4f}")
print(f"SR5: mean={stats[9]:.4f}")
print(f"VP_Corr: mean={stats[10]:.4f}")

# Step 10: 写出parquet
print(f"\n[10] 写出 {OUT} ...")
con.execute(f"""
COPY factors_merged TO '{OUT}' (FORMAT PARQUET, COMPRESSION 'zstd', ROW_GROUP_SIZE 500000)
""")

con.close()

elapsed = time.time() - t0
print(f"完成! 耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")
