# -*- coding: utf-8 -*-
"""
Rebuild factor cache from 2002.
Converts financial .SH/.SZ -> sh/sz prefix for JOIN.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from datetime import datetime

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
OUT = 'cache/factors_2002.parquet'

print('=' * 70)
print('Rebuilding Factor Cache from 2002')
print('=' * 70)

con = duckdb.connect(DB, read_only=True)

# ============================================================
# I组: 估值因子 (PE, PB, PS, log_mcap)
# ============================================================
print('\n[I] Valuation factors...')

# Convert financial ts_code format for JOIN
i_factors = con.execute("""
WITH fin AS (
    SELECT
        CASE WHEN ts_code LIKE '%.SH' THEN 'sh' || SUBSTR(ts_code, 1, 6)
             WHEN ts_code LIKE '%.SZ' THEN 'sz' || SUBSTR(ts_code, 1, 6)
             ELSE ts_code END AS ts_code,
        report_date, report_type,
        eps, net_profit, shareholders_equity, revenue, total_assets
    FROM financial_statements
    WHERE report_date >= '2001-01-01'
),
kline AS (
    SELECT ts_code, trade_date, close, total_share
    FROM kline_daily WHERE trade_date >= '2002-01-01'
),
-- Match each trade_date to most recent report_date (within 270 days)
joined AS (
    SELECT k.ts_code, k.trade_date, k.close, k.total_share,
           f.eps, f.net_profit, f.shareholders_equity, f.revenue, f.total_assets,
           f.report_date,
           ROW_NUMBER() OVER(PARTITION BY k.ts_code, k.trade_date
               ORDER BY f.report_date DESC) AS rn
    FROM kline k
    LEFT JOIN fin f ON k.ts_code = f.ts_code
        AND f.report_date <= k.trade_date
        AND f.report_date >= k.trade_date - INTERVAL 270 DAY
)
SELECT ts_code, trade_date, close, eps, net_profit, shareholders_equity,
       revenue, total_assets, total_share
FROM joined WHERE rn = 1
""").df()

# Compute I factors
i_factors['pe'] = i_factors['close'] / i_factors['eps'].clip(lower=0.01)
i_factors['pb'] = i_factors['close'] / (i_factors['shareholders_equity'] / i_factors['total_share']).clip(lower=0.01)
i_factors['ps'] = i_factors['close'] / ((i_factors['revenue'] / i_factors['total_share']).clip(lower=0.01))
i_factors['log_mcap'] = np.log((i_factors['close'] * i_factors['total_share'] / 10000).clip(lower=1))
i_factors = i_factors[['ts_code', 'trade_date', 'pe', 'pb', 'ps', 'log_mcap']]
print('  I-factors: %d rows' % len(i_factors))

# ============================================================
# B组: 质量因子 (ROE, margins, log_eps)
# ============================================================
print('[B] Quality factors...')

b_factors = con.execute("""
WITH fin AS (
    SELECT
        CASE WHEN ts_code LIKE '%.SH' THEN 'sh' || SUBSTR(ts_code, 1, 6)
             WHEN ts_code LIKE '%.SZ' THEN 'sz' || SUBSTR(ts_code, 1, 6)
             ELSE ts_code END AS ts_code,
        report_date, roe, gross_margin, net_margin, net_profit_margin, eps
    FROM financial_statements WHERE report_date >= '2001-01-01'
),
kline AS (
    SELECT ts_code, trade_date FROM kline_daily WHERE trade_date >= '2002-01-01'
),
joined AS (
    SELECT k.ts_code, k.trade_date,
           f.roe, f.gross_margin, f.net_margin, f.net_profit_margin, f.eps,
           ROW_NUMBER() OVER(PARTITION BY k.ts_code, k.trade_date ORDER BY f.report_date DESC) AS rn
    FROM kline k
    LEFT JOIN fin f ON k.ts_code = f.ts_code
        AND f.report_date <= k.trade_date
        AND f.report_date >= k.trade_date - INTERVAL 270 DAY
)
SELECT ts_code, trade_date, roe, gross_margin, net_margin,
       net_profit_margin AS profit_margin, eps
FROM joined WHERE rn = 1
""").df()

b_factors['log_eps'] = np.log(b_factors['eps'].clip(lower=0.001))
b_factors = b_factors.drop(columns=['eps'])
print('  B-factors: %d rows' % len(b_factors))

# ============================================================
# H组: 技术因子 (RSI, BOLL, MA, volume)
# ============================================================
print('[H] Technical factors...')

h_factors = con.execute("""
WITH prices AS (
    SELECT ts_code, trade_date, close, open, high, low, vol,
           LAG(close, 1) OVER w AS c1, LAG(close, 5) OVER w AS c5,
           LAG(close, 20) OVER w AS c20, LAG(close, 60) OVER w AS c60,
           LAG(close, 120) OVER w AS c120,
           AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS ma20,
           AVG(vol) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS vol_ma20,
           AVG(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 5 PRECEDING) AS ma5,
           STDDEV(close) OVER (PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS std20
    FROM kline_daily WHERE trade_date >= '2002-01-01'
    WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
),
rsi_calc AS (
    SELECT ts_code, trade_date, close, open, high, low, vol, c1, c5, c20, c60, c120, ma20, vol_ma20, ma5, std20,
           -- RSI6
           AVG(CASE WHEN close > c1 THEN close - c1 ELSE 0 END) OVER w6 /
           NULLIF(AVG(ABS(close - c1)) OVER w6, 0) AS rsi6_raw,
           -- RSI14
           AVG(CASE WHEN close > c1 THEN close - c1 ELSE 0 END) OVER w14 /
           NULLIF(AVG(ABS(close - c1)) OVER w14, 0) AS rsi14_raw
    FROM prices
    WINDOW w6 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 5 PRECEDING),
           w14 AS (PARTITION BY ts_code ORDER BY trade_date ROWS 13 PRECEDING)
)
SELECT ts_code, trade_date,
       100 * rsi6_raw / (1 + rsi6_raw) AS rsi6,
       100 * rsi14_raw / (1 + rsi14_raw) AS rsi14,
       -- Bollinger position
       (close - ma20) / NULLIF(std20, 0) AS boll_pos,
       std20 / NULLIF(ma20, 0) AS boll_width,
       -- MA divergences
       close / NULLIF(c20, 0) - 1 AS div_ma20,
       close / NULLIF(c60, 0) - 1 AS div_ma60,
       close / NULLIF(c120, 0) - 1 AS div_ma120,
       -- Volume ratio
       vol / NULLIF(vol_ma20, 0) AS vol_ratio,
       -- MA alignment score (5>20>60 = 3, etc.)
       (CASE WHEN ma5 > ma20 THEN 1 ELSE 0 END +
        CASE WHEN ma20 > c60 THEN 1 ELSE 0 END +
        CASE WHEN c60 > c120 THEN 1 ELSE 0 END) AS ma_score
FROM rsi_calc
WHERE c1 IS NOT NULL
""").df()

# RSI extreme
h_factors['rsi_extreme'] = np.where(h_factors['rsi6'] > 70, 1,
                           np.where(h_factors['rsi6'] < 30, -1, 0))
print('  H-factors: %d rows' % len(h_factors))

# ============================================================
# C组: 判断因子 (margin panic, streak, northbound)
# ============================================================
print('[C] Judgment factors...')

# Margin trading data (starts ~2010)
c_factors = con.execute("""
WITH margin AS (
    SELECT trade_date,
           margin_balance,
           margin_balance - LAG(margin_balance, 20) OVER(ORDER BY trade_date) AS mg_chg_20d,
           (margin_balance / NULLIF(LAG(margin_balance, 60) OVER(ORDER BY trade_date), 0) - 1) AS mg_chg_60d
    FROM margin_trading WHERE trade_date >= '2010-01-01'
),
prices AS (
    SELECT ts_code, trade_date, close,
           close / LAG(close, 1) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret_1d,
           LAG(close, 5) OVER(PARTITION BY ts_code ORDER BY trade_date) AS c5
    FROM kline_daily WHERE trade_date >= '2010-01-01'
),
streaks AS (
    SELECT ts_code, trade_date, ret_1d,
           SUM(CASE WHEN ret_1d < 0 THEN 1 ELSE 0 END) OVER(
               PARTITION BY ts_code, grp ORDER BY trade_date) AS streak_dn
    FROM (
        SELECT *, SUM(CASE WHEN ret_1d >= 0 THEN 1 ELSE 0 END) OVER(
            PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS grp
        FROM prices
    )
)
SELECT p.ts_code, p.trade_date,
       -- Margin panic: margin down >5% over 20d AND price down over 5d
       CASE WHEN m.mg_chg_20d / NULLIF(LAG(m.margin_balance, 20) OVER(ORDER BY m.trade_date), 0) < -0.05
             AND p.close < p.c5 THEN 1 ELSE 0 END AS margin_panic,
       -- 5-day losing streak
       s.streak_dn AS streak5_dn
FROM prices p
LEFT JOIN margin m ON p.trade_date = m.trade_date
LEFT JOIN streaks s ON p.ts_code = s.ts_code AND p.trade_date = s.trade_date
""").df()

c_factors['streak5_dn'] = c_factors['streak5_dn'].fillna(0)
c_factors['margin_panic'] = c_factors['margin_panic'].fillna(0)
print('  C-factors: %d rows' % len(c_factors))

con.close()

# ============================================================
# Merge all factor groups
# ============================================================
print('\n[M] Merging all factor groups...')

df = i_factors.merge(b_factors, on=['ts_code', 'trade_date'], how='outer')
df = df.merge(h_factors, on=['ts_code', 'trade_date'], how='outer')
df = df.merge(c_factors, on=['ts_code', 'trade_date'], how='outer')

# Fill missing C-factor values (pre-2010)
for col in ['margin_panic', 'streak5_dn']:
    if col in df.columns:
        df[col] = df[col].fillna(0)

print('  Total: %d rows, %d stocks' % (len(df), df['ts_code'].nunique()))
print('  Date range: %s ~ %s' % (df['trade_date'].min(), df['trade_date'].max()))

# Save
df.to_parquet(OUT)
print('\nSaved: %s (%.1f MB)' % (OUT, __import__('os').path.getsize(OUT)/1024/1024))
print('Done.')
