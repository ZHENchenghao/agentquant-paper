# -*- coding: utf-8 -*-
"""批量回填+财务因子回测"""
import sys, os, time
sys.path.insert(0, 'D:/AgentQuant/our')
from financial_backfill import backfill
import duckdb

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

# 取有K线数据的非ST非银行股票
c = duckdb.connect(DB, read_only=True)
stocks = c.execute("""
    SELECT DISTINCT REPLACE(REPLACE(k.ts_code, '.SH',''), '.SZ','') code
    FROM kline_daily k
    JOIN stock_basic s ON k.ts_code = s.ts_code
    WHERE k.trade_date >= '2026-06-01'
      AND s.is_st = 0
      AND k.ts_code NOT LIKE '601%'  -- 排除银行
      AND k.ts_code NOT LIKE '600015%'
      AND k.ts_code NOT LIKE '600016%'
      AND k.ts_code NOT LIKE '600036%'
    ORDER BY k.ts_code
    LIMIT 60
""").fetchall()
c.close()

codes = [x[0] for x in stocks]
print(f'Backfilling {len(codes)} stocks...')

# 分批回填(每批5只, 间隔1秒避免被封)
ok = 0
for i, code in enumerate(codes):
    r = backfill(code)
    if 'ERR' not in r and '无数据' not in r:
        ok += 1
    if (i+1) % 10 == 0:
        print(f'  [{i+1}/{len(codes)}] {ok} OK')
    time.sleep(1.0)  # 新浪限流

print(f'\nDone: {ok}/{len(codes)} stocks backfilled')

# 验证覆盖
c = duckdb.connect(DB, read_only=True)
n_ocf = c.execute('SELECT COUNT(DISTINCT ts_code) FROM financial_statements WHERE operating_cf IS NOT NULL').fetchone()[0]
n_ar = c.execute('SELECT COUNT(DISTINCT ts_code) FROM financial_statements WHERE accounts_receivable IS NOT NULL').fetchone()[0]
n_gw = c.execute('SELECT COUNT(*) FROM goodwill_detail').fetchone()[0]
c.close()
print(f'Coverage: OCF={n_ocf} AR={n_ar} GW={n_gw}')
