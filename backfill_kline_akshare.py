# -*- coding: utf-8 -*-
"""
AKShare全量K线补入DuckDB — 解决Sina API只返回293只的问题
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, akshare as ak, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
DAYS = 15  # 补最近15个交易日

print("=" * 60)
print("AKShare全量K线补入")
print("=" * 60)

# 1. 获取全A股列表
print("[1] 获取A股列表...")
stock_list = ak.stock_zh_a_spot_em()
print(f"  全A股: {len(stock_list)}只")

# 只保留股票代码
codes = stock_list['代码'].tolist()

# 2. 获取交易日历
print("[2] 获取交易日历...")
cal = ak.tool_trade_date_hist_sina()
cal_dates = sorted(cal['trade_date'].astype(str).values)
target_dates = cal_dates[-DAYS:]
print(f"  目标日期: {target_dates[0]} ~ {target_dates[-1]} ({len(target_dates)}天)")

# 3. 检查DuckDB中已有多少
con = duckdb.connect(DB)
existing = con.execute(f"""
    SELECT COUNT(DISTINCT ts_code) FROM kline_daily
    WHERE trade_date >= '{target_dates[0]}'
""").fetchone()[0]
print(f"  DuckDB现有: {existing}只")

# 4. 逐日补入
total_new = 0
for i, td in enumerate(target_dates):
    # 检查当日已有数量
    cnt = con.execute(f"SELECT COUNT(*) FROM kline_daily WHERE trade_date = '{td}'").fetchone()[0]
    if cnt >= 4500:
        print(f"  [{i+1}/{len(target_dates)}] {td}: 已有{cnt}只, 跳过")
        continue

    print(f"  [{i+1}/{len(target_dates)}] {td}: 已有{cnt}只, 采集...", end=' ', flush=True)

    try:
        # AKShare获取当日全市场行情
        df = ak.stock_zh_a_hist(symbol="", period="daily", start_date=td, end_date=td, adjust="")
        # 注意: stock_zh_a_hist对空symbol可能不支持全市场, 用循环
    except:
        pass

    # 改用逐只获取(抽样1000只加速)
    batch_size = 500
    new_rows = 0
    sample_codes = codes  # 全量, 但分批

    for j in range(0, len(sample_codes), batch_size):
        batch = sample_codes[j:j+batch_size]
        # 批量检查哪些还缺
        code_str = "','".join(batch)
        missing = set(batch) - set(
            con.execute(f"SELECT ts_code FROM kline_daily WHERE trade_date='{td}' AND ts_code IN ('{code_str}')").fetchdf()['ts_code'].tolist()
        )
        if not missing:
            continue

        for code in list(missing)[:200]:  # 每批最多200只(避免API限流)
            try:
                hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=td, end_date=td, adjust="")
                if hist is None or len(hist) == 0:
                    continue
                row = hist.iloc[-1]
                # 清洗: 处理NaN
                open_p = float(row['开盘']) if pd.notna(row.get('开盘')) else None
                high_p = float(row['最高']) if pd.notna(row.get('最高')) else None
                low_p = float(row['最低']) if pd.notna(row.get('最低')) else None
                close_p = float(row['收盘']) if pd.notna(row.get('收盘')) else None
                vol_p = float(row['成交量']) if pd.notna(row.get('成交量')) else None
                amt_p = float(row['成交额']) if pd.notna(row.get('成交额')) else None
                pct_p = float(row['涨跌幅']) if pd.notna(row.get('涨跌幅')) else None
                tr_p = float(row['换手率']) if pd.notna(row.get('换手率')) else None

                if close_p is None or close_p <= 0:
                    continue

                # INSERT OR REPLACE
                ts_code = code if '.' in code else (code[:6] + ('.SH' if code.startswith('6') else '.SZ'))
                con.execute(f"""
                    INSERT OR REPLACE INTO kline_daily
                    (ts_code, trade_date, open, high, low, close, pre_close, change_pct, vol, amount, turnover_rate)
                    VALUES ('{ts_code}', '{td}', {open_p or 'NULL'}, {high_p or 'NULL'}, {low_p or 'NULL'},
                            {close_p}, {close_p/(1+pct_p/100) if pct_p else 'NULL'},
                            {pct_p or 'NULL'}, {vol_p or 'NULL'}, {amt_p or 'NULL'}, {tr_p or 'NULL'})
                """)
                new_rows += 1
            except Exception as e:
                continue
            time.sleep(0.02)  # 温和限速

    print(f"+{new_rows}条")
    total_new += new_rows

con.close()
print(f"\n总补入: {total_new}条 | 耗时: {time.time()-t0:.0f}s")
