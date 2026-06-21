# -*- coding: utf-8 -*-
"""
Baostock全量K线补入 — 填6/11-6/18缺口
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, baostock as bs, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
START = '2026-06-11'; END = '2026-06-18'

print("=" * 60)
print(f"Baostock补入 {START}~{END}")
print("=" * 60)

# 1. 登录+取代码
bs.login()
rs = bs.query_stock_basic()
all_codes = []
while (rs.error_code == '0') & rs.next():
    row = rs.get_row_data()
    code = row[0]  # sh.600000 format
    if code.startswith('sh.6') or code.startswith('sz.0') or code.startswith('sz.3') or code.startswith('sz.2'):
        all_codes.append(code)
print(f"[1] A股: {len(all_codes)}只")

# 2. 检查DuckDB缺失情况
con = duckdb.connect(DB)
need_update = set()
for bs_code in all_codes:
    ts_code = bs_code.replace('sh.', '').replace('sz.', '')
    ts_code = ts_code + ('.SH' if bs_code.startswith('sh') else '.SZ')
    cnt = con.execute(f"SELECT COUNT(*) FROM kline_daily WHERE ts_code='{ts_code}' AND trade_date='{END}'").fetchone()[0]
    if cnt == 0:
        need_update.add((bs_code, ts_code))

print(f"[2] 需补: {len(need_update)}/{len(all_codes)}只")

# 3. 批量获取+插入
batch_size = 100; new_rows = 0
codes_list = list(need_update)

for i in range(0, len(codes_list), batch_size):
    batch = codes_list[i:i+batch_size]
    # 每只单独查询
    for bs_code, ts_code in batch:
        try:
            rs2 = bs.query_history_k_data_plus(bs_code,
                'date,open,high,low,close,preclose,volume,amount,turn,pctChg',
                start_date=START, end_date=END, frequency='d', adjustflag='2')
            if rs2.error_code != '0': continue

            rows = []
            while rs2.next(): rows.append(rs2.get_row_data())

            for row in rows:
                dt, op, hi, lo, cl, pc, vol, amt, turn, pct = row
                if cl == '' or float(cl) <= 0: continue
                op_v = float(op) if op else None
                hi_v = float(hi) if hi else None
                lo_v = float(lo) if lo else None
                cl_v = float(cl)
                pc_v = float(pc) if pc else None
                vol_v = float(vol) if vol else None
                amt_v = float(amt) if amt else None
                turn_v = float(turn) if turn else None
                pct_v = float(pct) if pct else None

                con.execute(f"""
                    INSERT OR REPLACE INTO kline_daily
                    (ts_code, trade_date, open, high, low, close, pre_close, change_pct, vol, amount, turnover_rate)
                    VALUES ('{ts_code}', '{dt}', {op_v or 'NULL'}, {hi_v or 'NULL'}, {lo_v or 'NULL'},
                            {cl_v}, {pc_v or 'NULL'}, {pct_v or 'NULL'},
                            {vol_v or 'NULL'}, {amt_v or 'NULL'}, {turn_v or 'NULL'})
                """)
                new_rows += 1
        except Exception as e:
            continue

    if (i // batch_size) % 50 == 0 and i > 0:
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (len(codes_list) - i) / rate / 60 if rate > 0 else 0
        print(f"  进度: {i}/{len(codes_list)} ({i/len(codes_list)*100:.0f}%) | {rate:.0f}只/秒 | ETA {eta:.0f}分 | +{new_rows}条")

bs.logout()
con.close()

print(f"\n总补入: {new_rows}条 | 耗时: {(time.time()-t0)/60:.1f}分")
