# -*- coding: utf-8 -*-
"""暴跌止损检查器 · 每日跑
持仓股跌>30%→标记清仓, 下月调仓时排除
"""
import duckdb, pandas as pd, numpy as np, json, sys
from datetime import date, datetime

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
PORTFOLIO_FILE = 'D:/AgentQuant/our/paper_portfolio_xiaozhong.json'
CRASH_STOP = -0.30  # 30%止损线

con = duckdb.connect(DB, read_only=True)

# 读持仓
try:
    with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
        pf = json.load(f)
    positions = pf.get('positions', [])
    entry_date = pf.get('date', '?')
except:
    print('无持仓文件')
    con.close()
    sys.exit(0)

if not positions:
    print('空仓')
    con.close()
    sys.exit(0)

# 取最新交易日
today = str(con.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0])
print(f'止损检查 · 持仓{len(positions)}只 · {entry_date}→{today}')

# 检查每只持仓
stopped = []
ok = []
for pos in positions:
    code = pos['code']
    entry_price = pos['buy_price']
    shares = pos['shares']

    # 当前价
    r = con.execute(f"""
        SELECT close FROM kline_daily WHERE ts_code='{code}'
        AND trade_date <= DATE '{today}' ORDER BY trade_date DESC LIMIT 1
    """).fetchone()
    if not r:
        print(f'  {code} {pos.get("name","?")}: 无数据')
        continue

    current_price = float(r[0])
    pnl = (current_price / entry_price - 1) * 100

    # 退市检查
    delisted = con.execute(f"""
        SELECT delist_date FROM stock_basic WHERE ts_code='{code}' AND delist_date IS NOT NULL
    """).fetchone()
    # ST检查
    is_st = con.execute(f"""
        SELECT is_st FROM stock_basic WHERE ts_code='{code}' AND is_st=true
    """).fetchone()

    flags = []
    if pnl < CRASH_STOP * 100:
        flags.append(f'!!暴跌{pnl:+.1f}%!!')
        stopped.append(pos)
    if is_st:
        flags.append('ST!')
        stopped.append(pos)
    if delisted:
        flags.append(f'退市({delisted[0]})')
        stopped.append(pos)

    flag_str = ' '.join(flags) if flags else 'OK'
    color = '🔴' if flags else '🟢'
    print(f'  {color} {code} {pos.get("name","?"):<8s} 入场{entry_price:.2f}→现{current_price:.2f} ({pnl:+.1f}%) {flag_str}')

con.close()

# 输出
print(f'\n正常: {len(ok)}只 | 需止损: {len(stopped)}只')
if stopped:
    print('止损清单:')
    total_loss = 0
    for pos in stopped:
        code = pos['code']
        entry = pos['buy_price']
        shares = pos['shares']
        # 取现价
        con2 = duckdb.connect(DB, read_only=True)
        r2 = con2.execute(f"SELECT close FROM kline_daily WHERE ts_code='{code}' AND trade_date<=DATE '{today}' ORDER BY trade_date DESC LIMIT 1").fetchone()
        con2.close()
        if r2:
            current = float(r2[0])
            loss = (current - entry) * shares
            total_loss += loss
            print(f'  {code} {pos.get("name","?")} 亏损{loss:+.0f}元')
    print(f'  预估总亏损: {total_loss:+.0f}元')

# 更新portfolio标注止损
if stopped:
    pf['stop_loss'] = {
        'date': today,
        'count': len(stopped),
        'codes': [p['code'] for p in stopped]
    }
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)
    print('已更新portfolio止损标记')
