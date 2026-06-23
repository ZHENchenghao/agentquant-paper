# -*- coding: utf-8 -*-
"""
圆桌会议 纸交执行器 v3.0 — 双模版 (去天眼依赖)
==============================================
paper   : 本地JSON纸交, DuckDB用于价格+涨跌停检查, 价格缺失→前溯
myquant : 掘金仿真, 完全不依赖DuckDB, 价格=计划价格, 撮合=掘金引擎

用法:
  python paper_executor.py                          # 纸交模式
  python paper_executor.py --mode myquant           # 掘金仿真模式
  python paper_executor.py --mode myquant --token xxx --account xxx
"""
import json, os, sys, io, time, warnings, argparse
from datetime import date, datetime, timedelta
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = date.today()
CAPITAL = 100000
CRASH_STOP = -0.30

# ═══════════ CLI ═══════════
p = argparse.ArgumentParser(description='圆桌会议 纸交执行器 v3.0')
p.add_argument('--mode', default='paper', choices=['paper', 'myquant'])
p.add_argument('--token', default=os.environ.get('MYQUANT_TOKEN', 'd1f2101ac1bb2e9380acf2c91b0dfc8e85369cdf'))
p.add_argument('--account', default=os.environ.get('MYQUANT_ACCOUNT', '15d64470-6d88-11f1-932c-00163e022aa6'))
args = p.parse_args()

print(f'=== 圆桌会议 纸交执行器 v3.0 [{args.mode.upper()}] {TODAY} ===')

# ═══════════ 1. 初始化 ═══════════
from myquant_bridge import TradingBridge

if args.mode == 'myquant':
    bridge = TradingBridge(mode='myquant', token=args.token, account_id=args.account, capital=CAPITAL)
    print(f'掘金仿真已连接')
else:
    bridge = TradingBridge(mode='paper', paper_dir=DIR, capital=CAPITAL)
    print(f'纸交模式: {bridge._paper_file}')

# ═══════════ 2. 找到当日批次 ═══════════
exec_files = sorted([f for f in os.listdir(DIR) if f.startswith('execution_plan_') and f.endswith('.json')],
                    reverse=True)
if not exec_files:
    print('无执行计划文件, 先跑 daily_runner.py')
    sys.exit(0)

exec_file = exec_files[0]
with open(os.path.join(DIR, exec_file), 'r', encoding='utf-8') as f:
    plan = json.load(f)
print(f'执行计划: {exec_file} ({plan["date"]})')

plan_date = datetime.strptime(plan['date'], '%Y-%m-%d').date()
days_diff = (TODAY - plan_date).days
if days_diff <= 0:
    print(f'计划日={plan_date}, 今日={TODAY}, 尚未到执行日')
    sys.exit(0)

trading_days_passed = 0
d = plan_date + timedelta(days=1)
while d <= TODAY:
    if d.weekday() < 5:
        trading_days_passed += 1
    d += timedelta(days=1)

target_day_label = f'T+{trading_days_passed}'
if trading_days_passed > 3:
    print(f'计划已过期 (T+{trading_days_passed} > T+3), 跳过')
    sys.exit(0)

print(f'信号日: {plan_date} | 今日: {TODAY} | 执行: {target_day_label}')

# ═══════════ 3. 筛选今日批次 ═══════════
all_batches = plan.get('execution_plan', [])
today_batches = [b for b in all_batches if b['Execution_Day'] == target_day_label]

if not today_batches:
    print(f'{target_day_label} 无待执行批次')
    sys.exit(0)

print(f'待执行: {len(today_batches)}笔')

# ═══════════ 4. 价格查找 ═══════════
# paper模式: DuckDB + 前溯回退
# myquant模式: 直接用计划价格, 掘金撮合引擎自己匹配
price_map = {}
names_map = {}
limit_map = {}

if args.mode == 'paper':
    import duckdb
    DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
    con = duckdb.connect(DB, read_only=True)
    latest_kline = str(con.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0])

    codes_needed = list(set(b['Ticker'] for b in today_batches))
    ts_codes = []
    for c in codes_needed:
        if c.startswith('sh'): ts_codes.append(f'{c[2:]}.SH')
        elif c.startswith('sz'): ts_codes.append(f'{c[2:]}.SZ')
        else: ts_codes.append(c)

    for ts in ts_codes:
        ticker = ts.split('.')[1].lower() + ts.split('.')[0]
        # 前溯查找: 从最新日期往回找, 最多5天
        for offset in range(5):
            r = con.execute(f"""
                SELECT close, pre_close
                FROM kline_daily
                WHERE ts_code = ? AND trade_date <= DATE '{latest_kline}' - INTERVAL {offset} DAY
                ORDER BY trade_date DESC LIMIT 1
            """, [ts]).fetchone()
            if r and r[0] and r[0] > 0:
                price_map[ticker] = float(r[0])
                # 涨跌停
                if r[1] and r[1] > 0:
                    chg = r[0] / r[1] - 1
                    if chg >= 0.095:
                        limit_map[ticker] = 'LIMIT_UP'
                    elif chg <= -0.095:
                        limit_map[ticker] = 'LIMIT_DOWN'
                break

    # 股票名称
    try:
        names_df = con.execute("SELECT ts_code, name FROM stock_basic").df()
        def _norm(c):
            c = str(c).strip()
            if '.' in c: return c.split('.')[1].lower() + c.split('.')[0]
            return c.lower()
        for _, row in names_df.iterrows():
            t = _norm(row['ts_code'])
            names_map[t] = str(row['name'])
    except Exception:
        pass

    con.close()
    print(f'价格: {len(price_map)}/{len(today_batches)}只有数据 (K线={latest_kline})')

else:
    # myquant模式: 用计划中的价格, 不查DuckDB
    for batch in today_batches:
        ticker = batch['Ticker']
        price_map[ticker] = batch.get('Limit_Price', 0)
        names_map[ticker] = batch.get('Name', ticker)
    print(f'掘金模式: {len(today_batches)}笔, 价格=计划价, 撮合=掘金引擎')

# ═══════════ 5. 执行交易 ═══════════
trades_today = []
alerts = []

for batch in today_batches:
    ticker = batch['Ticker']
    action = batch['Action']
    shares = batch['Batch_Shares']
    name = names_map.get(ticker, ticker)
    limit = limit_map.get(ticker, 'NORMAL')

    # 价格来源: price_map有就用, 没有就用计划价格
    fill_price = price_map.get(ticker, batch.get('Limit_Price', 0))
    if fill_price <= 0:
        alerts.append(f'[NO_PRICE] {ticker}: 无可用价格, 跳过')
        continue

    # 涨跌停阻断 (paper模式)
    if args.mode == 'paper':
        if action == 'BUY' and limit == 'LIMIT_UP':
            alerts.append(f'[LIMIT_UP_BLOCK] {ticker}: 涨停, 买入顺延')
            continue
        if action == 'SELL' and limit == 'LIMIT_DOWN':
            alerts.append(f'[LIMIT_DOWN_ALERT] {ticker}: 跌停, 无法卖出!')
            continue

    # 下单
    if action == 'BUY':
        result = bridge.buy(ticker, fill_price, shares, name)
        if result['success']:
            actual_shares = result.get('shares', shares)
            cost = result.get('cost', actual_shares * fill_price * 1.00033)
            trades_today.append(f'BUY  {name}({ticker}) {actual_shares}股 @{fill_price:.2f}')
        else:
            alerts.append(f'[BUY_FAIL] {ticker}: {result.get("error", "unknown")}')

    elif action == 'SELL':
        result = bridge.sell(ticker, fill_price, shares, name)
        if result['success']:
            actual_shares = result.get('shares', shares)
            revenue = result.get('revenue', actual_shares * fill_price * 0.99967)
            trades_today.append(f'SELL {name}({ticker}) {actual_shares}股 @{fill_price:.2f}')
        else:
            alerts.append(f'[SELL_FAIL] {ticker}: {result.get("error", "unknown")}')

# ═══════════ 6. 暴跌止损检查 ═══════════
positions = bridge.get_positions()
stopped = []

if args.mode == 'paper' and price_map:
    for pos in positions:
        code = pos['code']
        if code in price_map:
            current_price = price_map[code]
            pnl = current_price / pos['buy_price'] - 1
            if pnl < CRASH_STOP:
                stopped.append({'code': code, 'name': pos.get('name', ''), 'pnl': pnl,
                              'buy_price': pos['buy_price']})

elif args.mode == 'myquant':
    # 掘金持仓自带 last_price
    for pos in positions:
        current_price = pos.get('price', pos.get('buy_price', 0))
        buy_price = pos.get('buy_price', current_price)
        if buy_price > 0 and current_price > 0:
            pnl = current_price / buy_price - 1
            if pnl < CRASH_STOP:
                stopped.append({'code': pos['code'], 'name': pos.get('name', ''), 'pnl': pnl,
                              'buy_price': buy_price})

# 标记
if stopped:
    for s in stopped:
        alerts.append(f'[CRASH_STOP] {s["name"]}({s["code"]}) 亏损{s["pnl"]*100:.1f}% 触发止损!')

# 执行止损
for s in stopped:
    cp = price_map.get(s['code'], s['buy_price'])
    pos = next((p for p in positions if p['code'] == s['code']), None)
    if pos:
        result = bridge.sell(s['code'], cp, pos.get('shares', 0), s.get('name', ''))
        if result['success']:
            actual_shares = result.get('shares', pos.get('shares', 0))
            trades_today.append(f'STOP {s["name"]}({s["code"]}) {actual_shares}股 @{cp:.2f} 止损!')

# 刷新
positions = bridge.get_positions()

# ═══════════ 7. 输出 ═══════════
print(f'\n--- 执行结果 [{args.mode.upper()}] ---')
for t in trades_today:
    print(f'  {t}')

if alerts:
    print(f'\n[告警]')
    for a in alerts:
        print(f'  !! {a}')

status = bridge.get_status()
print(f'\n持仓: {status["positions_count"]}只 | 总资产: {status["total_value"]:.0f} | 现金: {status["cash"]:.0f}')
if status.get('pnl') is not None:
    print(f'累计盈亏: {status["pnl"]:+.0f} ({status["pnl"]/CAPITAL*100:+.2f}%)')

if args.mode == 'paper':
    print(f'\n已保存: {bridge._paper_file}')

print('Done.')
