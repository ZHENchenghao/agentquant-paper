# -*- coding: utf-8 -*-
"""自动纸交执行器 v1.0
- 读取当日到期的执行计划批次
- 模拟成交(用当日收盘价)
- 更新paper持仓
- 检查暴跌止损
"""
import duckdb, pandas as pd, numpy as np, json, os, sys, io, time, warnings
from datetime import date, datetime, timedelta
warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = date.today()
CAPITAL = 100000
CRASH_STOP = -0.30

print(f'=== 纸交执行器 {TODAY} ===')

# ═══════════ 1. 找到当日需执行的批次 ═══════════
con = duckdb.connect(DB, read_only=True)
latest_kline = str(con.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0])
con.close()
print(f'最新K线: {latest_kline}')

# 找最近的执行计划文件
exec_files = sorted([f for f in os.listdir(DIR) if f.startswith('execution_plan_') and f.endswith('.json')],
                    reverse=True)
if not exec_files:
    print('无执行计划文件, 先跑 daily_runner.py')
    sys.exit(0)

exec_file = exec_files[0]
with open(os.path.join(DIR, exec_file), 'r', encoding='utf-8') as f:
    plan = json.load(f)
print(f'执行计划: {exec_file} ({plan["date"]})')

# 确定今天是T+几
plan_date = datetime.strptime(plan['date'], '%Y-%m-%d').date()
# 计算交易日偏移: 从plan_date到今天之间有多少个交易日不等于0
# 简化: 直接用自然日近似, 跳过周末
days_diff = (TODAY - plan_date).days
if days_diff <= 0:
    print(f'计划日={plan_date}, 今日={TODAY}, 尚未到执行日')
    sys.exit(0)

# T+1 = 下一个交易日, T+2, T+3
# 简化: 考虑周末
trading_days_passed = 0
d = plan_date + timedelta(days=1)
while d <= TODAY:
    if d.weekday() < 5:  # 周一到周五
        trading_days_passed += 1
    d += timedelta(days=1)

target_day_label = f'T+{trading_days_passed}'
if trading_days_passed > 3:
    print(f'计划已过期 (T+{trading_days_passed} > T+3), 跳过')
    sys.exit(0)

print(f'信号日: {plan_date} | 今日: {TODAY} | 执行: {target_day_label}')

# ═══════════ 2. 筛选今日应执行的批次 ═══════════
all_batches = plan.get('execution_plan', [])
today_batches = [b for b in all_batches if b['Execution_Day'] == target_day_label]

if not today_batches:
    print(f'{target_day_label} 无待执行批次')
    sys.exit(0)

print(f'待执行: {len(today_batches)}笔')

# ═══════════ 3. 获取当日收盘价模拟成交 ═══════════
con = duckdb.connect(DB, read_only=True)
codes_needed = list(set(b['Ticker'] for b in today_batches))
# 转ts_code格式
ts_codes = []
for c in codes_needed:
    if c.startswith('sh'): ts_codes.append(f'{c[2:]}.SH')
    elif c.startswith('sz'): ts_codes.append(f'{c[2:]}.SZ')
    else: ts_codes.append(c)

codes_str = ','.join([f"'{c}'" for c in ts_codes])
prices_today = con.execute(f"""
    SELECT ts_code, close, pre_close,
           close/pre_close-1 AS change_pct,
           CASE WHEN close/pre_close >= 1.095 THEN 'LIMIT_UP'
                WHEN close/pre_close <= 0.905 THEN 'LIMIT_DOWN'
                ELSE 'NORMAL' END AS limit_status
    FROM kline_daily
    WHERE trade_date = '{latest_kline}'
      AND ts_code IN ({codes_str})
""").df()

# 加载股票名称
names_df = con.execute("SELECT ts_code, name FROM stock_basic").df()
con.close()

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

prices_today['ticker'] = prices_today['ts_code'].apply(norm)
price_map = prices_today.set_index('ticker')

# ═══════════ 4. 模拟执行 ═══════════
pf_file = os.path.join(DIR, 'paper_portfolio_xiaozhong.json')
if os.path.exists(pf_file):
    with open(pf_file, 'r', encoding='utf-8') as f:
        portfolio = json.load(f)
else:
    portfolio = {'strategy': '小众战法_vFinal_8F', 'date': str(TODAY), 'capital': CAPITAL,
                 'positions': [], 'cash': CAPITAL, 'invested': 0, 'trades': []}

positions = {p['code']: p for p in portfolio.get('positions', [])}
cash = portfolio.get('cash', CAPITAL)
trades_today = []
alerts = []

for batch in today_batches:
    ticker = batch['Ticker']
    action = batch['Action']
    shares = batch['Batch_Shares']

    if ticker not in price_map.index:
        alerts.append(f'[NO_PRICE] {ticker}: 无当日行情, 跳过')
        continue

    price_row = price_map.loc[ticker]
    fill_price = float(price_row['close'])
    limit = price_row.get('limit_status', 'NORMAL')

    # 🔴 涨跌停阻断
    if action == 'BUY' and limit == 'LIMIT_UP':
        alerts.append(f'[LIMIT_UP_BLOCK] {ticker}: 涨停, 买入顺延')
        continue
    if action == 'SELL' and limit == 'LIMIT_DOWN':
        alerts.append(f'[LIMIT_DOWN_ALERT] {ticker}: 跌停, 无法卖出!')
        continue

    name = names_df[names_df['ts_code'].apply(norm)==ticker]['name'].values
    name = name[0] if len(name) > 0 else ticker

    if action == 'BUY':
        cost = shares * fill_price * 1.00033  # 含手续费
        if cost > cash:
            actual_shares = int(cash / (fill_price * 1.00033) / 100) * 100
            if actual_shares == 0:
                alerts.append(f'[CASH_SHORT] {ticker}: 现金不足, 跳过')
                continue
            shares = actual_shares
            cost = shares * fill_price * 1.00033

        cash -= cost
        if ticker in positions:
            old = positions[ticker]
            total_shares = old['shares'] + shares
            avg_price = (old['buy_price'] * old['shares'] + fill_price * shares) / total_shares
            positions[ticker] = {'code': ticker, 'name': name, 'shares': total_shares,
                                 'buy_price': round(avg_price, 3), 'price': float(fill_price),
                                 'cost': old.get('cost', 0) + cost}
        else:
            positions[ticker] = {'code': ticker, 'name': name, 'shares': shares,
                                 'buy_price': float(fill_price), 'price': float(fill_price),
                                 'cost': float(cost)}
        trades_today.append(f'BUY  {name}({ticker}) {shares}股 @{fill_price:.2f} 花费{cost:.0f}')

    elif action == 'SELL':
        if ticker not in positions:
            alerts.append(f'[NO_POS] {ticker}: 无持仓可卖')
            continue
        pos = positions[ticker]
        actual_shares = min(shares, pos['shares'])
        revenue = actual_shares * fill_price * 0.99967  # 扣除手续费
        cash += revenue
        remaining = pos['shares'] - actual_shares
        if remaining > 0:
            positions[ticker]['shares'] = remaining
        else:
            del positions[ticker]
        trades_today.append(f'SELL {name}({ticker}) {actual_shares}股 @{fill_price:.2f} 回笼{revenue:.0f}')

# ═══════════ 5. 暴跌止损检查 ═══════════
stopped = []
for code, pos in positions.items():
    if code in price_map.index:
        current_price = float(price_map.loc[code, 'close'])
        pnl = current_price / pos['buy_price'] - 1
        if pnl < CRASH_STOP:
            stopped.append({'code': code, 'name': pos['name'], 'pnl': pnl, 'buy_price': pos['buy_price']})
            alerts.append(f'[CRASH_STOP] {pos["name"]}({code}) 亏损{pnl*100:.1f}% 触发止损!')

# 执行止损卖出
for s in stopped:
    code = s['code']
    if code in positions:
        current_price = float(price_map.loc[code, 'close'])
        pos = positions[code]
        revenue = pos['shares'] * current_price * 0.99967
        cash += revenue
        trades_today.append(f'STOP {pos["name"]}({code}) {pos["shares"]}股 @{current_price:.2f} 止损!')
        del positions[code]

# ═══════════ 6. 更新持仓市值 ═══════════
total_market_value = 0
for code, pos in positions.items():
    if code in price_map.index:
        pos['price'] = float(price_map.loc[code, 'close'])
        total_market_value += pos['shares'] * pos['price']

# ═══════════ 7. 保存 ═══════════
portfolio = {
    'strategy': '小众战法_vFinal_8F',
    'date': str(TODAY),
    'capital': CAPITAL,
    'cash': round(cash, 2),
    'invested': round(total_market_value, 2),
    'total_value': round(cash + total_market_value, 2),
    'pnl_total': round(cash + total_market_value - CAPITAL, 2),
    'pnl_pct': round((cash + total_market_value) / CAPITAL - 1, 4),
    'positions': list(positions.values()),
    'trades_today': trades_today,
    'alerts': alerts,
    'stop_loss_triggered': len(stopped) > 0
}

with open(pf_file, 'w', encoding='utf-8') as f:
    json.dump(portfolio, f, ensure_ascii=False, indent=2)

# ═══════════ 8. 输出 ═══════════
print(f'\n--- 执行结果 ---')
for t in trades_today:
    print(f'  {t}')
if alerts:
    print(f'\n[告警]')
    for a in alerts:
        print(f'  !! {a}')

print(f'\n持仓: {len(positions)}只 | 市值: {total_market_value:.0f} | 现金: {cash:.0f}')
print(f'总资产: {cash+total_market_value:.0f} | 累计盈亏: {cash+total_market_value-CAPITAL:+.0f}')
print(f'已保存: {pf_file}')
print('Done.')
