# -*- coding: utf-8 -*-
"""
AgentQuant · 纸交自动调仓
=========================
跑Agent分析持仓 → SELL清仓 / BUY买入 / HOLD不动 → 更新paper_portfolio
"""
import sys, os, json, time, io
sys.path.insert(0, 'D:/AgentQuant/our')
import duckdb
from datetime import date

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
PF = 'D:/AgentQuant/our/paper_portfolio.json'

def load_portfolio():
    with open(PF, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_portfolio(pf):
    with open(PF, 'w', encoding='utf-8') as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

def get_price(code, trade_date=None):
    c = duckdb.connect(DB, read_only=True)
    if trade_date is None:
        trade_date = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
    ts = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    r = c.execute("SELECT close FROM kline_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                  [ts, trade_date.isoformat() if hasattr(trade_date, 'isoformat') else trade_date]).fetchone()
    c.close()
    return r[0] if r else None

def run_agent_on_holdings(holdings):
    """对持仓跑Agent分析, 返回每只的信号"""
    os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY_PLACEHOLDER')
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from agent_macro_context import build_macro_context

    config = DEFAULT_CONFIG.copy()
    config.update({'llm_provider': 'deepseek', 'deep_think_llm': 'deepseek-chat',
                   'quick_think_llm': 'deepseek-chat', 'max_debate_rounds': 1, 'max_risk_discuss_rounds': 1})

    macro_ctx = build_macro_context()
    trade_date = date.today().isoformat()

    results = {}
    for code in holdings:
        ta = TradingAgentsGraph(debug=False, config=config)
        orig_method = ta.memory_log.get_past_context
        def patched_get(ticker, ctx=macro_ctx, orig=orig_method):
            old_ctx = orig(ticker) if callable(orig) else ''
            return ctx + '\n\n---\n\n' + str(old_ctx) if old_ctx else ctx
        ta.memory_log.get_past_context = patched_get

        print(f'  {code}...', end=' ', flush=True)
        try:
            fs, decision = ta.propagate(code, trade_date)
            d_str = str(decision).lower()
            if 'sell' in d_str or 'underweight' in d_str:
                signal = 'SELL'
            elif 'buy' in d_str or 'overweight' in d_str:
                signal = 'BUY'
            else:
                signal = 'HOLD'
            print(signal)
            results[code] = signal
        except Exception as e:
            print(f'ERR: {e}')
            results[code] = 'HOLD'  # 出错不动
        time.sleep(1)
    return results

def execute(pf, signals, today_prices):
    """执行调仓: SELL→清仓, HOLD→不动"""
    td = str(date.today())
    trades = []

    for code, pos in list(pf['positions'].items()):
        signal = signals.get(code.replace('.SH', '').replace('.SZ', ''), 'HOLD')
        px = today_prices.get(code)
        if not px:
            continue

        if signal == 'SELL':
            # 清仓
            shares = pos['shares']
            cost = pos['buy_price']
            value = shares * px
            pnl = value - shares * cost
            pf['cash'] += value * 0.9989  # 扣除卖出手续费+滑点
            del pf['positions'][code]
            trades.append({
                'date': td, 'action': 'SELL', 'code': code,
                'price': round(px, 2), 'shares': shares, 'value': round(value, 2),
                'pnl': round(pnl, 2), 'reason': f'Agent判{signal}'
            })
            print(f'  SELL {code}: {cost}→{px:.2f} {shares}股 PnL={pnl:+.0f}')

        elif signal == 'BUY':
            # 已持有, 不加仓 (纸交简化: 不重复买)
            pass

    pf['history'].extend(trades)
    return pf

if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    pf = load_portfolio()
    holdings = list(pf['positions'].keys())
    codes = [c.replace('.SH', '').replace('.SZ', '') for c in holdings]

    print(f'纸交自动调仓 — {date.today()}')
    print(f'持仓: {len(holdings)}只 现金: ¥{pf["cash"]:,.0f}')
    print()

    # 1. Agent分析
    print('Agent分析持仓...')
    signals = run_agent_on_holdings(codes)

    # 2. 获取今日价格
    today_prices = {}
    for code_str in holdings:
        px = get_price(code_str)
        if px:
            today_prices[code_str] = px

    # 3. 执行调仓
    print()
    print('执行调仓...')
    pf = execute(pf, signals, today_prices)

    # 4. 保存
    pf['saved_at'] = str(date.today())
    save_portfolio(pf)

    # 5. 汇总
    mv = 0
    for code, pos in pf['positions'].items():
        px = today_prices.get(code, pos['buy_price'])
        mv += pos['shares'] * px
    print(f'\n调仓后: 现金¥{pf["cash"]:,.0f} 市值¥{mv:,.0f} 总资产¥{pf["cash"]+mv:,.0f}')
    print(f'持仓: {len(pf["positions"])}只')
    if len(pf['history']) > 0:
        last = pf['history'][-1]
        print(f'上次交易: {last["date"]} {last["action"]} {last["code"]}')
