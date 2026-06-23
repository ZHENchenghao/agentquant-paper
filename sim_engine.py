# -*- coding: utf-8 -*-
"""
圆桌会议 · 本地仿真撮合引擎 v1.0
================================
零外部依赖, 基于DuckDB K线数据做日线级仿真撮合。

特性:
  - 限价单/市价单
  - 滑点模型 (基于ADV)
  - 完整手续费 (佣金0.03% + 印花税0.05%卖 + 过户费0.001%)
  - T+1 锁定
  - 涨跌停阻断
  - 持仓+现金+绩效追踪
  - 日度快照, 支持回滚

用法:
  engine = SimEngine(db_path='...')
  engine.submit(code='sh600519', side='BUY', price=1800, shares=100)
  engine.run_day('2026-06-20')  # 撮合当日所有委托
  print(engine.performance())
"""

import json, os, sys, io
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import duckdb
import numpy as np
import pandas as pd


class Side(Enum):
    BUY = 'BUY'
    SELL = 'SELL'


class OrderStatus(Enum):
    PENDING = 'PENDING'
    FILLED = 'FILLED'
    PARTIAL = 'PARTIAL'
    REJECTED = 'REJECTED'
    CANCELLED = 'CANCELLED'


@dataclass
class Order:
    order_id: int
    code: str
    side: Side
    price: float           # 0 = market order
    shares: int
    status: OrderStatus = OrderStatus.PENDING
    filled_shares: int = 0
    filled_price: float = 0.0
    submit_date: str = ''
    filled_date: str = ''
    commission: float = 0.0
    t_plus_1_locked: bool = False  # 当日买入, 次日才可卖


@dataclass
class Position:
    code: str
    name: str = ''
    shares: int = 0
    avg_cost: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


# ═══════════════════════════════════════════
# 撮合引擎
# ═══════════════════════════════════════════

class SimEngine:
    """本地仿真撮合引擎"""

    def __init__(
        self,
        db_path: str = 'D:/FreeFinanceData/data/duckdb/finance.db',
        initial_cash: float = 100000,
        commission_rate: float = 0.0003,    # 佣金
        stamp_tax: float = 0.0005,          # 印花税(卖)
        transfer_fee: float = 0.00001,       # 过户费
        slippage_bps: float = 3.0,           # 滑点(基点)
        snapshot_dir: Optional[str] = None,
    ):
        self.db_path = db_path
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
        self.slippage_bps = slippage_bps / 10000.0  # bps → 小数

        self.orders: List[Order] = []
        self._order_counter = 1
        self.positions: Dict[str, Position] = {}
        self.snapshots: List[Dict] = []  # 每日快照
        self.trade_log: List[Dict] = []
        self.current_date: str = ''
        self._t_plus_1_lock: Dict[str, int] = {}  # code → shares locked until next day

        # 快照目录
        self.snapshot_dir = snapshot_dir or os.path.dirname(os.path.abspath(__file__))
        self._snapshot_file = os.path.join(self.snapshot_dir, 'sim_account.json')

        # 加载已有状态
        self._load()

    # ═══════════ 持久化 ═══════════

    def _save(self):
        data = {
            'initial_cash': self.initial_cash,
            'cash': self.cash,
            'current_date': self.current_date,
            'positions': {k: {
                'code': v.code, 'name': v.name, 'shares': v.shares,
                'avg_cost': v.avg_cost, 'market_price': v.market_price,
                'market_value': v.market_value, 'pnl': v.pnl, 'pnl_pct': v.pnl_pct,
            } for k, v in self.positions.items()},
            't_plus_1_lock': self._t_plus_1_lock,
            'order_counter': self._order_counter,
            'snapshots': self.snapshots[-60:],  # 只保留最近60天
            'trade_log': self.trade_log[-200:],
        }
        with open(self._snapshot_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _load(self):
        if os.path.exists(self._snapshot_file):
            with open(self._snapshot_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.cash = data.get('cash', self.initial_cash)
            self.current_date = data.get('current_date', '')
            self._order_counter = data.get('order_counter', 1)
            self._t_plus_1_lock = data.get('t_plus_1_lock', {})
            self.snapshots = data.get('snapshots', [])
            self.trade_log = data.get('trade_log', [])
            for k, v in data.get('positions', {}).items():
                self.positions[k] = Position(**v)

    # ═══════════ 行情获取 ═══════════

    def _get_market(self, codes: List[str], trade_date: str) -> Dict[str, Dict]:
        """从DuckDB获取指定日期的行情"""
        ts_codes = []
        mapping = {}
        for c in codes:
            c = str(c).strip().lower()
            if c.startswith('sh'):
                ts = f'{c[2:]}.SH'
            elif c.startswith('sz'):
                ts = f'{c[2:]}.SZ'
            else:
                ts = c
            ts_codes.append(ts)
            mapping[ts] = c

        codes_str = ','.join([f"'{c}'" for c in ts_codes])

        con = duckdb.connect(self.db_path, read_only=True)
        df = con.execute(f"""
            SELECT ts_code, close, open, high, low, pre_close, vol, amount,
                   close/pre_close-1 AS change_pct,
                   CASE WHEN high/low >= 1.095 AND close/pre_close > 1.09 THEN 'LIMIT_UP'
                        WHEN low/high <= 0.905 AND close/pre_close < -0.09 THEN 'LIMIT_DOWN'
                        ELSE 'NORMAL' END AS limit_status,
                   AVG(COALESCE(amount, vol*close)) OVER(
                       PARTITION BY ts_code ORDER BY trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                   ) AS adv_20d
            FROM kline_daily
            WHERE trade_date = '{trade_date}'
              AND ts_code IN ({codes_str})
        """).df()

        # 获取名称
        names = con.execute("SELECT ts_code, name FROM stock_basic").df()
        con.close()

        name_map = dict(zip(names['ts_code'], names['name']))
        result = {}
        for _, row in df.iterrows():
            ts = row['ts_code']
            rt_code = mapping.get(ts, ts)
            result[rt_code] = {
                'name': name_map.get(ts, ''),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'pre_close': float(row['pre_close']),
                'change_pct': float(row['change_pct'] or 0),
                'vol': float(row['vol'] or 0),
                'amount': float(row['amount'] or 0),
                'adv_20d': float(row['adv_20d'] or 1e9),
                'limit_status': row['limit_status'],
            }
        return result

    # ═══════════ 下单 ═══════════

    def submit(self, code: str, side: str, price: float, shares: int) -> Order:
        """提交委托单

        Args:
            code: 股票代码 (sh600519 / sz000001)
            side: BUY / SELL
            price: 限价 (0 = 市价单)
            shares: 股数 (必须是100的整数倍)
        """
        if shares <= 0:
            raise ValueError(f'股数无效: {shares}')
        if shares % 100 != 0:
            shares = (shares // 100) * 100
            if shares == 0:
                raise ValueError('股数不足1手')

        order = Order(
            order_id=self._order_counter,
            code=code.lower(),
            side=Side(side.upper()),
            price=price,
            shares=shares,
            submit_date=self.current_date or str(date.today()),
        )
        self._order_counter += 1
        self.orders.append(order)
        return order

    def buy(self, code: str, price: float, shares: int) -> Order:
        return self.submit(code, 'BUY', price, shares)

    def sell(self, code: str, price: float, shares: int) -> Order:
        return self.submit(code, 'SELL', price, shares)

    def cancel(self, order_id: int) -> bool:
        for o in self.orders:
            if o.order_id == order_id and o.status == OrderStatus.PENDING:
                o.status = OrderStatus.CANCELLED
                return True
        return False

    # ═══════════ 撮合 ═══════════

    def run_day(self, trade_date: str) -> Dict:
        """撮合当日所有待成交委托

        撮合规则:
          1. 市价单 → 以当日收盘价成交 (含滑点)
          2. 限价买单 → 当日最低价 <= 限价 → 以min(限价, 收盘价)成交
          3. 限价卖单 → 当日最高价 >= 限价 → 以max(限价, 收盘价)成交
          4. 涨跌停 → 拒绝
          5. T+1 → 当日买入不能当日卖出
          6. 现金不足 → 部分成交或拒绝
        """
        self.current_date = trade_date
        if not self.orders:
            return {'date': trade_date, 'filled': 0, 'rejected': 0}

        # 获取所有涉及股票的行情
        codes_in_orders = list(set(o.code for o in self.orders if o.status == OrderStatus.PENDING))
        codes_in_positions = list(self.positions.keys())
        all_codes = list(set(codes_in_orders + codes_in_positions))
        market = self._get_market(all_codes, trade_date)

        # 释放前一天的T+1锁
        self._t_plus_1_lock = {}

        filled_count = 0
        rejected_count = 0

        for order in [o for o in self.orders if o.status == OrderStatus.PENDING]:
            m = market.get(order.code)
            if not m:
                order.status = OrderStatus.REJECTED
                rejected_count += 1
                continue

            # 涨跌停阻断
            if order.side == Side.BUY and m['limit_status'] == 'LIMIT_UP':
                order.status = OrderStatus.REJECTED
                rejected_count += 1
                continue
            if order.side == Side.SELL and m['limit_status'] == 'LIMIT_DOWN':
                order.status = OrderStatus.REJECTED
                rejected_count += 1
                continue

            # T+1: 当日买入的不能卖出
            if order.side == Side.SELL:
                locked = self._t_plus_1_lock.get(order.code, 0)
                pos = self.positions.get(order.code)
                available = (pos.shares - locked) if pos else 0
                if available <= 0:
                    order.status = OrderStatus.REJECTED
                    rejected_count += 1
                    continue
                order.shares = min(order.shares, available)

            # 撮合价格
            fill_price = self._match_price(order, m)

            # 现金检查(买入)
            if order.side == Side.BUY:
                cost = fill_price * order.shares * (1 + self.commission_rate + self.transfer_fee)
                if cost > self.cash:
                    # 部分成交
                    max_shares = int(self.cash / (fill_price * (1 + self.commission_rate + self.transfer_fee)) / 100) * 100
                    if max_shares == 0:
                        order.status = OrderStatus.REJECTED
                        rejected_count += 1
                        continue
                    order.shares = max_shares
                    order.status = OrderStatus.PARTIAL

            # 执行成交
            self._execute_fill(order, fill_price, m['name'], trade_date)
            filled_count += 1

        # 更新市值
        self._mark_to_market(market, trade_date)

        # 日度快照
        self._make_snapshot(trade_date)
        self._save()

        return {
            'date': trade_date,
            'filled': filled_count,
            'rejected': rejected_count,
            'cash': round(self.cash, 2),
            'total_value': round(self.total_value(), 2),
            'pnl': round(self.total_value() - self.initial_cash, 2),
        }

    def _match_price(self, order: Order, m: Dict) -> float:
        """计算撮合价格 (含滑点)"""
        if order.price == 0:
            # 市价单 → 收盘价 + 滑点
            base = m['close']
        elif order.side == Side.BUY:
            # 限价买单: 最低价 <= 限价 才能成交
            if m['low'] > order.price:
                return 0  # 未触及限价, 保持PENDING, 不成交
            base = min(order.price, m['close'])
        else:
            # 限价卖单: 最高价 >= 限价 才能成交
            if m['high'] < order.price:
                return 0
            base = max(order.price, m['close'])

        # 滑点: 成交额相对ADV越大, 滑点越大
        trade_value = base * order.shares
        adv = max(m['adv_20d'], 1e6)
        impact = min(trade_value / adv * self.slippage_bps * 100, 0.01)  # 上限1%

        if order.side == Side.BUY:
            return base * (1 + impact)
        else:
            return base * (1 - impact)

    def _execute_fill(self, order: Order, fill_price: float, name: str, trade_date: str):
        """执行成交, 更新持仓和现金"""
        if fill_price <= 0:
            order.status = OrderStatus.REJECTED
            return

        order.status = OrderStatus.FILLED
        order.filled_shares = order.shares
        order.filled_price = fill_price
        order.filled_date = trade_date

        trade_value = fill_price * order.shares
        commission = trade_value * self.commission_rate
        commission = max(commission, 5.0)  # 最低5元佣金
        stamp = trade_value * self.stamp_tax if order.side == Side.SELL else 0
        transfer = trade_value * self.transfer_fee
        order.commission = commission + stamp + transfer

        if order.side == Side.BUY:
            cost = trade_value + order.commission
            self.cash -= cost

            # 更新持仓
            if order.code in self.positions:
                pos = self.positions[order.code]
                total_shares = pos.shares + order.shares
                pos.avg_cost = (pos.avg_cost * pos.shares + fill_price * order.shares) / total_shares
                pos.shares = total_shares
            else:
                self.positions[order.code] = Position(
                    code=order.code, name=name, shares=order.shares,
                    avg_cost=fill_price,
                )

            # T+1 锁定
            self._t_plus_1_lock[order.code] = self._t_plus_1_lock.get(order.code, 0) + order.shares

        else:
            revenue = trade_value - order.commission
            self.cash += revenue

            pos = self.positions.get(order.code)
            if pos:
                pos.shares -= order.shares
                if pos.shares <= 0:
                    del self.positions[order.code]

        # 交易日志
        self.trade_log.append({
            'date': trade_date,
            'code': order.code,
            'name': name,
            'side': order.side.value,
            'shares': order.shares,
            'price': round(fill_price, 2),
            'commission': round(order.commission, 2),
            'order_id': order.order_id,
        })

    def _mark_to_market(self, market: Dict, trade_date: str):
        """按市价更新持仓"""
        for code, pos in self.positions.items():
            m = market.get(code)
            if m:
                pos.market_price = m['close']
                pos.market_value = pos.shares * pos.market_price
                pos.pnl = (pos.market_price - pos.avg_cost) * pos.shares
                pos.pnl_pct = (pos.market_price / pos.avg_cost - 1) * 100 if pos.avg_cost > 0 else 0

    def _make_snapshot(self, trade_date: str):
        """日度快照"""
        total_mv = sum(p.market_value for p in self.positions.values())
        total_value = self.cash + total_mv
        self.snapshots.append({
            'date': trade_date,
            'cash': round(self.cash, 2),
            'market_value': round(total_mv, 2),
            'total_value': round(total_value, 2),
            'pnl': round(total_value - self.initial_cash, 2),
            'pnl_pct': round(total_value / self.initial_cash - 1, 4),
            'positions_count': len(self.positions),
        })

    # ═══════════ 查询 ═══════════

    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def get_positions(self) -> List[Dict]:
        return [{
            'code': p.code, 'name': p.name, 'shares': p.shares,
            'avg_cost': round(p.avg_cost, 2), 'market_price': round(p.market_price, 2),
            'market_value': round(p.market_value, 2),
            'pnl': round(p.pnl, 2), 'pnl_pct': round(p.pnl_pct, 2),
        } for p in self.positions.values()]

    def get_pending_orders(self) -> List[Dict]:
        return [{
            'order_id': o.order_id, 'code': o.code, 'side': o.side.value,
            'price': o.price, 'shares': o.shares, 'status': o.status.value,
        } for o in self.orders if o.status == OrderStatus.PENDING]

    def get_trades(self, limit: int = 50) -> List[Dict]:
        return self.trade_log[-limit:]

    def performance(self) -> Dict:
        """绩效摘要"""
        tv = self.total_value()
        total_pnl = tv - self.initial_cash
        days = len(self.snapshots)
        if days < 2:
            return {
                'total_pnl': round(total_pnl, 2),
                'total_pnl_pct': round(total_pnl / self.initial_cash * 100, 2),
                'cash': round(self.cash, 2),
                'total_value': round(tv, 2),
                'positions': len(self.positions),
                'total_trades': len(self.trade_log),
                'trading_days': days,
                'daily_pnl_series': None,
                'sharpe': None, 'max_drawdown': None, 'win_rate': None,
            }

        # 日收益序列
        daily_pnl = [s['pnl'] for s in self.snapshots]
        daily_returns = []
        for i in range(1, len(self.snapshots)):
            prev_tv = self.snapshots[i-1]['total_value']
            if prev_tv > 0:
                daily_returns.append((self.snapshots[i]['total_value'] - prev_tv) / prev_tv)

        # 夏普比率 (年化)
        if daily_returns and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0

        # 最大回撤
        peak = self.initial_cash
        mdd = 0
        for s in self.snapshots:
            tv_s = s['total_value']
            if tv_s > peak:
                peak = tv_s
            dd = (peak - tv_s) / peak
            if dd > mdd:
                mdd = dd

        # 胜率
        winning_trades = 0
        buy_prices = {}
        for t in self.trade_log:
            code = t['code']
            if t['side'] == 'BUY':
                if code not in buy_prices:
                    buy_prices[code] = []
                buy_prices[code].append((t['shares'], t['price']))
            elif t['side'] == 'SELL' and code in buy_prices and buy_prices[code]:
                avg_buy = sum(b[0]*b[1] for b in buy_prices[code]) / sum(b[0] for b in buy_prices[code])
                if t['price'] > avg_buy:
                    winning_trades += 1
                buy_prices[code] = []

        total_round_trips = len([t for t in self.trade_log if t['side'] == 'SELL'])
        win_rate = winning_trades / total_round_trips if total_round_trips > 0 else 0

        return {
            'total_pnl': round(total_pnl, 2),
            'total_pnl_pct': round(total_pnl / self.initial_cash * 100, 2),
            'cash': round(self.cash, 2),
            'total_value': round(tv, 2),
            'positions': len(self.positions),
            'total_trades': len(self.trade_log),
            'trading_days': days,
            'daily_pnl_series': daily_pnl,
            'sharpe': round(sharpe, 2),
            'max_drawdown': round(mdd * 100, 2),
            'win_rate': round(win_rate * 100, 1),
        }

    def print_report(self):
        """打印绩效报告"""
        perf = self.performance()
        print(f"""
{'='*55}
  圆桌会议 仿真交易绩效报告
{'='*55}
  初始资金:    {self.initial_cash:,.0f}
  当前总资产:  {perf['total_value']:,.0f}
  累计盈亏:    {perf['total_pnl']:+,.0f} ({perf['total_pnl_pct']:+.2f}%)
  持仓数:      {perf['positions']}只
  总成交:      {perf['total_trades']}笔
  交易天数:    {perf['trading_days']}天
{'='*55}
  夏普比率:    {perf['sharpe']}
  最大回撤:    {perf['max_drawdown']}%
  胜率:        {perf['win_rate']}%
{'='*55}""")


# ═══════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════

if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print('=== 圆桌会议 仿真撮合引擎 测试 ===\n')

    engine = SimEngine(initial_cash=200000)

    # 获取最近交易日
    con = duckdb.connect(engine.db_path, read_only=True)
    dates = con.execute("""
        SELECT DISTINCT trade_date FROM kline_daily
        WHERE trade_date >= '2026-06-01'
        ORDER BY trade_date DESC LIMIT 15
    """).fetchall()
    con.close()
    trade_dates = [d[0] for d in reversed(dates)]

    if not trade_dates:
        print('无交易数据, 请先跑 daily_runner.py 更新K线')
        sys.exit(1)

    print(f'回测区间: {trade_dates[0]} ~ {trade_dates[-1]} ({len(trade_dates)}天)\n')

    # 简单策略: 等权买入前3只有数据的股票
    test_codes = ['sh600519', 'sz000858', 'sh600036']
    per_stock = engine.initial_cash * 0.3 / len(test_codes)

    for d in trade_dates:
        # 每天对每只股票挂买单
        for code in test_codes:
            engine.buy(code, price=0, shares=int(per_stock / 100) * 100)  # 市价单

        result = engine.run_day(str(d))
        if result['filled'] > 0:
            print(f"  {d}: 成交{result['filled']}笔 现金={result['cash']:.0f} 总资产={result['total_value']:.0f} 盈亏={result['pnl']:+.0f}")

    engine.print_report()
