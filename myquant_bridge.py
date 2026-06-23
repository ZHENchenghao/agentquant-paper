# -*- coding: utf-8 -*-
"""
圆桌会议 · 交易桥接层 v2.0
==========================
统一交易接口, 三模切换:
  - paper   : 本地JSON纸交 (零依赖, 默认)
  - sim     : 本地仿真撮合引擎 (SimEngine, 日线级撮合+滑点+手续费)
  - myquant : 掘金仿真交易 (gmtrade SDK, 需token+account)

用法:
  bridge = TradingBridge(mode='paper')              # 纸交
  bridge = TradingBridge(mode='sim', capital=100000) # 仿真撮合
  bridge = TradingBridge(mode='myquant',             # 掘金仿真
                          token='xxx', account_id='xxx')

代码格式: sh600519 / sz000001
"""

import json, os, time, logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
# 代码格式转换
# ═══════════════════════════════════════════

def to_myquant_code(rt_code: str) -> str:
    """圆桌会议 → 掘金代码: sh600519 → SHSE.600519"""
    rt_code = str(rt_code).strip().lower()
    if rt_code.startswith('sh'):
        return f'SHSE.{rt_code[2:]}'
    elif rt_code.startswith('sz'):
        return f'SZSE.{rt_code[2:]}'
    elif rt_code.startswith('bj'):
        return f'BSE.{rt_code[2:]}'
    # 无前缀, 6位数字判断
    if len(rt_code) == 6 and rt_code.isdigit():
        if rt_code.startswith(('6', '9')):
            return f'SHSE.{rt_code}'
        else:
            return f'SZSE.{rt_code}'
    return rt_code.upper()


def from_myquant_code(mq_code: str) -> str:
    """掘金 → 圆桌会议代码: SHSE.600519 → sh600519"""
    parts = mq_code.split('.')
    if len(parts) == 2:
        exchange = parts[0].lower()
        code = parts[1]
        if exchange == 'shse':
            return f'sh{code}'
        elif exchange == 'szse':
            return f'sz{code}'
        elif exchange == 'bse':
            return f'bj{code}'
    return mq_code.lower()


# ═══════════════════════════════════════════
# TradingBridge
# ═══════════════════════════════════════════

class TradingBridge:
    """统一交易接口 — paper / sim / myquant 三模"""

    def __init__(
        self,
        mode: str = 'paper',
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        endpoint: str = 'api.myquant.cn:9000',
        paper_dir: Optional[str] = None,
        paper_file: str = 'paper_portfolio_xiaozhong.json',
        capital: float = 100000,
        db_path: str = 'D:/FreeFinanceData/data/duckdb/finance.db',
    ):
        self.mode = mode
        self.token = token
        self.account_id = account_id
        self.endpoint = endpoint
        self.capital = capital
        self.db_path = db_path
        self._mq = None
        self._sim = None
        self._logged_in = False

        if mode == 'paper':
            self._paper_dir = paper_dir or os.path.dirname(os.path.abspath(__file__))
            self._paper_file = os.path.join(self._paper_dir, paper_file)
            self._load_paper_portfolio()
            logger.info(f'桥接层: PAPER模式 → {self._paper_file}')
        elif mode == 'sim':
            from sim_engine import SimEngine
            snapshot_dir = paper_dir or os.path.dirname(os.path.abspath(__file__))
            self._sim = SimEngine(db_path=db_path, initial_cash=capital,
                                  snapshot_dir=snapshot_dir)
            logger.info(f'桥接层: SIM仿真模式 → 初始资金{capital:,.0f}')
        elif mode == 'myquant':
            if not token or not account_id:
                raise ValueError('myquant模式需要 token 和 account_id')
            self._init_myquant()
            logger.info(f'桥接层: MYQUANT仿真模式 → {endpoint}')

    # ═══════════ 纸交模式 ═══════════

    def _load_paper_portfolio(self):
        if os.path.exists(self._paper_file):
            with open(self._paper_file, 'r', encoding='utf-8') as f:
                self._paper = json.load(f)
        else:
            self._paper = {
                'strategy': '圆桌会议_纸交',
                'date': str(date.today()),
                'capital': self.capital,
                'cash': self.capital,
                'invested': 0,
                'total_value': self.capital,
                'positions': [],
                'trades': [],
            }
        # 确保关键字段存在
        self._paper.setdefault('cash', self.capital)
        self._paper.setdefault('positions', [])
        self._paper.setdefault('trades', [])

    def _save_paper_portfolio(self):
        self._paper['date'] = str(date.today())
        total_value = self._paper['cash']
        for p in self._paper['positions']:
            total_value += p.get('shares', 0) * p.get('price', p.get('buy_price', 0))
        self._paper['total_value'] = round(total_value, 2)
        self._paper['invested'] = round(total_value - self._paper['cash'], 2)
        with open(self._paper_file, 'w', encoding='utf-8') as f:
            json.dump(self._paper, f, ensure_ascii=False, indent=2)

    def _paper_buy(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        cost = shares * price * 1.00033
        if cost > self._paper['cash']:
            actual_shares = int(self._paper['cash'] / (price * 1.00033) / 100) * 100
            if actual_shares == 0:
                return {'success': False, 'error': '现金不足', 'code': code}
            shares = actual_shares
            cost = shares * price * 1.00033

        self._paper['cash'] -= cost
        positions = {p['code']: p for p in self._paper['positions']}
        if code in positions:
            old = positions[code]
            total_shares = old['shares'] + shares
            avg_price = (old['buy_price'] * old['shares'] + price * shares) / total_shares
            positions[code] = {
                'code': code, 'name': name or old.get('name', ''),
                'shares': total_shares, 'buy_price': round(avg_price, 3),
                'price': price, 'cost': old.get('cost', 0) + cost,
            }
        else:
            positions[code] = {
                'code': code, 'name': name, 'shares': shares,
                'buy_price': price, 'price': price, 'cost': float(cost),
            }
        self._paper['positions'] = list(positions.values())
        self._paper['trades'].append({
            'date': str(date.today()), 'action': 'BUY', 'code': code,
            'shares': shares, 'price': price, 'cost': round(cost, 2),
        })
        self._save_paper_portfolio()
        return {'success': True, 'code': code, 'action': 'BUY', 'shares': shares,
                'price': price, 'cost': round(cost, 2)}

    def _paper_sell(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        positions = {p['code']: p for p in self._paper['positions']}
        if code not in positions:
            return {'success': False, 'error': '无持仓', 'code': code}
        pos = positions[code]
        actual_shares = min(shares, pos['shares'])
        revenue = actual_shares * price * 0.99967
        self._paper['cash'] += revenue
        remaining = pos['shares'] - actual_shares
        if remaining > 0:
            positions[code]['shares'] = remaining
        else:
            del positions[code]
        self._paper['positions'] = list(positions.values())
        self._paper['trades'].append({
            'date': str(date.today()), 'action': 'SELL', 'code': code,
            'shares': actual_shares, 'price': price, 'revenue': round(revenue, 2),
        })
        self._save_paper_portfolio()
        return {'success': True, 'code': code, 'action': 'SELL',
                'shares': actual_shares, 'price': price, 'revenue': round(revenue, 2)}

    # ═══════════ 掘金仿真模式 ═══════════

    def _init_myquant(self):
        try:
            from gmtrade.api import set_token, set_endpoint, account, login
            set_token(self.token)
            set_endpoint(self.endpoint)
            self._mq = {
                'set_token': set_token, 'set_endpoint': set_endpoint,
                'account': account, 'login': login,
            }
            # 登录
            a1 = account(account_id=self.account_id, account_alias='')
            login(a1)
            self._logged_in = True
            logger.info(f'掘金仿真登录成功: {self.account_id}')
        except ImportError:
            raise RuntimeError('请先安装掘金SDK: pip install gm')
        except Exception as e:
            logger.error(f'掘金仿真登录失败: {e}')
            raise

    def _ensure_myquant(self):
        if not self._logged_in:
            self._init_myquant()

    def _myquant_buy(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        self._ensure_myquant()
        from gmtrade.api import (
            order_volume, OrderSide_Buy, OrderType_Limit,
            PositionEffect_Open, get_cash,
        )
        mq_code = to_myquant_code(code)
        # 检查资金
        cash = get_cash()
        if cash.available < shares * price * 1.00033:
            shares = int(cash.available / (price * 1.00033) / 100) * 100
            if shares == 0:
                return {'success': False, 'error': '现金不足', 'code': code}
        try:
            order_volume(
                symbol=mq_code, volume=shares,
                side=OrderSide_Buy, order_type=OrderType_Limit,
                position_effect=PositionEffect_Open, price=price,
            )
            logger.info(f'掘金买入: {mq_code} {shares}股 @{price}')
            return {'success': True, 'code': code, 'action': 'BUY',
                    'shares': shares, 'price': price}
        except Exception as e:
            return {'success': False, 'error': str(e), 'code': code}

    def _myquant_sell(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        self._ensure_myquant()
        from gmtrade.api import (
            order_volume, OrderSide_Sell, OrderType_Limit,
            PositionEffect_Close, get_positions,
        )
        mq_code = to_myquant_code(code)
        # 检查持仓
        positions = get_positions()
        pos = next((p for p in positions if p.symbol == mq_code), None)
        if not pos or pos.available_volume <= 0:
            return {'success': False, 'error': '无可用持仓', 'code': code}
        actual_shares = min(shares, pos.available_volume)
        try:
            order_volume(
                symbol=mq_code, volume=actual_shares,
                side=OrderSide_Sell, order_type=OrderType_Limit,
                position_effect=PositionEffect_Close, price=price,
            )
            logger.info(f'掘金卖出: {mq_code} {actual_shares}股 @{price}')
            return {'success': True, 'code': code, 'action': 'SELL',
                    'shares': actual_shares, 'price': price}
        except Exception as e:
            return {'success': False, 'error': str(e), 'code': code}

    # ═══════════ 统一接口 ═══════════

    def _sim_buy(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        """仿真模式买入 — 挂单到SimEngine"""
        order = self._sim.buy(code, price, shares)
        return {'success': True, 'code': code, 'action': 'BUY',
                'shares': shares, 'price': price, 'order_id': order.order_id}

    def _sim_sell(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        """仿真模式卖出"""
        order = self._sim.sell(code, price, shares)
        return {'success': True, 'code': code, 'action': 'SELL',
                'shares': shares, 'price': price, 'order_id': order.order_id}

    def run_day(self, trade_date: str = None) -> Dict:
        """仿真模式: 撮合当日所有委托单"""
        if self.mode != 'sim':
            return {'error': 'run_day 仅用于 sim 模式'}
        td = trade_date or str(date.today())
        return self._sim.run_day(td)

    def performance(self) -> Dict:
        """仿真模式: 获取绩效摘要"""
        if self.mode == 'sim':
            return self._sim.performance()
        return {}

    def buy(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        """买入下单"""
        if shares <= 0:
            return {'success': False, 'error': '股数无效', 'code': code}
        if self.mode == 'paper':
            return self._paper_buy(code, price, shares, name)
        elif self.mode == 'sim':
            return self._sim_buy(code, price, shares, name)
        return self._myquant_buy(code, price, shares, name)

    def sell(self, code: str, price: float, shares: int, name: str = '') -> Dict:
        """卖出下单"""
        if shares <= 0:
            return {'success': False, 'error': '股数无效', 'code': code}
        if self.mode == 'paper':
            return self._paper_sell(code, price, shares, name)
        elif self.mode == 'sim':
            return self._sim_sell(code, price, shares, name)
        return self._myquant_sell(code, price, shares, name)

    def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        if self.mode == 'paper':
            return self._paper.get('positions', [])
        elif self.mode == 'sim':
            return self._sim.get_positions()
        self._ensure_myquant()
        from gmtrade.api import get_positions
        raw = get_positions()
        return [{
            'code': from_myquant_code(p.symbol),
            'name': '', 'shares': p.volume,
            'buy_price': p.vwap, 'price': p.last_price,
            'pnl': p.pnl,
        } for p in raw]

    def get_cash(self) -> float:
        """获取可用资金"""
        if self.mode == 'paper':
            return self._paper.get('cash', 0)
        elif self.mode == 'sim':
            return self._sim.cash
        self._ensure_myquant()
        from gmtrade.api import get_cash
        return float(get_cash().available)

    def get_total_value(self) -> float:
        """获取总资产"""
        if self.mode == 'paper':
            return self._paper.get('total_value', 0)
        elif self.mode == 'sim':
            return self._sim.total_value()
        self._ensure_myquant()
        from gmtrade.api import get_cash, get_positions
        cash = float(get_cash().available)
        positions = get_positions()
        market_value = sum(p.volume * p.last_price for p in positions)
        return cash + market_value

    def get_trades(self, limit: int = 50) -> List[Dict]:
        """获取最近成交记录"""
        if self.mode == 'paper':
            trades = self._paper.get('trades', [])
            return trades[-limit:] if len(trades) > limit else trades
        elif self.mode == 'sim':
            return self._sim.get_trades(limit)
        self._ensure_myquant()
        from gmtrade.api import get_history_orders
        try:
            orders = get_history_orders()
            return [{
                'code': from_myquant_code(o.symbol),
                'action': 'BUY' if o.side == 1 else 'SELL',
                'shares': o.volume, 'price': o.price,
                'status': o.status,
            } for o in orders[-limit:]]
        except Exception:
            return []

    def get_status(self) -> Dict:
        """获取账户状态摘要"""
        positions = self.get_positions()
        cash = self.get_cash()
        total = self.get_total_value()
        if self.mode == 'sim':
            perf = self._sim.performance()
            pnl = perf['total_pnl']
        elif self.mode == 'paper':
            pnl = total - self.capital
        else:
            pnl = None
        return {
            'mode': self.mode,
            'cash': round(cash, 2),
            'positions_count': len(positions),
            'total_value': round(total, 2),
            'pnl': round(pnl, 2) if pnl is not None else None,
        }


# ═══════════════════════════════════════════
# 快速创建函数
# ═══════════════════════════════════════════

def create_paper_bridge(paper_dir: str = None, capital: float = 100000) -> TradingBridge:
    """创建纸交桥接"""
    return TradingBridge(mode='paper', paper_dir=paper_dir, capital=capital)


def create_myquant_bridge(token: str, account_id: str, capital: float = 100000) -> TradingBridge:
    """创建掘金仿真桥接"""
    return TradingBridge(mode='myquant', token=token, account_id=account_id, capital=capital)


# ═══════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════

if __name__ == '__main__':
    print('=== 圆桌会议 交易桥接层 测试 ===\n')

    # 测试纸交模式
    bridge = create_paper_bridge()
    print(f'纸交模式: {bridge.get_status()}')

    # 测试买入
    r = bridge.buy('sh600519', 1800.0, 100, '贵州茅台')
    print(f'买入: {r}')

    # 测试卖出
    r = bridge.sell('sh600519', 1850.0, 50, '贵州茅台')
    print(f'卖出: {r}')

    # 测试持仓
    positions = bridge.get_positions()
    print(f'持仓: {positions}')

    # 测试成交记录
    trades = bridge.get_trades()
    print(f'成交: {trades}')

    # 代码转换测试
    print(f'\n代码转换:')
    print(f'  sh600519 → {to_myquant_code("sh600519")}')
    print(f'  sz000001 → {to_myquant_code("sz000001")}')
    print(f'  SHSE.600519 → {from_myquant_code("SHSE.600519")}')

    print('\n[OK] 桥接层测试完成')
