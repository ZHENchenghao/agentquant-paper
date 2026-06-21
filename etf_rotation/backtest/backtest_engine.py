# -*- coding: utf-8 -*-
"""
ETF轮动回测引擎: 2002-2026全周期
月度调仓, 交易成本考虑, 基准对比
"""
import pandas as pd
import numpy as np


class ETFBacktest:
    """ETF轮动回测引擎

    假设:
    - 月度调仓, 每月末选ETF, 次月持有
    - 等权配置 Top-N ETF
    - 单边交易成本: 0.1% (ETF佣金+滑点)
    - 基准: 沪深300 或 所有行业等权
    """

    def __init__(self, return_matrix, benchmark_returns, top_n=5,
                 transaction_cost=0.001, initial_capital=1.0):
        """
        Args:
            return_matrix: DataFrame [date x industry] 日收益
            benchmark_returns: Series [date] 基准日收益
            top_n: 持仓数
            transaction_cost: 单边费率
            initial_capital: 初始资金
        """
        self.ret = return_matrix
        self.bench_ret = benchmark_returns
        self.top_n = top_n
        self.cost = transaction_cost
        self.initial = initial_capital

    def run(self, selections, weights=None):
        """
        执行回测

        Args:
            selections: dict {rebalance_date: [top etf codes]}
            weights: dict {rebalance_date: {etf_code: weight}}  (可选, 用于风险平价)

        Returns:
            portfolio: Series [date] 组合净值
            trades: list of {date, action, etf, weight}
            stats: dict 绩效统计
        """
        if not selections:
            return None, [], {}

        # 构建持仓序列
        holdings = self._build_holdings(selections, weights)
        trades = self._build_trades(holdings)

        # 计算组合日收益
        portfolio_daily = self._compute_portfolio_returns(holdings)

        # 绩效统计
        stats = self._compute_stats(portfolio_daily)

        return portfolio_daily, trades, stats

    def _build_holdings(self, selections, weights=None):
        """构建每日持仓映射"""
        rebalance_dates = sorted(selections.keys())
        holdings = {}

        for i, reb_date in enumerate(rebalance_dates):
            if i < len(rebalance_dates) - 1:
                next_reb = rebalance_dates[i + 1]
            else:
                next_reb = self.ret.index[-1]

            # 持仓: 有权重用dict, 无权重用list
            if weights and reb_date in weights:
                holding = weights[reb_date]
            else:
                holding = selections[reb_date]

            mask = (self.ret.index > reb_date) & (self.ret.index <= next_reb)
            for date in self.ret.index[mask]:
                holdings[date] = holding

        return holdings

    def _build_trades(self, holdings):
        """构建交易记录"""
        trades = []
        prev = None
        for date in sorted(holdings.keys()):
            current = set(holdings[date])
            if prev is not None:
                sold = prev - current
                bought = current - prev
                for e in sold:
                    trades.append({'date': date, 'action': 'SELL', 'etf': e})
                for e in bought:
                    trades.append({'date': date, 'action': 'BUY', 'etf': e, 'cost': self.cost})
            prev = current
        return trades

    def _compute_portfolio_returns(self, holdings):
        """计算组合日收益 (支持不等权)"""
        dates = sorted(holdings.keys())
        daily_rets = {}

        for date in dates:
            item = holdings[date]
            # 兼容等权和加权两种格式
            if isinstance(item, list):
                etfs = item
                w = {e: 1.0/len(etfs) for e in etfs} if etfs else {}
            else:
                w = item  # dict {etf: weight}
                etfs = list(w.keys())

            available = [(e, w[e]) for e in etfs
                         if e in self.ret.columns and date in self.ret.index and w[e] > 0]
            if not available:
                daily_rets[date] = 0
                continue

            day_ret = sum(self.ret.loc[date, e] * wt for e, wt in available)

            # 调仓日扣除交易成本
            if date == dates[0] or holdings.get(date) != holdings.get(dates[dates.index(date) - 1]):
                day_ret -= self.cost * len(available) / len(etfs) if etfs else 0

            daily_rets[date] = day_ret

        return pd.Series(daily_rets, name='portfolio')

    def _compute_stats(self, portfolio_daily):
        """计算绩效统计"""
        rets = portfolio_daily.dropna()
        if len(rets) < 20:
            return {}

        # 累积净值
        cum = (1 + rets).cumprod()

        # 年化
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        # 回撤
        peak = cum.cummax()
        dd = cum / peak - 1
        mdd = dd.min()

        # 基准对比
        aligned_bench = self.bench_ret.reindex(rets.index).dropna()
        if len(aligned_bench) > 20:
            bench_cum = (1 + aligned_bench).cumprod()
            bench_ann = aligned_bench.mean() * 252
            bench_vol = aligned_bench.std() * np.sqrt(252)
            bench_sharpe = bench_ann / bench_vol if bench_vol > 0 else 0
            bench_mdd = (bench_cum / bench_cum.cummax() - 1).min()

            # 超额
            excess = rets.reindex(aligned_bench.index).dropna() - aligned_bench
            info_ratio = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        else:
            bench_cum = pd.Series(dtype=float)
            bench_ann = bench_vol = bench_sharpe = bench_mdd = info_ratio = 0

        # 胜率
        win_rate = (rets > 0).mean()
        monthly_rets = rets.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_win_rate = (monthly_rets > 0).mean()

        return {
            'total_return': round(float(cum.iloc[-1] - 1) * 100, 1),
            'ann_return': round(float(ann_ret) * 100, 1),
            'ann_volatility': round(float(ann_vol) * 100, 1),
            'sharpe': round(float(sharpe), 3),
            'max_drawdown': round(float(mdd) * 100, 1),
            'calmar': round(float(ann_ret / abs(mdd)), 2) if mdd != 0 else 0,
            'daily_win_rate': round(float(win_rate) * 100, 1),
            'monthly_win_rate': round(float(monthly_win_rate) * 100, 1),
            'benchmark_return': round(float(bench_cum.iloc[-1] - 1) * 100, 1) if len(bench_cum) > 0 else 0,
            'benchmark_sharpe': round(float(bench_sharpe), 3),
            'benchmark_mdd': round(float(bench_mdd) * 100, 1),
            'excess_return': round(float(cum.iloc[-1] / bench_cum.iloc[-1] - 1) * 100, 1) if len(bench_cum) > 0 and bench_cum.iloc[-1] != 0 else 0,
            'info_ratio': round(float(info_ratio), 3),
            'start_date': str(rets.index[0].date()) if hasattr(rets.index[0], 'date') else str(rets.index[0]),
            'end_date': str(rets.index[-1].date()) if hasattr(rets.index[-1], 'date') else str(rets.index[-1]),
        }

    def yearly_breakdown(self, portfolio_daily):
        """分年绩效"""
        rets = portfolio_daily.dropna()
        yearly = {}
        for yr, group in rets.groupby(rets.index.year):
            ann = group.mean() * 252
            vol = group.std() * np.sqrt(252)
            sh = ann / vol if vol > 0 else 0
            cum = (1 + group).prod() - 1
            peak = (1 + group).cumprod().cummax()
            dd = (1 + group).cumprod() / peak - 1
            mdd = dd.min()
            yearly[yr] = {
                'return': round(float(ann) * 100, 1),
                'sharpe': round(float(sh), 2),
                'mdd': round(float(mdd) * 100, 1),
                'cum': round(float(cum) * 100, 1),
                'wr': round(float((group > 0).mean()) * 100, 1),
            }
        return yearly
