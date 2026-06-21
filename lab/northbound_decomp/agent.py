# -*- coding: utf-8 -*-
"""
Agent 2: 北向资金拆解Agent
交易所维度 → 行业偏好推断 → 三方聪明钱分歧 → 持续性评估
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from base_agent import BaseAgent
from shared.db import get_conn
from northbound_decomp.sector_flow import NorthboundDecomposer


class NorthboundDecompAgent(BaseAgent):
    """北向资金拆解Agent

    每日执行:
    1. 加载北向资金日度数据 (沪港通/深港通)
    2. 行业维度偏好推断 (行业指数收益 × 北向净流相关性)
    3. 三方聪明钱分歧检测 (北向 vs 融资 vs 主力)
    4. 资金流持续性评估
    """

    def __init__(self):
        super().__init__(
            name='NorthboundDecomp',
            description='北向资金拆解: 交易所维度 → 行业偏好 → 三方分歧 → 持续性'
        )
        self.decomposer = NorthboundDecomposer(lookback_days=60)
        self.northbound_df = None
        self.margin_df = None
        self.capital_flow_df = None
        self.industry_returns = None

    def validate_data(self):
        """验证数据可用"""
        conn = get_conn()
        try:
            # 1. 北向数据 (用新表 lab_northbound_daily, 10年)
            self.northbound_df = conn.execute("""
                SELECT trade_date, net_flow, buy_amount, sell_amount,
                       cum_net, hs300_close, hs300_pct
                FROM lab_northbound_daily
                WHERE net_flow IS NOT NULL
                  AND trade_date >= DATE '2016-01-01'
                ORDER BY trade_date
            """).df()

            # 2. 融资数据 (用新表 lab_margin_daily, 10年)
            self.margin_df = conn.execute("""
                SELECT trade_date, margin_balance, margin_buy
                FROM lab_margin_daily
                WHERE trade_date >= DATE '2016-01-01'
                ORDER BY trade_date
            """).df()

            # 3. 主力资金数据 (降级: lab_capital_flow 回填未完成)
            try:
                self.capital_flow_df = conn.execute("""
                    SELECT trade_date, main_net
                    FROM lab_capital_flow
                    WHERE trade_date >= DATE '2016-01-01'
                    ORDER BY trade_date
                """).df()
            except:
                self.capital_flow_df = pd.DataFrame()

            # 4. 行业指数数据 (10年)
            ai_list = "', '".join(['电子', '计算机', '通信', '电力设备',
                                    '有色金属', '银行', '非银金融', '医药生物',
                                    '食品饮料', '汽车', '国防军工', '基础化工'])
            self.industry_returns = conn.execute(f"""
                WITH industry_daily_ret AS (
                    SELECT trade_date, industry,
                           (close - LAG(close) OVER(
                               PARTITION BY industry ORDER BY trade_date
                           )) / NULLIF(LAG(close) OVER(
                               PARTITION BY industry ORDER BY trade_date
                           ), 0) AS daily_return
                    FROM proxy_industry_daily
                    WHERE industry IN ('{ai_list}')
                      AND trade_date >= DATE '2016-01-01'
                )
                SELECT trade_date, industry AS ind_name, daily_return
                FROM industry_daily_ret
                WHERE daily_return IS NOT NULL
                ORDER BY trade_date
            """).df()

            self.data_status = {
                'northbound': f'{len(self.northbound_df)}行',
                'margin': f'{len(self.margin_df)}行',
                'capital_flow': f'{len(self.capital_flow_df)}行',
                'industry_returns': f'{len(self.industry_returns)}行',
            }

            nb_dr = f"{self.northbound_df['trade_date'].min()} ~ {self.northbound_df['trade_date'].max()}"
            print(f"    北向: {len(self.northbound_df)}行 ({nb_dr}) | "
                  f"融资: {len(self.margin_df)}行 | "
                  f"行业: {self.industry_returns['ind_name'].nunique() if not self.industry_returns.empty else 0}个")

            return len(self.northbound_df) > 100

        except Exception as e:
            self.data_status['error'] = str(e)
            return False
        finally:
            conn.close()

    def analyze(self):
        """核心分析: 市场级北向+融资+行业"""
        # 构建北向日序列
        nb = self.northbound_df.copy()
        nb['trade_date'] = pd.to_datetime(nb['trade_date'])
        nb = nb.sort_values('trade_date')
        nb_series = nb.set_index('trade_date')['net_flow']

        # 1. 市场级北向分析 (替代交易所拆解)
        print("    北向市场级分析...")
        market_flow = self._analyze_market_flow(nb)

        # 2. 行业偏好推断
        print("    推断行业偏好...")
        sector_pref = {}
        if not self.industry_returns.empty:
            sector_pref = self.decomposer.infer_sector_preference(
                self.industry_returns, nb_series
            )

        # 3. 北向 vs 融资 分歧检测
        print("    聪明钱分歧检测...")
        margin = self.margin_df.copy()
        margin['trade_date'] = pd.to_datetime(margin['trade_date'])
        margin_series = margin.set_index('trade_date')['margin_buy'].sort_index()

        cf_series = pd.Series(dtype=float)
        if not self.capital_flow_df.empty:
            cf = self.capital_flow_df.copy()
            cf['trade_date'] = pd.to_datetime(cf['trade_date'])
            cf_series = cf.set_index('trade_date')['main_net'].sort_index()

        divergence = self.decomposer.detect_smart_money_divergence(
            nb_series, margin_series, cf_series
        )

        # 4. 持续性评估 (用全部有效数据)
        valid_nb = nb_series.dropna()
        persistence = self.decomposer.flow_persistence(valid_nb)

        # 5. 北向 vs 沪深300 偏离分析
        print("    北向vs沪深300偏离分析...")
        nb_hs300_divergence = self._analyze_nb_vs_hs300(nb)

        # 6. 行业偏好汇总
        favored = [(k, v) for k, v in sector_pref.items() if v.get('bias') == 'favored']
        avoided = [(k, v) for k, v in sector_pref.items() if v.get('bias') == 'avoided']
        favored.sort(key=lambda x: x[1]['correlation'], reverse=True)
        avoided.sort(key=lambda x: x[1]['correlation'])

        # 最新数据日期(可能滞后)
        tail_valid = valid_nb.tail(60)
        latest_date = str(valid_nb.index[-1].date()) if len(valid_nb) > 0 else None

        return {
            'market_flow': market_flow,
            'sector_preference': {
                'favored': [{'industry': k, **v} for k, v in favored[:8]],
                'avoided': [{'industry': k, **v} for k, v in avoided[:8]],
                'n_industries': len(sector_pref),
            },
            'divergence': divergence,
            'persistence': persistence,
            'nb_hs300_divergence': nb_hs300_divergence,
            'raw_summary': {
                'nb_total_5d': round(float(tail_valid.tail(5).sum()), 2),
                'nb_total_20d': round(float(tail_valid.tail(20).sum()), 2),
                'nb_total_60d': round(float(tail_valid.sum()), 2),
                'latest_nb': round(float(tail_valid.tail(1).iloc[0]), 2) if len(tail_valid) > 0 else 0,
                'latest_date': latest_date,
            },
        }

    def _analyze_market_flow(self, nb):
        """市场级北向资金分析"""
        valid = nb[nb['net_flow'].notna()].copy()
        if len(valid) < 20:
            return {'status': 'insufficient', 'n_days': len(valid)}

        valid = valid.sort_values('trade_date')
        recent = valid.tail(60)

        # 累计净流向趋势
        recent['cum_net_60d'] = recent['net_flow'].cumsum()

        # 月度汇总
        recent['month'] = recent['trade_date'].dt.to_period('M')
        monthly = recent.groupby('month')['net_flow'].sum()

        # 方向信号
        net_5d = recent['net_flow'].tail(5).sum()
        net_20d = recent['net_flow'].tail(20).sum()
        net_60d = recent['cum_net_60d'].iloc[-1]

        # 买卖力度比
        buy_20d = recent['buy_amount'].tail(20).sum()
        sell_20d = recent['sell_amount'].tail(20).sum()
        buy_sell_ratio = buy_20d / sell_20d if sell_20d > 0 else 1

        # 方向判定
        if net_20d > 50:
            direction = 'strong_inflow'
        elif net_20d > 10:
            direction = 'mild_inflow'
        elif net_20d > -10:
            direction = 'neutral'
        elif net_20d > -50:
            direction = 'mild_outflow'
        else:
            direction = 'strong_outflow'

        return {
            'status': 'ok',
            'n_days': len(valid),
            'net_5d': round(float(net_5d), 2),
            'net_20d': round(float(net_20d), 2),
            'net_60d': round(float(net_60d), 2),
            'buy_sell_ratio_20d': round(float(buy_sell_ratio), 3),
            'direction': direction,
            'monthly_summary': {str(k): round(float(v), 2) for k, v in monthly.tail(6).items()},
            'latest_date': str(recent['trade_date'].max().date()),
        }

    def _analyze_nb_vs_hs300(self, nb):
        """北向资金流向 vs 沪深300走势偏离"""
        valid = nb[nb['net_flow'].notna() & nb['hs300_pct'].notna()].copy()
        if len(valid) < 60:
            return {}

        valid = valid.sort_values('trade_date')
        recent = valid.tail(60)

        # 北向累计 vs HS300累计收益
        nb_cum = recent['net_flow'].cumsum()
        hs300_cum = (1 + recent['hs300_pct'] / 100).cumprod()

        # 相关性
        corr = nb_cum.corr(hs300_cum)

        # 背离检测: 北向流入但HS300下跌 (外资抄底) 或 北向流出但HS300上涨 (外资逃顶)
        nb_dir = recent['net_flow'].tail(20).sum()
        hs300_ret = recent['hs300_pct'].tail(20).sum()

        if nb_dir > 0 and hs300_ret < -2:
            divergence_type = '外资抄底: 北向买入但大盘下跌'
        elif nb_dir < 0 and hs300_ret > 2:
            divergence_type = '外资逃顶: 北向卖出但大盘上涨'
        elif nb_dir > 0 and hs300_ret > 0:
            divergence_type = '共振上涨: 北向+大盘同向'
        elif nb_dir < 0 and hs300_ret < 0:
            divergence_type = '共振下跌: 北向+大盘同向'
        else:
            divergence_type = '无明显背离'

        return {
            'correlation_60d': round(float(corr), 3),
            'nb_20d_direction': 'inflow' if nb_dir > 0 else 'outflow',
            'hs300_20d_return': round(float(hs300_ret), 2),
            'divergence_type': divergence_type,
        }

    def report(self, result):
        """生成报告"""
        alerts = []

        market = result.get('market_flow', {})
        divergence = result.get('divergence', {})
        persistence = result.get('persistence', {})
        sector = result.get('sector_preference', {})
        nb_hs300 = result.get('nb_hs300_divergence', {})

        # 告警: 北向外流
        if market.get('net_20d', 0) < -50:
            alerts.append(f"[北向] 20日累计净流出{market['net_20d']:.0f}亿→外资持续撤离")

        # 告警: 北向vs大盘背离
        if '外资逃顶' in nb_hs300.get('divergence_type', ''):
            alerts.append(f"[北向] {nb_hs300['divergence_type']}→警惕外资借涨出货")

        # 告警: 聪明钱分歧
        for sig in divergence.get('divergence_signals', []):
            if sig.get('confidence') == 'high':
                alerts.append(f"[北向] {sig['signal']}: {sig['interpretation']}")

        # 持续性告警
        if persistence.get('consecutive_outflow_days', 0) >= 5:
            alerts.append(f"[北向] 连续{persistence['consecutive_outflow_days']}日净流出→趋势性撤离")

        # 摘要
        raw = result.get('raw_summary', {})
        nb_5d = raw.get('nb_total_5d', 0)
        nb_dir = '流入' if nb_5d > 0 else '流出'
        summary = (
            f"北向5日{nb_dir}{abs(nb_5d):.0f}亿 | "
            f"20日{market.get('net_20d',0):.0f}亿 | "
            f"方向:{market.get('direction','?')} | "
            f"北向vsHS300:{nb_hs300.get('divergence_type','?')} | "
            f"数据截止:{raw.get('latest_date','?')}"
        )

        return {
            'status': 'ok',
            'summary': summary,
            'market_flow': market,
            'sector_preference': sector,
            'divergence': divergence,
            'persistence': persistence,
            'nb_hs300_divergence': nb_hs300,
            'raw_summary': raw,
            'data_status': self.data_status,
            'alerts': alerts,
        }


if __name__ == '__main__':
    agent = NorthboundDecompAgent()
    result = agent.run()
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
