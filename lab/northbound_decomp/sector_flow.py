# -*- coding: utf-8 -*-
"""
北向资金拆解引擎
核心能力: 交易所维度拆解 → 行业偏好推断 → 北向vs融资分歧 → 主力指纹
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class NorthboundDecomposer:
    """北向资金拆解器

    由于北向数据只有交易所汇总(无个股明细), 采用:
    1. 交易所维度: 沪港通vs深港通净流向
    2. 行业推断: 行业指数涨跌与北向净流向的相关性 → 推断外资偏好
    3. 聪明钱分歧: 北向 vs 融资 vs 主力资金 三方分歧度
    4. 持仓集中度: 北向净流向的持续性和集中度
    """

    def __init__(self, lookback_days=60):
        self.lookback_days = lookback_days

    def decompose_exchange(self, northbound_df):
        """
        拆解沪港通vs深港通净流向

        Args:
            northbound_df: 含 ts_code(HSGT.N/HSGT.S), trade_date, net_flow

        Returns:
            dict with sh/sz flows, cumulative, divergence
        """
        if northbound_df.empty:
            return {'status': 'no_data'}

        # 分离沪港通/深港通
        sh_flow = northbound_df[northbound_df['ts_code'] == 'HSGT.N'].copy()
        sz_flow = northbound_df[northbound_df['ts_code'] == 'HSGT.S'].copy()

        if sh_flow.empty or sz_flow.empty:
            return {'status': 'insufficient_data'}

        sh_flow = sh_flow.sort_values('trade_date')
        sz_flow = sz_flow.sort_values('trade_date')

        # 近期数据
        sh_recent = sh_flow.tail(self.lookback_days)
        sz_recent = sz_flow.tail(self.lookback_days)

        sh_net = sh_recent['net_flow'].sum()
        sz_net = sz_recent['net_flow'].sum()
        total_net = sh_net + sz_net

        # 滚动相关性 → 沪港通与深港通是否同步
        merged = sh_recent[['trade_date', 'net_flow']].merge(
            sz_recent[['trade_date', 'net_flow']],
            on='trade_date', suffixes=('_sh', '_sz')
        )
        if len(merged) > 5:
            corr = merged['net_flow_sh'].corr(merged['net_flow_sz'])
        else:
            corr = 0

        # 分歧检测: 若沪港通和深港通方向相反
        sh_direction = 'inflow' if sh_net > 0 else 'outflow'
        sz_direction = 'inflow' if sz_net > 0 else 'outflow'
        diverged = sh_direction != sz_direction

        # 近期趋势
        sh_5d = sh_recent.tail(5)['net_flow'].sum()
        sz_5d = sz_recent.tail(5)['net_flow'].sum()
        sh_20d = sh_recent.tail(20)['net_flow'].sum()
        sz_20d = sz_recent.tail(20)['net_flow'].sum()

        return {
            'status': 'ok',
            '沪港通': {
                'net_5d': round(float(sh_5d), 2),
                'net_20d': round(float(sh_20d), 2),
                'net_60d': round(float(sh_net), 2),
                'direction': sh_direction,
                'avg_daily': round(float(sh_recent['net_flow'].mean()), 2),
            },
            '深港通': {
                'net_5d': round(float(sz_5d), 2),
                'net_20d': round(float(sz_20d), 2),
                'net_60d': round(float(sz_net), 2),
                'direction': sz_direction,
                'avg_daily': round(float(sz_recent['net_flow'].mean()), 2),
            },
            'total_net_60d': round(float(total_net), 2),
            'correlation_sh_sz': round(float(corr), 3),
            'diverged': diverged,
            'interpretation': self._interpret_exchange(
                sh_5d, sz_5d, sh_20d, sz_20d, corr, diverged
            ),
        }

    def _interpret_exchange(self, sh_5d, sz_5d, sh_20d, sz_20d, corr, diverged):
        """解读交易所维度信号"""
        if diverged:
            return '沪港通与深港通方向背离→外资在沪市和深市间切换，信号混乱'
        elif sh_5d > 0 and sz_5d > 0 and sh_20d > 0 and sz_20d > 0:
            return '双通道持续流入→外资积极做多A股，强烈看多信号'
        elif sh_5d < 0 and sz_5d < 0 and sh_20d < 0 and sz_20d < 0:
            return '双通道持续流出→外资撤离A股，强烈看空信号'
        elif sh_5d + sz_5d > 0 > sh_20d + sz_20d:
            return '短期流入但中期流出→可能是短期反弹买盘，不具持续性'
        elif sh_5d + sz_5d < 0 < sh_20d + sz_20d:
            return '短期流出但中期流入→可能是短期获利了结，中期趋势仍好'
        else:
            return '信号混合，方向不明'

    def infer_sector_preference(self, industry_returns, northbound_flow):
        """
        通过行业指数收益与北向流量的相关性推断外资行业偏好

        Args:
            industry_returns: DataFrame with [trade_date, ind_name, daily_return]
            northbound_flow: Series indexed by trade_date, net flow values

        Returns:
            sector_bias: dict mapping industry to correlation with northbound
        """
        if industry_returns.empty:
            return {}

        # 将北向流量对齐到行业收益日期
        nb_series = northbound_flow.copy()
        if isinstance(nb_series, pd.DataFrame):
            nb_series = nb_series.set_index('trade_date')['net_flow']

        sector_bias = {}
        for ind_name, group in industry_returns.groupby('ind_name'):
            merged = group[['trade_date', 'daily_return']].copy()
            merged = merged.set_index('trade_date')
            # 对齐
            common_dates = merged.index.intersection(nb_series.index)
            if len(common_dates) < 20:
                continue

            corr = merged.loc[common_dates, 'daily_return'].corr(
                nb_series.loc[common_dates]
            )
            sector_bias[ind_name] = {
                'correlation': round(corr, 3),
                'n_days': len(common_dates),
                'bias': 'favored' if corr > 0.1 else ('avoided' if corr < -0.1 else 'neutral'),
            }

        return sector_bias

    def detect_smart_money_divergence(self, northbound_net, margin_net, capital_flow_main):
        """
        三方聪明钱分歧检测

        北向(外资) vs 融资(散户杠杆) vs 主力(大单)

        Returns:
            divergence: dict with signals
        """
        signals = []

        # 北向 vs 融资
        nb_5d = northbound_net.tail(5).sum() if len(northbound_net) >= 5 else 0
        margin_5d = margin_net.tail(5).sum() if len(margin_net) >= 5 else 0
        main_5d = capital_flow_main.tail(5).sum() if len(capital_flow_main) >= 5 else 0

        nb_dir = '流入' if nb_5d > 0 else '流出'
        margin_dir = '加杠杆' if margin_5d > 0 else '降杠杆'
        main_dir = '净买' if main_5d > 0 else '净卖'

        # 分歧1: 北向流入 vs 融资降杠杆 → 外资抄底 vs 散户恐慌
        if nb_5d > 0 and margin_5d < 0:
            signals.append({
                'type': '外资vs散户',
                'signal': '北向抄底/散户恐慌',
                'northbound': nb_dir,
                'margin': margin_dir,
                'interpretation': '外资逆势买入，散户割肉→历史上偏向北向方向',
                'confidence': 'medium',
            })
        # 分歧2: 北向流出 vs 融资加杠杆 → 外资撤退 vs 散户追高
        elif nb_5d < 0 and margin_5d > 0:
            signals.append({
                'type': '外资vs散户',
                'signal': '外资撤退/散户追高',
                'northbound': nb_dir,
                'margin': margin_dir,
                'interpretation': '外资获利了结，散户接盘→危险信号',
                'confidence': 'high',
            })

        # 分歧3: 主力 vs 北向
        if main_5d > 0 and nb_5d < 0:
            signals.append({
                'type': '主力vs外资',
                'signal': '主力买/外资卖',
                'northbound': nb_dir,
                'main_force': main_dir,
                'interpretation': '国内主力接盘外资抛售→关注后续主力是否持续',
                'confidence': 'medium',
            })
        elif main_5d < 0 and nb_5d > 0:
            signals.append({
                'type': '主力vs外资',
                'signal': '主力卖/外资买',
                'northbound': nb_dir,
                'main_force': main_dir,
                'interpretation': '外资接盘主力出货→警惕主力借外资掩护撤退',
                'confidence': 'high',
            })

        # 三方共识检测
        if nb_5d > 0 and margin_5d > 0 and main_5d > 0:
            consensus = '三方共振做多→强烈看多'
        elif nb_5d < 0 and margin_5d < 0 and main_5d < 0:
            consensus = '三方共振做空→强烈看空'
        else:
            consensus = '分歧→方向不明'

        return {
            'status': 'ok',
            'divergence_signals': signals,
            'consensus': consensus,
            'summary': {
                'northbound_5d': round(float(nb_5d), 2),
                'margin_5d': round(float(margin_5d), 2),
                'main_force_5d': round(float(main_5d), 2),
            },
            'n_divergences': len(signals),
        }

    def flow_persistence(self, northbound_series):
        """
        计算北向资金流的持续性指标

        Returns:
            {连续流入天数, 连续流出天数, 波动率, 趋势强度}
        """
        if len(northbound_series) < 5:
            return {}

        values = northbound_series.values if hasattr(northbound_series, 'values') else northbound_series

        # 连续同向天数
        streak_in = 0
        streak_out = 0
        for v in values[::-1]:  # 从最新往前
            if v > 0:
                if streak_out == 0:
                    streak_in += 1
                else:
                    break
            elif v < 0:
                if streak_in == 0:
                    streak_out += 1
                else:
                    break
            else:
                break

        # 波动率
        vol = np.std(values) if len(values) > 1 else 0
        mean_flow = np.mean(values)

        # 趋势强度: mean / std → 类似信息比率
        trend_strength = mean_flow / vol if vol > 0 else 0

        # 方向一致性: 同向天数占比
        same_dir_ratio = max(
            (values > 0).mean(),
            (values < 0).mean()
        )

        return {
            'consecutive_inflow_days': streak_in,
            'consecutive_outflow_days': streak_out,
            'daily_volatility': round(float(vol), 2),
            'mean_daily_flow': round(float(mean_flow), 2),
            'trend_strength': round(float(trend_strength), 3),
            'direction_consistency': round(float(same_dir_ratio), 2),
        }
