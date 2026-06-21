# -*- coding: utf-8 -*-
"""
因子IC监控引擎
核心能力: 滚动IC计算 → 方向翻转检测 → 集体翻牌告警
"""
import numpy as np
import pandas as pd
from scipy import stats


class ICMonitor:
    """因子IC滚动监控器

    检测三类信号:
    1. 单因子翻牌: 单个因子IC方向翻转 (正→负 或 负→正)
    2. 集体翻牌: N个因子同时翻牌 → 市场定价逻辑切换
    3. IC衰减: 因子IC绝对值持续下降 → 因子失效预警
    """

    def __init__(self, window_months=12, flip_threshold=0.3,
                 collective_threshold=0.25, min_obs=30):
        """
        Args:
            window_months: 滚动窗口月数
            flip_threshold: 单因子翻牌阈值，IC变化超过此值视为翻牌
            collective_threshold: 集体翻牌阈值，超过此比例的因子翻牌视为集体翻牌
            min_obs: 最小观测数
        """
        self.window_months = window_months
        self.flip_threshold = flip_threshold
        self.collective_threshold = collective_threshold
        self.min_obs = min_obs

    def compute_factor_ic_series(self, df, factor_cols, target='excess_ret',
                                  freq='M'):
        """
        对所有因子计算滚动IC序列

        Args:
            df: 含因子+收益率的面板数据
            factor_cols: 因子列名列表
            target: 目标列名
            freq: 'M'月频 或 'W'周频

        Returns:
            ic_df: pivot table, index=period, columns=factor, values=IC
        """
        df = df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        if freq == 'M':
            df['period'] = df['trade_date'].dt.to_period('M')
        elif freq == 'W':
            df['period'] = df['trade_date'].dt.to_period('W')
        else:
            df['period'] = df['trade_date']

        records = []
        for period, group in df.groupby('period'):
            if len(group) < self.min_obs:
                continue
            for col in factor_cols:
                valid = group[[col, target]].dropna()
                if len(valid) < self.min_obs:
                    continue
                ic = valid[col].rank().corr(valid[target].rank())
                records.append({
                    'period': str(period),
                    'factor': col,
                    'ic': ic,
                    'n': len(valid)
                })

        ic_df = pd.DataFrame(records)
        if len(ic_df) == 0:
            return pd.DataFrame()

        return ic_df.pivot(index='period', columns='factor', values='ic')

    def detect_flips(self, ic_pivot):
        """
        检测因子方向翻转

        翻转判定: IC符号从正变负或从负变正

        Returns:
            flips: DataFrame with [factor, period, prev_ic, cur_ic, flip_direction]
        """
        if ic_pivot.empty or len(ic_pivot) < 2:
            return pd.DataFrame()

        flips = []
        factors = ic_pivot.columns

        for factor in factors:
            ic_series = ic_pivot[factor].dropna()
            if len(ic_series) < 2:
                continue

            for i in range(1, len(ic_series)):
                prev = ic_series.iloc[i - 1]
                cur = ic_series.iloc[i]
                period = ic_series.index[i]
                prev_period = ic_series.index[i - 1]

                # 符号翻转检测
                if prev > 0 and cur < -self.flip_threshold:
                    flips.append({
                        'factor': factor,
                        'period': period,
                        'prev_period': prev_period,
                        'prev_ic': round(prev, 4),
                        'cur_ic': round(cur, 4),
                        'direction': 'positive_to_negative',
                        'severity': 'bearish',
                        'delta_ic': round(cur - prev, 4),
                    })
                elif prev < 0 and cur > self.flip_threshold:
                    flips.append({
                        'factor': factor,
                        'period': period,
                        'prev_period': prev_period,
                        'prev_ic': round(prev, 4),
                        'cur_ic': round(cur, 4),
                        'direction': 'negative_to_positive',
                        'severity': 'bullish',
                        'delta_ic': round(cur - prev, 4),
                    })

                # IC加速衰减检测 (同向但加速恶化)
                elif prev > 0 and cur > 0 and cur < prev * 0.5:
                    flips.append({
                        'factor': factor,
                        'period': period,
                        'prev_period': prev_period,
                        'prev_ic': round(prev, 4),
                        'cur_ic': round(cur, 4),
                        'direction': 'positive_decaying',
                        'severity': 'warning',
                        'delta_ic': round(cur - prev, 4),
                    })

        return pd.DataFrame(flips)

    def detect_collective_flip(self, flips_df, ic_pivot):
        """
        检测集体翻牌: 超过collective_threshold比例的因子在同一期翻牌

        Returns:
            collective: list of {period, flip_count, total_factors, ratio, direction, alerts}
        """
        if flips_df.empty:
            return []

        total_factors = len(ic_pivot.columns)
        collective = []

        for period, group in flips_df.groupby('period'):
            n_bearish = (group['severity'] == 'bearish').sum()
            n_bullish = (group['severity'] == 'bullish').sum()
            n_warning = (group['severity'] == 'warning').sum()
            total_flips = len(group)

            ratio = total_flips / total_factors

            if ratio >= self.collective_threshold:
                # 判断主导方向
                if n_bearish > n_bullish:
                    dominant = 'bearish'
                elif n_bullish > n_bearish:
                    dominant = 'bullish'
                else:
                    dominant = 'mixed'

                collective.append({
                    'period': period,
                    'flip_count': total_flips,
                    'total_factors': total_factors,
                    'ratio': round(ratio, 3),
                    'bearish_count': n_bearish,
                    'bullish_count': n_bullish,
                    'warning_count': n_warning,
                    'dominant_direction': dominant,
                    'factors': group['factor'].tolist(),
                })

        return collective

    def compute_ic_stability(self, ic_pivot, lookback=6):
        """
        计算IC稳定性得分
        - IC符号一致性: 过去N期内同向比例
        - IC波动率: IC的标准差
        """
        if ic_pivot.empty:
            return {}

        recent = ic_pivot.tail(lookback)
        stability = {}

        for factor in recent.columns:
            ic_series = recent[factor].dropna()
            if len(ic_series) < 3:
                continue

            positive_ratio = (ic_series > 0).mean()
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            # 稳定性 = 方向一致度 * (1 - 变异系数)
            cv = abs(ic_std / ic_mean) if ic_mean != 0 else 999
            consistency = abs(positive_ratio - 0.5) * 2  # 0=完全随机, 1=完全一致

            stability[factor] = {
                'mean_ic': round(ic_mean, 4),
                'std_ic': round(ic_std, 4),
                'positive_ratio': round(positive_ratio, 2),
                'direction_consistency': round(consistency, 2),
                'stability_score': round(consistency * max(0, 1 - min(cv, 1)), 3),
            }

        return stability

    def generate_signal(self, collective_flips, stability, latest_ic):
        """
        根据集体翻牌和稳定性生成仓位信号

        Returns:
            signal: dict with action, confidence, reason
        """
        if not collective_flips:
            # 检查稳定性是否恶化
            low_stability = [k for k, v in stability.items()
                             if v.get('stability_score', 1) < 0.3]
            if len(low_stability) > len(stability) * 0.3:
                return {
                    'action': 'CAUTION',
                    'confidence': 'medium',
                    'reason': f'因子IC稳定性普遍下降 ({len(low_stability)}/{len(stability)}个因子)',
                    'suggested_action': '减仓10-20%，观察因子结构',
                }
            return {
                'action': 'NORMAL',
                'confidence': 'high',
                'reason': '因子IC结构稳定，无集体翻牌',
                'suggested_action': '维持正常仓位',
            }

        latest = collective_flips[-1]
        ratio = latest['ratio']
        direction = latest['dominant_direction']

        if direction == 'bearish' and ratio >= 0.4:
            return {
                'action': 'REDUCE',
                'confidence': 'high' if ratio >= 0.5 else 'medium',
                'reason': f"因子集体向空翻牌: {latest['flip_count']}/{latest['total_factors']} "
                          f"({ratio*100:.0f}%)个因子IC转负",
                'suggested_action': '降仓30-50%，切换至防御模式',
                'flip_factors': latest['factors'],
            }
        elif direction == 'bearish' and ratio >= 0.25:
            return {
                'action': 'CAUTION',
                'confidence': 'medium',
                'reason': f"因子部分向空翻牌: {latest['flip_count']}/{latest['total_factors']}个因子IC转负",
                'suggested_action': '降仓10-20%，收紧止损',
                'flip_factors': latest['factors'],
            }
        elif direction == 'bullish':
            return {
                'action': 'NORMAL',
                'confidence': 'medium',
                'reason': f"因子集体向多翻牌: 可能是新主线形成",
                'suggested_action': '维持仓位，观察新因子结构持续性',
            }

        return {
            'action': 'NORMAL',
            'confidence': 'low',
            'reason': '因子翻牌信号混杂，方向不明',
            'suggested_action': '观望，不操作',
        }
