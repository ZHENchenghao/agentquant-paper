# -*- coding: utf-8 -*-
"""
多因子关联检测引擎
检测: 因子拥挤度 / 因子IC衰减 / 因子族内相关性 / 因子状态切换
"""
import pandas as pd
import numpy as np
from scipy import stats


class FactorCorrelation:
    """多因子关联检测器

    三个检测维度:
    1. 因子间相关矩阵 → 拥挤度 = mean(|corr|)
    2. 因子IC滚动 → 方向一致性 = 同类因子同向比例
    3. 因子收益率 → 多空收益差
    """

    def __init__(self, window_months=12):
        self.window = window_months

    def pairwise_correlation(self, normed_factors):
        """
        计算因子间截面相关性矩阵 (每月)

        Args:
            normed_factors: 标准化因子字典

        Returns:
            corr_df: DataFrame [date x ('factor1_vs_factor2')]
            crowding_idx: Series [date] 因子拥挤度
        """
        all_corr = {}
        crowding = {}

        # 获取公有日期
        common_dates = None
        for df in normed_factors.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates & set(df.index)

        factors = list(normed_factors.keys())
        n_factors = len(factors)

        for date in sorted(common_dates):
            # 当天各因子在各行业的得分矩阵 [factors x industries]
            # 先对齐行业索引, 避免shape不一致
            all_industries = None
            for f in factors:
                df = normed_factors[f]
                if date in df.index:
                    row = df.loc[date].dropna()
                    if all_industries is None:
                        all_industries = set(row.index)
                    else:
                        all_industries = all_industries & set(row.index)

            if all_industries is None or len(all_industries) < 5:
                continue

            common_inds = sorted(all_industries)
            scores = []
            valid_factors = []
            for f in factors:
                df = normed_factors[f]
                if date in df.index:
                    row = df.loc[date][common_inds]
                    if row.notna().sum() >= 5:
                        scores.append(row.fillna(0.5).values)
                        valid_factors.append(f)

            if len(scores) < 3:
                continue

            score_arr = np.array([s for s in scores])
            corr_matrix = np.corrcoef(score_arr)
            # 拥挤度 = 所有因子对相关系数绝对值均值
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            crowding[date] = np.mean(np.abs(upper_tri))

            # 记录各因子对相关系数
            for i in range(len(valid_factors)):
                for j in range(i + 1, len(valid_factors)):
                    pair = f'{valid_factors[i]}_vs_{valid_factors[j]}'
                    if pair not in all_corr:
                        all_corr[pair] = {}
                    all_corr[pair][date] = corr_matrix[i, j]

        crowding_series = pd.Series(crowding, name='crowding_index')
        corr_df = pd.DataFrame(all_corr)
        corr_df.index.name = 'date'

        return corr_df, crowding_series

    def factor_ic_trend(self, factor_scores, returns, freq='M'):
        """
        计算各因子IC(信息系数)滚动序列

        因子IC = 每月末因子得分与下月行业收益的Rank相关

        Args:
            factor_scores: dict {factor_name: DataFrame [date x industry]}
            returns: 日收益矩阵

        Returns:
            ic_df: DataFrame [period x factor]
        """
        # 构建月度收益
        monthly_ret = returns.resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )

        ic_records = []
        for name, scores in factor_scores.items():
            for i, period in enumerate(monthly_ret.index):
                if i == 0:
                    continue
                prev_period = monthly_ret.index[i - 1]
                # 用上月末因子得分预测本月收益
                score_dates = scores.index[scores.index <= prev_period]
                if len(score_dates) == 0:
                    continue
                score_date = score_dates[-1]

                score_row = scores.loc[score_date].dropna()
                ret_row = monthly_ret.loc[period].dropna()

                common = score_row.index.intersection(ret_row.index)
                if len(common) < 5:
                    continue

                ic = stats.spearmanr(
                    score_row[common].rank(),
                    ret_row[common].rank()
                )[0]

                ic_records.append({
                    'period': period,
                    'factor': name,
                    'ic': ic,
                    'n': len(common),
                })

        if not ic_records:
            return pd.DataFrame()

        ic_df = pd.DataFrame(ic_records)
        return ic_df.pivot(index='period', columns='factor', values='ic')

    def factor_group_correlation(self, ic_df, normed_factors, factor_groups=None):
        """
        检测因子族内部的集团相关性

        factor_groups: dict, e.g. {
            'momentum': ['mom_21d', 'mom_63d', 'mom_126d', 'mom_252d'],
            'low_vol': ['vol_21d', 'vol_63d', 'downside_vol_63d'],
            'quality': ['sharpe_63d', 'stability_63d'],
        }
        """
        if factor_groups is None:
            factor_groups = {
                'momentum': [f for f in ic_df.columns if 'mom' in f],
                'low_vol': [f for f in ic_df.columns if 'vol' in f or 'dd' in f.lower()],
                'quality': [f for f in ic_df.columns if 'sharpe' in f or 'stability' in f],
            }

        group_consistency = {}
        for group_name, factors in factor_groups.items():
            valid = [f for f in factors if f in ic_df.columns]
            if len(valid) < 2:
                continue

            group_ic = ic_df[valid].dropna()
            if len(group_ic) < 3:
                continue

            # 组内因子同向性
            signs = np.sign(group_ic)
            consistency = (signs.sum(axis=1).abs() / len(valid)).mean()
            group_consistency[group_name] = {
                'consistency': round(float(consistency), 3),
                'n_factors': len(valid),
                'factors': valid,
            }

        return group_consistency

    def detect_factor_regime(self, ic_df, lookback=12):
        """
        检测因子状态切换

        Returns:
            regime: dict {factor: {current_trend, flip_risk, stability}}
        """
        if ic_df.empty or len(ic_df) < lookback:
            return {}

        regime = {}
        for factor in ic_df.columns:
            series = ic_df[factor].dropna()
            if len(series) < lookback:
                continue

            recent = series.tail(lookback)
            # 趋势: 最近6月IC均值 vs 前6月
            first_half = recent.head(lookback // 2).mean()
            second_half = recent.tail(lookback // 2).mean()

            # 翻转风险: 符号不稳定
            sign_changes = (np.sign(recent).diff() != 0).sum()
            flip_risk = sign_changes / len(recent)

            regime[factor] = {
                'ic_mean': round(float(recent.mean()), 4),
                'ic_std': round(float(recent.std()), 4),
                'trend': 'improving' if second_half > first_half else 'decaying',
                'flip_risk': round(float(flip_risk), 2),
                'stable': flip_risk < 0.3 and abs(recent.mean()) > recent.std() * 0.5,
            }

        return regime
