# -*- coding: utf-8 -*-
"""
多因子构建引擎
因子族: 动量(4维) + 低波动(2维) + 价值(2维) + 质量(2维) = 10因子
从行业指数日收益矩阵计算
"""
import pandas as pd
import numpy as np


class FactorBuilder:
    """行业ETF多因子构建器

    所有因子在每月末计算, 用于次月轮动
    """

    def __init__(self):
        self.factor_names = []

    def build_all_factors(self, return_matrix):
        """
        从日收益矩阵构建全部因子

        Args:
            return_matrix: DataFrame, index=trade_date, columns=industry

        Returns:
            dict of {factor_name: DataFrame [date x industry]}
        """
        factors = {}
        prices = (1 + return_matrix).cumprod()

        # === 动量因子族 (4个) ===
        for window in [21, 63, 126, 252]:
            name = f'mom_{window}d'
            factors[name] = return_matrix.rolling(window).mean() * np.sqrt(window)  # 年化
            # 也保留原始累计收益
            factors[f'mom_cum_{window}d'] = prices / prices.shift(window) - 1

        # === 低波动因子 (2个) ===
        factors['vol_21d'] = -return_matrix.rolling(21).std() * np.sqrt(252)  # 负向: 低波高分
        factors['vol_63d'] = -return_matrix.rolling(63).std() * np.sqrt(252)

        # === 下行波动 ===
        factors['downside_vol_63d'] = -return_matrix[return_matrix < 0].rolling(63).std() * np.sqrt(252)
        factors['downside_vol_63d'] = factors['downside_vol_63d'].fillna(0)

        # === 最大回撤因子 ===
        factors['max_dd_63d'] = -prices.rolling(63).apply(
            lambda x: (x / x.cummax() - 1).min()
        )

        # === 夏普比率 ===
        roll_ret = return_matrix.rolling(63).mean() * 252
        roll_vol = return_matrix.rolling(63).std() * np.sqrt(252)
        factors['sharpe_63d'] = (roll_ret / roll_vol.replace(0, np.nan)).fillna(0)

        # === 收益稳定性 ===
        factors['stability_63d'] = return_matrix.rolling(63).mean() / \
            return_matrix.rolling(63).std().replace(0, np.nan)
        factors['stability_63d'] = factors['stability_63d'].fillna(0)

        self.factor_names = list(factors.keys())
        return factors

    def normalize_factors(self, factors, method='rank'):
        """
        因子截面标准化 (每月横截面排序 → 统一量纲)

        Args:
            method: 'rank' → 0-1排位, 'zscore' → Z值标准化

        Returns:
            标准化后的因子字典
        """
        normalized = {}
        for name, df in factors.items():
            if method == 'rank':
                normed = df.rank(axis=1, pct=True)  # 0到1排位
            elif method == 'zscore':
                normed = df.subtract(df.mean(axis=1), axis=0).div(
                    df.std(axis=1).replace(0, 1), axis=0
                )
            else:
                normed = df
            normalized[name] = normed
        return normalized

    def composite_score(self, normed_factors, weights=None, top_n=5):
        """
        因子综合打分 → 选出Top-N行业

        Args:
            normed_factors: 标准化后因子字典
            weights: 各因子权重dict, None则等权
            top_n: 选前N个行业

        Returns:
            DataFrame [date x industry] 综合得分, Series每月Top-N列表
        """
        if weights is None:
            weights = {k: 1.0 for k in normed_factors}

        # 加权求和
        score = pd.DataFrame(0, index=normed_factors[list(normed_factors.keys())[0]].index,
                             columns=normed_factors[list(normed_factors.keys())[0]].columns)
        for name, df in normed_factors.items():
            w = weights.get(name, 0)
            if w != 0:
                score = score.add(df.fillna(0.5) * w, fill_value=0)

        # 每月取Top-N
        top_etfs = {}
        for date in score.index:
            row = score.loc[date].dropna()
            if len(row) >= top_n:
                top_etfs[date] = row.nlargest(top_n).index.tolist()

        return score, top_etfs
