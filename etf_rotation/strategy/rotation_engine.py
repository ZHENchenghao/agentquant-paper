# -*- coding: utf-8 -*-
"""
轮动引擎: 多因子综合打分 → 行业选择 → 月度调仓
"""
import pandas as pd
import numpy as np


class RotationEngine:
    """ETF轮动策略引擎

    工作流:
    1. 每月末计算各因子得分
    2. 因子关联检测 → 动态调整权重
    3. 综合打分 → Top-N行业
    4. 仓位分配
    """

    def __init__(self, top_n=5, rebalance_freq='M'):
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.default_weights = {
            'mom_21d': 1.5, 'mom_63d': 2.0, 'mom_126d': 1.5, 'mom_252d': 1.0,
            'mom_cum_21d': 1.0, 'mom_cum_63d': 1.5, 'mom_cum_126d': 1.0, 'mom_cum_252d': 0.5,
            'vol_21d': 1.5, 'vol_63d': 1.5,
            'downside_vol_63d': 1.0, 'max_dd_63d': 1.0,
            'sharpe_63d': 1.5, 'stability_63d': 1.0,
        }

    def set_weights(self, weights):
        self.default_weights = weights

    def adjust_weights_by_correlation(self, crowding_index):
        """
        根据因子拥挤度动态调整权重:
        - 拥挤度 > 0.7 → 降动量权重, 升低波权重 (因子拥挤→动量可能反转)
        - 拥挤度 < 0.3 → 等权 (因子独立, 各维度都有信息)
        """
        weights = self.default_weights.copy()
        if crowding_index is None or pd.isna(crowding_index):
            return weights

        if crowding_index > 0.7:
            # 因子高度拥挤 → 降低动量依赖
            for k in weights:
                if 'mom' in k:
                    weights[k] *= 0.5
                elif 'vol' in k or 'dd' in k:
                    weights[k] *= 1.5
        elif crowding_index > 0.5:
            for k in weights:
                if 'mom' in k:
                    weights[k] *= 0.8

        return weights

    def select(self, normed_factors, crowding_index=None, weights=None,
               fragile_mask=None):
        """月度选ETF, 脆牛期限防御行业"""
        DEFENSIVE = {'银行','公用事业','食品饮料','交通运输','医药生物','家用电器','非银金融','建筑装饰'}

        if weights is None:
            weights = self.adjust_weights_by_correlation(crowding_index)

        ref_idx = normed_factors[list(normed_factors.keys())[0]].index
        ref_cols = normed_factors[list(normed_factors.keys())[0]].columns
        score = pd.DataFrame(0.0, index=ref_idx, columns=ref_cols)

        for name, df in normed_factors.items():
            w = weights.get(name, 0)
            if w != 0 and name in normed_factors:
                score = score.add(df.fillna(0.5) * w, fill_value=0)

        monthly_score = score.resample('M').last()
        selections = {}

        for date in monthly_score.index:
            row = monthly_score.loc[date].dropna()
            if len(row) < self.top_n:
                continue

            is_fragile = False
            if fragile_mask is not None:
                mask_dates = fragile_mask.index[fragile_mask.index <= date]
                if len(mask_dates) > 0:
                    is_fragile = fragile_mask.loc[mask_dates[-1]]

            if is_fragile:
                def_avail = [c for c in row.index if c in DEFENSIVE]
                if len(def_avail) >= self.top_n:
                    row = row[def_avail]

            selections[date] = row.nlargest(self.top_n).index.tolist()

        return selections, monthly_score

    def compute_weights(self, selections, return_matrix,
                        vol_target=0.20, vol_lookback=60, vol_floor=0.10):
        """
        波动率目标仓位缩放 + 风险平价加权

        Args:
            selections: {date: [top etf codes]}
            return_matrix: 日收益矩阵
            vol_target: 目标年化波动率 (0.20=20%)
            vol_lookback: 波动率回看天数
            vol_floor: 最低波动率 (防止放大)

        Returns:
            weights: {date: {etf_code: weight}}  总仓位可能<1(部分现金)
        """
        weights = {}

        for date, etfs in selections.items():
            n = len(etfs)
            etf_rets = {}
            for e in etfs:
                if e in return_matrix.columns:
                    past = return_matrix[e].loc[:date].tail(vol_lookback).dropna()
                    if len(past) >= 20:
                        etf_rets[e] = past
                    else:
                        etf_rets[e] = None

            if not etf_rets:
                continue

            # Step 1: 风险平价 (1/σ 加权)
            vol_weights = {}
            total_inv_vol = 0
            vols = {}
            for e, past in etf_rets.items():
                if past is not None and past.std() > 0:
                    v = past.std() * np.sqrt(252)  # 年化波动率
                    vols[e] = v
                else:
                    vols[e] = vol_target
                inv_vol = 1.0 / max(vols[e], vol_floor)
                vol_weights[e] = inv_vol
                total_inv_vol += inv_vol

            for e in vol_weights:
                vol_weights[e] /= total_inv_vol  # 归一化到总和=1

            # Step 2: 波动率目标缩放
            # 计算Top-N组合的下行波动率 (Sortino风格: 只罚下跌波动)
            top_rets = pd.DataFrame({
                e: return_matrix[e].loc[:date].tail(vol_lookback).dropna()
                for e in etfs if e in return_matrix.columns
            })
            if len(top_rets.columns) >= 2 and len(top_rets) >= 20:
                eq_ret = top_rets.mean(axis=1)
                # 下行波动率: 只看负收益日
                down_rets = eq_ret[eq_ret < 0]
                if len(down_rets) >= 10:
                    down_vol = down_rets.std() * np.sqrt(252)
                else:
                    down_vol = eq_ret.std() * np.sqrt(252) * 0.7  # fallback
                # 组合相关调整
                corr = top_rets.corr().values
                upper = corr[np.triu_indices_from(corr, k=1)]
                avg_corr = np.mean(np.abs(upper)) if len(upper) > 0 else 0.5
                avg_vol = np.mean(list(vols.values()))
                pred_vol = avg_vol * np.sqrt(1/n + (1-1/n)*avg_corr)
                # 混合: 70%下行波动 + 30%整体波动 (避免过于激进)
                pred_vol = 0.7 * down_vol + 0.3 * pred_vol
            else:
                pred_vol = vol_target

            # 仓位缩放 (可选, 默认关闭)
            k = 1.0  # = 不缩仓位, 只做风险平价
            # k = min(1.0, vol_target / max(pred_vol, vol_floor))

            # 最终权重 (风险平价加权, 总和=1)
            weights[date] = {e: vol_weights.get(e, 1/n) * k for e in etfs}

        return weights
