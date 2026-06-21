# -*- coding: utf-8 -*-
"""
QuantLab 共享层: 因子数据加载器
基于 factors_2002.parquet (1.1GB, 2002年至今全A股因子)
"""
import pandas as pd
import numpy as np

FACTOR_PATH = 'D:/AgentQuant/our/cache/factors_2002.parquet'


def load_factors(start_date=None, end_date=None, ts_codes=None):
    """加载因子数据，支持日期和股票筛选"""
    df = pd.read_parquet(FACTOR_PATH)
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    if start_date:
        df = df[df['trade_date'] >= start_date]
    if end_date:
        df = df[df['trade_date'] <= end_date]
    if ts_codes:
        df = df[df['ts_code'].isin(ts_codes)]

    return df


def get_factor_columns(df):
    """提取纯因子列名（排除元数据列）"""
    exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k',
               'report_date', 'excess_ret', 'ind_name', 'mcap', 'board',
               'mkt_sent_5d', 'sent_20d']
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def compute_ic(df, factor_col, target_col='excess_ret', method='rank'):
    """
    计算单因子IC (Information Coefficient)
    method='rank' → Rank IC (Spearman)
    method='pearson' → Pearson IC
    """
    if target_col not in df.columns:
        return None

    valid = df[[factor_col, target_col]].dropna()
    if len(valid) < 30:
        return None

    if method == 'rank':
        return valid[factor_col].rank().corr(valid[target_col].rank())
    else:
        return valid[factor_col].corr(valid[target_col])


def compute_rolling_ic(df, factor_col, target_col='excess_ret',
                       freq='M', min_obs=30):
    """
    计算滚动IC序列
    返回 DataFrame: [period, ic, n_obs]
    """
    df = df.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    if freq == 'M':
        df['period'] = df['trade_date'].dt.to_period('M')
    elif freq == 'W':
        df['period'] = df['trade_date'].dt.to_period('W')
    else:
        df['period'] = df['trade_date'].dt.to_period('D')

    results = []
    for period, group in df.groupby('period'):
        ic = compute_ic(group, factor_col, target_col)
        if ic is not None:
            results.append({
                'period': str(period),
                'ic': ic,
                'n_obs': len(group.dropna(subset=[factor_col, target_col]))
            })

    return pd.DataFrame(results)
