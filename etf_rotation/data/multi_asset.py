# -*- coding: utf-8 -*-
"""
多资产防御池: 国债 + 黄金 + 纳指
熊市时替代空仓, 实现"主动切换大类资产"
"""
import pandas as pd
import numpy as np
import akshare as ak
import warnings
warnings.filterwarnings('ignore')


def build_bond_returns():
    """
    中债10年期国债收益率 → 近似日收益
    Bond return ≈ -duration × Δyield  (duration≈8 for 10Y)
    """
    df = ak.bond_zh_us_rate()
    # 第1列=日期, 第4列=10年期 (列顺序: 日期,2年,5年,10年,30年)
    cols = list(df.columns)
    date_col = cols[0]
    y10_col = cols[3]  # 10年期国债收益率

    df = df.rename(columns={date_col: 'trade_date', y10_col: 'yield_10y'})
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df[['trade_date', 'yield_10y']].copy()
    df['yield_10y'] = pd.to_numeric(df['yield_10y'], errors='coerce')
    df = df.dropna().sort_values('trade_date')

    dy = df['yield_10y'].diff() / 100
    df['ret'] = -8.0 * dy
    df = df.set_index('trade_date')

    return df[['ret']].copy()


def build_gold_returns():
    """COMEX黄金期货日收益 (2006起)"""
    df = ak.futures_foreign_hist(symbol='XAU')
    df['trade_date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('trade_date')
    df['ret'] = df['close'].pct_change()
    df = df.set_index('trade_date')
    return df[['ret']].copy()


def build_nasdaq_returns():
    """纳斯达克综合指数日收益 (2004起)"""
    df = ak.index_us_stock_sina(symbol='.IXIC')
    df['trade_date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close']).sort_values('trade_date')
    df['ret'] = df['close'].pct_change()
    df = df.set_index('trade_date')
    return df[['ret']].copy()


def build_defense_pool():
    """
    构建防御资产池日收益矩阵
    Returns: DataFrame [date x asset] 日收益
    """
    print('  加载国债收益率...')
    bond = build_bond_returns()
    print(f'    国债: {len(bond)}天, {bond.index[0].date()} ~ {bond.index[-1].date()}')

    print('  加载COMEX黄金...')
    gold = build_gold_returns()
    print(f'    黄金: {len(gold)}天, {gold.index[0].date()} ~ {gold.index[-1].date()}')

    print('  加载纳斯达克...')
    nasdaq = build_nasdaq_returns()
    print(f'    纳指: {len(nasdaq)}天, {nasdaq.index[0].date()} ~ {nasdaq.index[-1].date()}')

    # 合并
    defense = pd.DataFrame(index=bond.index)
    defense['bond'] = bond['ret']
    defense['gold'] = gold['ret'].reindex(defense.index)
    defense['nasdaq'] = nasdaq['ret'].reindex(defense.index)

    # 早期(2002-2004)没有黄金和纳指, 用债券填补
    defense['gold'] = defense['gold'].fillna(defense['bond'])
    defense['nasdaq'] = defense['nasdaq'].fillna(defense['bond'])

    defense = defense.dropna()
    print(f'  合并: {len(defense)}天, {defense.index[0].date()} ~ {defense.index[-1].date()}')

    return defense


def defense_portfolio_return(defense_rets, weights=None):
    """
    防御组合日收益 (加权)
    默认: 债券40% + 黄金30% + 纳指30%
    """
    if weights is None:
        weights = {'bond': 0.40, 'gold': 0.30, 'nasdaq': 0.30}

    port = pd.Series(0.0, index=defense_rets.index)
    for asset, w in weights.items():
        if asset in defense_rets.columns:
            port += defense_rets[asset].fillna(0) * w
    return port
