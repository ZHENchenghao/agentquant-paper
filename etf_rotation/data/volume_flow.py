# -*- coding: utf-8 -*-
"""
行业成交量聚合 + 资金流代理因子
从kline_daily聚合30个申万行业的日成交量 → 构建资金流信号
"""
import duckdb
import pandas as pd
import numpy as np

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def build_industry_volume(start='2002-01-01', end='2026-06-19'):
    """
    从个股K线聚合到行业日成交量

    返回: DataFrame [trade_date x industry] 日成交量总和
    """
    con = duckdb.connect(DB, read_only=True)

    # 行业映射 + K线 内联结, 聚合成交量
    sql = f"""
        SELECT k.trade_date, m.ind_name AS industry,
               SUM(k.vol) AS total_vol,
               SUM(k.amount) AS total_amount,
               COUNT(DISTINCT k.ts_code) AS n_stocks
        FROM kline_daily k
        JOIN (
            SELECT ts_code, ind_name FROM (
                SELECT ts_code, ind_name,
                       ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
                FROM stock_industry_map
            ) WHERE rn = 1
        ) m ON k.ts_code = m.ts_code
        WHERE k.trade_date BETWEEN '{start}' AND '{end}'
        GROUP BY k.trade_date, m.ind_name
        ORDER BY m.ind_name, k.trade_date
    """

    df = con.execute(sql).df()
    con.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])

    # 透视
    vol_pivot = df.pivot(index='trade_date', columns='industry', values='total_vol')
    amt_pivot = df.pivot(index='trade_date', columns='industry', values='total_amount')
    n_pivot = df.pivot(index='trade_date', columns='industry', values='n_stocks')

    return vol_pivot, amt_pivot, n_pivot


def build_flow_factors(vol_pivot, ret_matrix):
    """
    A股行业成交量资金流因子 (学术验证版)

    方向修正: A股行业ETF资金流是反转因子!
    - 大量流入 → 后续收益差 (散户追涨)
    - 大量流出 → 后续收益好 (恐慌出清)
    - Ref: 国盛证券 ETF资金流反转 (2025), 渤海证券量价因子 (IR 0.893)

    5因子:
    - flow_reversal: 量增=负信号 (反转)
    - flow_overheat: 短期量>长期量=过度关注=反转
    - flow_vol_stability: 量波动率低=筹码稳=看涨
    - flow_amihud: 低价格冲击=抛压小=看涨
    - flow_diverge: 跌放量+涨缩量=危险
    """
    factors = {}

    common_dates = vol_pivot.index.intersection(ret_matrix.index)
    common_cols = vol_pivot.columns.intersection(ret_matrix.columns)
    vol = vol_pivot.reindex(index=common_dates, columns=common_cols).fillna(0)
    ret = ret_matrix.reindex(index=common_dates, columns=common_cols).fillna(0)

    # 1. 资金流反转: 量增=负信号 (A股ETF核心发现)
    vol_ma20 = vol.rolling(20).mean()
    vol_ratio = (vol / vol_ma20.replace(0, np.nan)).clip(0.2, 5.0)
    factors['flow_reversal'] = -vol_ratio

    # 2. 量能过热: 短量>长量=过度关注
    vol_ma5 = vol.rolling(5).mean()
    vol_ma60 = vol.rolling(60).mean()
    factors['flow_overheat'] = -(vol_ma5 / vol_ma60.replace(0, np.nan)).clip(0.3, 3.0)

    # 3. 成交量波动率: 低波动=筹码稳定 (渤海IR 0.893)
    vol_std_20 = vol.pct_change().rolling(20).std()
    factors['flow_vol_stability'] = -vol_std_20.fillna(0)

    # 4. Amihud价格冲击: |ret|/vol 越低=抛压越小
    amihud = np.abs(ret) / (vol.replace(0, np.nan) + 1)
    factors['flow_amihud'] = -amihud.rolling(20).mean().fillna(0)

    # 5. 价量背离: 跌放量+涨缩量=机构出货
    vol_chg = vol.pct_change().fillna(0)
    diverge = np.where(ret > 0, -np.sign(vol_chg), np.sign(vol_chg))
    factors['flow_diverge'] = pd.DataFrame(
        diverge, index=ret.index, columns=ret.columns
    ).rolling(20).mean().fillna(0)

    return factors
