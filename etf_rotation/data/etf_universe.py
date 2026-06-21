# -*- coding: utf-8 -*-
"""
ETF池: 30个申万行业指数 = ETF代理
数据源: proxy_industry_daily (1999-2026, 30行业)
2005年前的ETF不存在，用行业指数收益模拟ETF持有收益
"""
import duckdb
import pandas as pd
import numpy as np

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def load_industry_universe(start='2002-01-01', end='2026-06-19'):
    """加载30个申万行业指数日线 → ETF代理池"""
    con = duckdb.connect(DB, read_only=True)
    df = con.execute(f"""
        SELECT trade_date, industry AS etf_code, close
        FROM proxy_industry_daily
        WHERE trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY industry, trade_date
    """).df()
    con.close()

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def build_return_matrix(price_df):
    """
    将价格数据转为日收益矩阵
    Returns: pivot table [date x industry] of daily returns
    """
    rets = price_df.copy()
    rets['ret'] = rets.groupby('etf_code')['close'].pct_change()
    rets = rets.dropna(subset=['ret'])
    pivot = rets.pivot(index='trade_date', columns='etf_code', values='ret')
    return pivot


def get_benchmark(start='2002-01-01', end='2026-06-19'):
    """获取沪深300基准 (proxy: 用全行业等权或上证指数)"""
    con = duckdb.connect(DB, read_only=True)
    # 用hs300_close作为基准 (存在lab_northbound_daily里)
    bench = con.execute(f"""
        SELECT trade_date, hs300_close
        FROM lab_northbound_daily
        WHERE trade_date BETWEEN '{start}' AND '{end}'
          AND hs300_close IS NOT NULL
        ORDER BY trade_date
    """).df()
    con.close()

    if bench.empty:
        # fallback: 用kline_daily的沪深300
        con = duckdb.connect(DB, read_only=True)
        bench = con.execute(f"""
            SELECT trade_date, close AS hs300_close
            FROM kline_daily
            WHERE ts_code = 'sh000300'
              AND trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY trade_date
        """).df()
        con.close()

    if not bench.empty:
        bench['trade_date'] = pd.to_datetime(bench['trade_date'])
        bench['bench_ret'] = bench['hs300_close'].pct_change()
        bench = bench.dropna(subset=['bench_ret'])
        bench = bench.set_index('trade_date')

    return bench


def get_industry_names():
    """获取行业中文名映射"""
    con = duckdb.connect(DB, read_only=True)
    names = con.execute("""
        SELECT DISTINCT industry FROM proxy_industry_daily ORDER BY industry
    """).fetchall()
    con.close()
    return [n[0] for n in names]
