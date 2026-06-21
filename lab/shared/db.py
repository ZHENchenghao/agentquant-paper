# -*- coding: utf-8 -*-
"""
QuantLab 共享层: DuckDB连接池 + 常用查询
"""
import duckdb
import pandas as pd
from datetime import date, timedelta

DB_PATH = 'D:/FreeFinanceData/data/duckdb/finance.db'


def get_conn(read_only=True):
    """获取DuckDB连接"""
    return duckdb.connect(DB_PATH, read_only=read_only)


def latest_trade_date(conn=None):
    """获取DuckDB最新K线日期"""
    close = conn is None
    if close:
        conn = get_conn()
    try:
        d = conn.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
        return str(d)
    finally:
        if close:
            conn.close()


def get_kline_range(conn, ts_codes, start, end):
    """批量取K线"""
    codes_str = ','.join([f"'{c}'" for c in ts_codes])
    return conn.execute(f"""
        SELECT ts_code, trade_date, close, volume, amount
        FROM kline_daily
        WHERE ts_code IN ({codes_str})
          AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY ts_code, trade_date
    """).df()


def get_global_index(conn, index_codes, start, end):
    """取全球指数日线"""
    codes_str = ','.join([f"'{c}'" for c in index_codes])
    return conn.execute(f"""
        SELECT ts_code, trade_date, close, volume
        FROM global_index_daily
        WHERE ts_code IN ({codes_str})
          AND trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY ts_code, trade_date
    """).df()


def get_northbound_daily(conn, start, end):
    """取北向资金日度数据"""
    return conn.execute(f"""
        SELECT trade_date,
               SUM(buy_amount) AS total_buy,
               SUM(sell_amount) AS total_sell,
               SUM(net_flow) AS net_flow
        FROM north_bound_flow
        WHERE trade_date BETWEEN '{start}' AND '{end}'
        GROUP BY trade_date
        ORDER BY trade_date
    """).df()


def get_northbound_by_stock(conn, start, end):
    """取北向资金个股维度"""
    return conn.execute(f"""
        SELECT ts_code, trade_date,
               buy_amount, sell_amount, net_flow,
               hold_shares, hold_ratio
        FROM north_bound_flow
        WHERE trade_date BETWEEN '{start}' AND '{end}'
        ORDER BY ts_code, trade_date
    """).df()


def get_margin_data(conn, start, end):
    """取融资融券数据"""
    return conn.execute(f"""
        SELECT trade_date,
               SUM(margin_buy) AS total_margin_buy,
               SUM(margin_balance) AS total_margin_balance
        FROM margin_trading
        WHERE trade_date BETWEEN '{start}' AND '{end}'
        GROUP BY trade_date
        ORDER BY trade_date
    """).df()


def get_industry_map(conn):
    """取申万行业映射"""
    return conn.execute("""
        SELECT ts_code, ind_name FROM (
            SELECT ts_code, ind_name,
                   ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
            FROM stock_industry_map
        ) WHERE rn = 1
    """).df()
