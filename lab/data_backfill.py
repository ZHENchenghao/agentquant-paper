# -*- coding: utf-8 -*-
"""
QuantLab 数据回填脚本
用 AKShare + yfinance 回填10年数据到 DuckDB

回填项:
1. 北向资金日度净流向 (stock_hsgt_hist_em) 2014-2026
2. 融资融券日度 (stock_margin_detail_sse/szse)
3. 大盘主力资金流 (stock_market_fund_flow)
4. SOX 费城半导体指数 (yfinance, backfill to 10yr)

运行: python data_backfill.py
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import duckdb
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def backfill_northbound():
    """回填北向资金日度净流向"""
    print('\n' + '='*60)
    print('[1/4] 北向资金日度数据 (AKShare stock_hsgt_hist_em)')
    print('='*60)

    import akshare as ak

    # 获取北向历史 (2014-11-17 起)
    print('  获取北向历史数据...')
    df = ak.stock_hsgt_hist_em(symbol='北向资金')

    # 列名映射 (东方财富中文列名 → 英文字段名)
    col_map = {
        '日期': 'trade_date',
        '当日成交净买额': 'net_flow',
        '买入成交额': 'buy_amount',
        '卖出成交额': 'sell_amount',
        '历史累计净买额': 'cum_net',
        '当日资金流入': 'daily_inflow',
        '持股市值': 'hold_mv',
        '沪深300': 'hs300_close',
        '沪深300-涨跌幅': 'hs300_pct',
    }

    df = df.rename(columns=col_map)
    keep_cols = [v for v in col_map.values() if v in df.columns]
    df = df[keep_cols].copy()

    # 日期标准化
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')

    # 数值列清洗
    for c in ['net_flow', 'buy_amount', 'sell_amount', 'cum_net', 'daily_inflow']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 统计
    valid = df['net_flow'].notna().sum()
    total = len(df)
    print(f'  获取到 {total} 行, 有效净流向 {valid} 行 ({valid/total*100:.1f}%)')
    print(f'  日期范围: {df["trade_date"].min()} ~ {df["trade_date"].max()}')
    print(f'  最近有效日期: {df[df["net_flow"].notna()]["trade_date"].max()}')

    # 写入 DuckDB (新建表 lab_northbound_daily)
    con = duckdb.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS lab_northbound_daily (
            trade_date DATE PRIMARY KEY,
            net_flow DOUBLE,
            buy_amount DOUBLE,
            sell_amount DOUBLE,
            cum_net DOUBLE,
            daily_inflow DOUBLE,
            hs300_close DOUBLE,
            hs300_pct DOUBLE
        )
    """)
    con.execute("DELETE FROM lab_northbound_daily")
    # 只插入表里有的列
    tbl_cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='lab_northbound_daily'"
    ).fetchall()]
    insert_cols = [c for c in tbl_cols if c in df.columns]
    con.execute(f"INSERT INTO lab_northbound_daily ({','.join(insert_cols)}) SELECT {','.join(insert_cols)} FROM df")
    con.close()
    print(f'  ✅ 写入 lab_northbound_daily: {len(df)} 行')

    return df


def backfill_margin():
    """回填融资融券数据"""
    print('\n' + '='*60)
    print('[2/4] 融资融券日度数据 (AKShare)')
    print('='*60)

    import akshare as ak

    try:
        # 沪市融资融券
        print('  获取沪市融资融券...')
        m_sh = ak.macro_china_market_margin_sh()

        # 检查实际列名
        sh_cols_actual = list(m_sh.columns)
        print(f'  沪市列名: {sh_cols_actual[:5]}...')

        # 智能列名映射
        sh_map = {}
        for c in sh_cols_actual:
            if '日期' in c or 'date' in c.lower():
                sh_map[c] = 'trade_date'
            elif '融资余额' in c or 'margin' in c.lower() and 'balance' in c.lower():
                sh_map[c] = 'margin_balance_sh'
            elif '融资买入' in c or 'buy' in c.lower():
                sh_map[c] = 'margin_buy_sh'
            elif '融券' in c and '余' in c:
                sh_map[c] = 'short_shares_sh'

        m_sh = m_sh.rename(columns=sh_map)

        # 深市
        print('  获取深市融资融券...')
        m_sz = ak.macro_china_market_margin_sz()
        sz_cols_actual = list(m_sz.columns)
        print(f'  深市列名: {sz_cols_actual[:5]}...')

        sz_map = {}
        for c in sz_cols_actual:
            if '日期' in c or 'date' in c.lower():
                sz_map[c] = 'trade_date'
            elif '融资余额' in c or 'margin' in c.lower() and 'balance' in c.lower():
                sz_map[c] = 'margin_balance_sz'
            elif '融资买入' in c or 'buy' in c.lower():
                sz_map[c] = 'margin_buy_sz'
            elif '融券' in c and '余' in c:
                sz_map[c] = 'short_shares_sz'

        m_sz = m_sz.rename(columns=sz_map)

        print(f'  沪市: {len(m_sh)} 行, 深市: {len(m_sz)} 行')

        # 只保留需要的列
        sh_cols = [c for c in ['trade_date', 'margin_balance_sh', 'margin_buy_sh', 'short_shares_sh'] if c in m_sh.columns]
        sz_cols = [c for c in ['trade_date', 'margin_balance_sz', 'margin_buy_sz', 'short_shares_sz'] if c in m_sz.columns]

        merged = m_sh[sh_cols].merge(m_sz[sz_cols], on='trade_date', how='outer')
        merged['trade_date'] = pd.to_datetime(merged['trade_date']).dt.strftime('%Y-%m-%d')

        # 总融资余额
        merged['margin_balance'] = merged.get('margin_balance_sh', 0).fillna(0) + merged.get('margin_balance_sz', 0).fillna(0)
        merged['margin_buy'] = merged.get('margin_buy_sh', 0).fillna(0) + merged.get('margin_buy_sz', 0).fillna(0)

        con = duckdb.connect(DB)
        con.execute("""
            CREATE TABLE IF NOT EXISTS lab_margin_daily (
                trade_date DATE PRIMARY KEY,
                margin_balance DOUBLE,
                margin_buy DOUBLE,
                margin_balance_sh DOUBLE,
                margin_buy_sh DOUBLE,
                margin_balance_sz DOUBLE,
                margin_buy_sz DOUBLE
            )
        """)
        con.execute("DELETE FROM lab_margin_daily")
        keep_cols = [c for c in ['trade_date','margin_balance','margin_buy','margin_balance_sh','margin_buy_sh','margin_balance_sz','margin_buy_sz'] if c in merged.columns]
        tbl_cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='lab_margin_daily'").fetchall()]
        insert_cols = [c for c in tbl_cols if c in keep_cols]
        con.execute(f"INSERT INTO lab_margin_daily ({','.join(insert_cols)}) SELECT {','.join(insert_cols)} FROM merged")
        con.close()

        print(f'  ✅ 写入 lab_margin_daily: {len(merged)} 行')
        print(f'  日期范围: {merged["trade_date"].min()} ~ {merged["trade_date"].max()}')

    except Exception as e:
        print(f'  ⚠ 融资数据获取失败: {e}')


def backfill_capital_flow():
    """回填大盘主力资金流 (可获取约120天)"""
    print('\n' + '='*60)
    print('[3/4] 大盘主力资金流 (AKShare stock_market_fund_flow)')
    print('='*60)

    import akshare as ak
    import time as _time

    df = None
    for attempt in range(3):
        try:
            df = ak.stock_market_fund_flow()
            break
        except Exception as e:
            print(f'  尝试 {attempt+1}/3 失败: {e}')
            if attempt < 2:
                _time.sleep(3)

    if df is None:
        print('  ⚠ 3次重试均失败，跳过主力资金回填')
        return

    col_map = {
        '日期': 'trade_date',
        '上证-收盘价': 'sh_close',
        '上证-涨跌幅': 'sh_pct',
        '深证-收盘价': 'sz_close',
        '深证-涨跌幅': 'sz_pct',
        '主力净流入-净额': 'main_net',
        '主力净流入-净占比': 'main_pct',
        '超大单净流入-净额': 'super_large_net',
        '超大单净流入-净占比': 'super_large_pct',
        '大单净流入-净额': 'large_net',
        '大单净流入-净占比': 'large_pct',
        '中单净流入-净额': 'medium_net',
        '中单净流入-净占比': 'medium_pct',
        '小单净流入-净额': 'small_net',
        '小单净流入-净占比': 'small_pct',
    }
    df = df.rename(columns=col_map)
    keep = [v for v in col_map.values() if v in df.columns]
    df = df[keep].copy()
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')

    for c in keep:
        if c != 'trade_date':
            df[c] = pd.to_numeric(df[c], errors='coerce')

    print(f'  获取到 {len(df)} 行, 日期: {df["trade_date"].min()} ~ {df["trade_date"].max()}')

    con = duckdb.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS lab_capital_flow (
            trade_date DATE PRIMARY KEY,
            main_net DOUBLE, main_pct DOUBLE,
            super_large_net DOUBLE, super_large_pct DOUBLE,
            large_net DOUBLE, large_pct DOUBLE,
            medium_net DOUBLE, medium_pct DOUBLE,
            small_net DOUBLE, small_pct DOUBLE,
            sh_close DOUBLE, sh_pct DOUBLE
        )
    """)
    con.execute("DELETE FROM lab_capital_flow")
    tbl_cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns WHERE table_name='lab_capital_flow'").fetchall()]
    insert_cols = [c for c in tbl_cols if c in df.columns]
    con.execute(f"INSERT INTO lab_capital_flow ({','.join(insert_cols)}) SELECT {','.join(insert_cols)} FROM df")
    con.close()
    print(f'  ✅ 写入 lab_capital_flow: {len(df)} 行')


def backfill_sox():
    """用 AKShare macro_global_sox_index 回填 SOX 数据 (1994-2026, 8000+行)"""
    print('\n' + '='*60)
    print('[4/4] SOX 费城半导体指数 (AKShare macro_global_sox_index)')
    print('='*60)

    import akshare as ak

    print('  获取 SOX 历史数据 (1994年起)...')
    df = ak.macro_global_sox_index()

    # 列名映射
    col_map = {
        '日期': 'trade_date',
        '数值': 'close',
        '涨跌幅': 'pct_change',
    }
    df = df.rename(columns=col_map)

    # 只保留需要的列
    keep = ['trade_date', 'close']
    if 'pct_change' in df.columns:
        keep.append('pct_change')
    df = df[keep].copy()

    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df = df.dropna(subset=['close'])

    print(f'  获取到 {len(df)} 行, 日期: {df["trade_date"].min()} ~ {df["trade_date"].max()}')
    print(f'  最新收盘: {df["close"].iloc[-1]:.2f} ({df["trade_date"].iloc[-1]})')

    con = duckdb.connect(DB)

    # 清除旧SOX数据，写入新数据
    con.execute("DELETE FROM global_index_daily WHERE index_code='SOX'")

    for _, row in df.iterrows():
        close_val = float(row['close'])
        con.execute("""
            INSERT INTO global_index_daily (index_code, trade_date, open, high, low, close, volume, amount)
            VALUES (?, ?, ?, ?, ?, ?, 0, 0)
        """, ['SOX', row['trade_date'], close_val, close_val, close_val, close_val])

    total = con.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM global_index_daily WHERE index_code='SOX'").fetchone()
    con.close()

    print(f'  ✅ SOX 总计: {total[0]} 行, {total[1]} ~ {total[2]}')


def verify_all():
    """验证所有回填表"""
    print('\n' + '='*60)
    print('数据验证')
    print('='*60)

    con = duckdb.connect(DB, read_only=True)
    tables = ['lab_northbound_daily', 'lab_margin_daily', 'lab_capital_flow']

    for t in tables:
        try:
            cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            dr = con.execute(f"SELECT MIN(trade_date), MAX(trade_date) FROM {t}").fetchone()
            print(f'  {t}: {cnt} 行, {dr[0]} ~ {dr[1]}')
        except Exception as e:
            print(f'  {t}: ⚠ {e}')

    # SOX
    sox = con.execute("SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM global_index_daily WHERE index_code='SOX'").fetchone()
    print(f'  global_index_daily (SOX): {sox[0]} 行, {sox[1]} ~ {sox[2]}')

    con.close()


if __name__ == '__main__':
    start = time.time()

    for name, func in [
        ('北向资金', backfill_northbound),
        ('融资融券', backfill_margin),
        ('主力资金', backfill_capital_flow),
        ('SOX指数', backfill_sox),
    ]:
        try:
            func()
        except Exception as e:
            print(f'\n  ⚠ [{name}] 回填失败: {e}')
            print(f'  继续下一项...')

    verify_all()

    elapsed = time.time() - start
    print(f'\n全部回填完成 ({elapsed:.1f}s)')
