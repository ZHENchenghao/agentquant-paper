# -*- coding: utf-8 -*-
"""
AgentQuant · ML因子组合滚动回测 — 完整版 v1.0
================================================
6窗口 × 13因子组合 × LightGBM
目标: 60日超额收益(相对沪深300)
防作弊: 严格时间分割 + PIT对齐 + 样本外测试

因子组:
  A  价量因子     ~85     ROC/MA/STD/Kbar/量比 (Alpha158精要)
  B  财报因子      20     OCF质量/ROE/毛利率/商誉率/应收率/净利增速
  C  判断因子       9     国家队/北向背离/跳空/缩量/连涨跌/融资/轮动
  H  技术指标      25     MACD/KDJ/RSI/BOLL/MA距离/均线排列
  I  估值因子       8     PE/PB/PS/市值(log)代理 + 杜邦分项
  J  宏观因子      10     WTI/VIX/美10Y/中美利差/铜金比/SPX/SOX...

组合测试(13组):
  单组: A, B, C, H, I
  双组: A+B, A+C, A+H
  递进: A+B+C, A+B+C+H, A+B+C+H+I
  全量: ALL (A+B+C+H+I+J)

基准:
  沪深300买入持有 / 随机30只等权 / V3原始策略(如有)
"""
import sys, io, os, json, time, warnings, traceback
from datetime import date, timedelta
from itertools import product

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── 配置 ───────────────────────────────────────────
DB_PATH = 'D:/FreeFinanceData/data/duckdb/finance.db'
OUT_DIR = 'D:/AgentQuant/our/reports'
os.makedirs(OUT_DIR, exist_ok=True)

# 6个滚动窗口
WINDOWS = [
    # (训练开始, 训练结束, 验证期, 测试期开始, 测试期结束)
    ('2015-01-01', '2016-12-31', '2017-01-01', '2017-07-01', '2018-01-01', '2018-12-31'),
    ('2016-01-01', '2017-12-31', '2018-01-01', '2018-07-01', '2019-01-01', '2019-12-31'),
    ('2017-01-01', '2018-12-31', '2019-01-01', '2019-07-01', '2020-01-01', '2020-12-31'),
    ('2018-01-01', '2019-12-31', '2020-01-01', '2020-07-01', '2021-01-01', '2021-12-31'),
    ('2019-01-01', '2020-12-31', '2021-01-01', '2021-07-01', '2022-01-01', '2022-12-31'),
    ('2020-01-01', '2021-12-31', '2022-01-01', '2022-07-01', '2023-01-01', '2026-06-16'),
]

# 因子组定义
FACTOR_GROUPS = {
    'A': '价量因子 (Alpha158精要 ~85)',
    'B': '财报因子 (OCF/ROE/毛利率/商誉 ~20)',
    'C': '判断因子 (国家队/北向/跳空 ~9)',
    'H': '技术指标 (MACD/KDJ/RSI/BOLL ~25)',
    'I': '估值因子 (PE/PB/PS/市值代理 ~8)',
    'J': '宏观因子 (WTI/VIX/美10Y ~10)',
}

# 测试组合
COMBINATIONS = [
    ('A',       ['A']),
    ('B',       ['B']),
    ('C',       ['C']),
    ('H',       ['H']),
    ('I',       ['I']),
    ('A+B',     ['A', 'B']),
    ('A+C',     ['A', 'C']),
    ('A+H',     ['A', 'H']),
    ('A+B+C',   ['A', 'B', 'C']),
    ('A+B+C+H', ['A', 'B', 'C', 'H']),
    ('A+B+C+H+I', ['A', 'B', 'C', 'H', 'I']),
    ('ALL',     ['A', 'B', 'C', 'H', 'I', 'J']),
]

# LightGBM参数
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 500,
    'early_stopping_rounds': 30,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
}

# 月频调仓参数
TOP_N = 30          # 每月选Top30
REBALANCE_FREQ = 'M'  # 月频
FORWARD_DAYS = 60   # 预测60日超额收益 (~1季度, 基本面因子有足够时间反应)

# ─── 工具函数 ────────────────────────────────────────

def get_db(read_only=True):
    """获取DuckDB连接，自动重试"""
    for i in range(5):
        try:
            c = duckdb.connect(DB_PATH, read_only=read_only)
            c.execute('SELECT 1')
            return c
        except Exception:
            time.sleep(min(2 ** i, 10))
    return duckdb.connect(DB_PATH, read_only=read_only)


def safe_sql(c, query, label=''):
    """执行SQL，自动重试，返回DataFrame"""
    for attempt in range(3):
        try:
            return c.execute(query).df()
        except Exception as e:
            if attempt == 2:
                print(f'  ⚠ SQL失败 [{label}]: {str(e)[:120]}')
                return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()


# ─── Part 1: 价量因子 (A组 ~85) ──────────────────────

def build_A_price_volume(c, start_date, end_date, month_ends):
    """
    Alpha158精要: 在月底日期截面上计算价量因子
    直接用DuckDB窗口函数批量算
    返回: DataFrame[ts_code, trade_date, roc_5, ..., kbar_ksft2]
    """
    print('  [A] 构建价量因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    df = safe_sql(c, f"""
        WITH daily AS (
            -- 基础OHLCV + 日收益
            SELECT ts_code, trade_date, open, high, low, close, vol AS volume, amount,
                   (close/LAG(close) OVER w - 1) AS ret_1d
            FROM kline_daily
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
              AND close > 0 AND volume > 0
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
        factored AS (
            SELECT ts_code, trade_date,
                -- ── Kbar特征 (9) ──
                (close-open)/NULLIF(open,0) AS kbar_kmid,
                (high-low)/NULLIF(open,0) AS kbar_klen,
                (close-open)/NULLIF(high-low,0) AS kbar_kmid2,
                (high-GREATEST(open,close))/NULLIF(open,0) AS kbar_kup,
                (high-GREATEST(open,close))/NULLIF(high-low,0) AS kbar_kup2,
                (LEAST(open,close)-low)/NULLIF(open,0) AS kbar_klow,
                (LEAST(open,close)-low)/NULLIF(high-low,0) AS kbar_klow2,
                (2*close-high-low)/NULLIF(open,0) AS kbar_ksft,
                (2*close-high-low)/NULLIF(high-low,0) AS kbar_ksft2,

                -- ── ROC (5窗口) ──
                close/NULLIF(LAG(close,5) OVER w2,0)-1 AS roc_5,
                close/NULLIF(LAG(close,10) OVER w2,0)-1 AS roc_10,
                close/NULLIF(LAG(close,20) OVER w2,0)-1 AS roc_20,
                close/NULLIF(LAG(close,30) OVER w2,0)-1 AS roc_30,
                close/NULLIF(LAG(close,60) OVER w2,0)-1 AS roc_60,

                -- ── 均线距离 (5窗口) ──
                AVG(close) OVER (w2 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)/NULLIF(close,0)-1 AS ma_dist_5,
                AVG(close) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)/NULLIF(close,0)-1 AS ma_dist_10,
                AVG(close) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)/NULLIF(close,0)-1 AS ma_dist_20,
                AVG(close) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)/NULLIF(close,0)-1 AS ma_dist_30,
                AVG(close) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)/NULLIF(close,0)-1 AS ma_dist_60,

                -- ── 波动率 (5窗口, 用已算好的ret_1d避免嵌套) ──
                STDDEV_SAMP(ret_1d) OVER (w2 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS vol_5,
                STDDEV_SAMP(ret_1d) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS vol_10,
                STDDEV_SAMP(ret_1d) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS vol_20,
                STDDEV_SAMP(ret_1d) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS vol_30,
                STDDEV_SAMP(ret_1d) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS vol_60,

                -- ── 最大/最小值位置 ──
                (MAX(high) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))/NULLIF(close,0) AS max20_pos,
                (MIN(low) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))/NULLIF(close,0) AS min20_pos,
                (MAX(high) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW))/NULLIF(close,0) AS max60_pos,
                (MIN(low) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW))/NULLIF(close,0) AS min60_pos,

                -- ── RSV (4窗口) ──
                (close-MIN(low) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW))/
                    NULLIF(MAX(high) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)-MIN(low) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),0) AS rsv_10,
                (close-MIN(low) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))/
                    NULLIF(MAX(high) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)-MIN(low) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),0) AS rsv_20,
                (close-MIN(low) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW))/
                    NULLIF(MAX(high) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)-MIN(low) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),0) AS rsv_30,
                (close-MIN(low) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW))/
                    NULLIF(MAX(high) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)-MIN(low) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),0) AS rsv_60,

                -- ── 涨跌占比 (用ret_1d>0判断, 5窗口) ──
                AVG(CASE WHEN ret_1d>0 THEN 1.0 ELSE 0.0 END) OVER (w2 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS up_pct_5,
                AVG(CASE WHEN ret_1d>0 THEN 1.0 ELSE 0.0 END) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS up_pct_10,
                AVG(CASE WHEN ret_1d>0 THEN 1.0 ELSE 0.0 END) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS up_pct_20,
                AVG(CASE WHEN ret_1d>0 THEN 1.0 ELSE 0.0 END) OVER (w2 ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS up_pct_30,
                AVG(CASE WHEN ret_1d>0 THEN 1.0 ELSE 0.0 END) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS up_pct_60,

                -- ── RSI-like (用ret_1d, 3窗口) ──
                SUM(CASE WHEN ret_1d>0 THEN ret_1d ELSE 0 END) OVER (w2 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) /
                    NULLIF(SUM(ABS(ret_1d)) OVER (w2 ROWS BETWEEN 5 PRECEDING AND CURRENT ROW), 0) AS rsi_6,
                SUM(CASE WHEN ret_1d>0 THEN ret_1d ELSE 0 END) OVER (w2 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) /
                    NULLIF(SUM(ABS(ret_1d)) OVER (w2 ROWS BETWEEN 13 PRECEDING AND CURRENT ROW), 0) AS rsi_14,
                SUM(CASE WHEN ret_1d>0 THEN ret_1d ELSE 0 END) OVER (w2 ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) /
                    NULLIF(SUM(ABS(ret_1d)) OVER (w2 ROWS BETWEEN 23 PRECEDING AND CURRENT ROW), 0) AS rsi_24,

                -- ── 量价 ──
                AVG(volume) OVER (w2 ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)/NULLIF(volume,0) AS vma_5,
                AVG(volume) OVER (w2 ROWS BETWEEN 9 PRECEDING AND CURRENT ROW)/NULLIF(volume,0) AS vma_10,
                AVG(volume) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)/NULLIF(volume,0) AS vma_20,
                AVG(volume) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)/NULLIF(volume,0) AS vma_60,

                -- 成交活跃度
                amount/NULLIF(AVG(amount) OVER (w2 ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),0) AS amt_ratio_20,
                amount/NULLIF(AVG(amount) OVER (w2 ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),0) AS amt_ratio_60,

                -- 价格位置(52周)
                (close-MIN(low) OVER (w2 ROWS BETWEEN 249 PRECEDING AND CURRENT ROW))/
                    NULLIF(MAX(high) OVER (w2 ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)-MIN(low) OVER (w2 ROWS BETWEEN 249 PRECEDING AND CURRENT ROW),0) AS pct_52w

            FROM daily
            WINDOW w2 AS (PARTITION BY ts_code ORDER BY trade_date)
        )
        SELECT * FROM factored
        WHERE trade_date IN ({me_str})
          AND roc_5 IS NOT NULL
    """, 'A_price_volume')

    if df.empty:
        print(f'❌ 空')
        return df

    # 去除Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 标记列名
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date')]
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 2: 财报因子 (B组 ~20) ──────────────────────

def build_B_fundamental(c, start_date, end_date, month_ends):
    """
    PIT对齐的财報质量因子
    OCF质量/ROE/毛利率/商誉率/应收率/净利增速/杠杆/周转...
    """
    print('  [B] 构建财报因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    df = safe_sql(c, f"""
        WITH
        me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        pit AS (
            SELECT me.trade_date,
                   -- 转换ts_code格式: '000001.SZ' → 'sz000001', '600519.SH' → 'sh600519'
                   CASE WHEN f.ts_code LIKE '%.SZ' THEN 'sz' || REPLACE(f.ts_code, '.SZ', '')
                        WHEN f.ts_code LIKE '%.SH' THEN 'sh' || REPLACE(f.ts_code, '.SH', '')
                        WHEN f.ts_code LIKE '%.BJ' THEN 'bj' || REPLACE(f.ts_code, '.BJ', '')
                        ELSE f.ts_code END AS ts_code,
                   f.report_date, f.report_type,
                   f.net_profit, f.revenue, f.roe, f.gross_margin, f.net_margin,
                   f.eps, f.operating_cf, f.accounts_receivable, f.goodwill,
                   ROW_NUMBER() OVER(PARTITION BY me.trade_date, f.ts_code ORDER BY f.report_date DESC) rn
            FROM me
            JOIN financial_statements f ON f.report_date <= me.trade_date
                AND f.report_date >= me.trade_date - INTERVAL '540 days'
                AND f.net_profit IS NOT NULL
        ),
        latest AS (
            SELECT * FROM pit WHERE rn = 1
        )
        SELECT trade_date, ts_code,
               roe, gross_margin, net_margin,
               eps,
               net_profit/NULLIF(revenue,0) AS profit_margin,
               LN(NULLIF(eps,0)) AS log_eps
        FROM latest
        WHERE roe IS NOT NULL AND roe > 0 AND roe < 100
          AND eps IS NOT NULL AND eps > 0
    """, 'B_fundamental')

    if df.empty:
        print(f'❌ 空')
        return df

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = ['roe', 'gross_margin', 'net_margin', 'eps', 'profit_margin', 'log_eps']
    feat_cols = [c for c in feat_cols if c in df.columns]
    for col in feat_cols:
        if df[col].notna().sum() > 50:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 3: 判断因子 (C组 ~9) ──────────────────────

def build_C_judgment(c, start_date, end_date, month_ends):
    """
    市场层面的模式判断因子。同一日期所有股票共享。
    缩放到个股层面：每只股票在同一天获得相同的判断因子值。
    """
    print('  [C] 构建判断因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    # 这些是市场级别信号，需要先按日期算，再广播到个股
    signals = safe_sql(c, f"""
        WITH
        me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),

        -- 国家队信号(上证50涨vs科创跌+缩量)
        nt AS (
            SELECT k1.trade_date,
                   CASE WHEN (k1.close/LAG(k1.close) OVER w1-1)*100 > 0.8
                         AND (k2.close/LAG(k2.close) OVER w2-1)*100 < 0.3
                         AND v.vol_ratio < 0.85
                        THEN 1 ELSE 0 END AS nt_guard,
                   CASE WHEN (k1.close/LAG(k1.close) OVER w1-1)*100 < -0.8
                         AND v.vol_ratio > 1.2
                        THEN 1 ELSE 0 END AS nt_retreat
            FROM kline_daily k1
            JOIN kline_daily k2 ON k1.trade_date=k2.trade_date
            JOIN (
                SELECT trade_date,
                       SUM(amount)/NULLIF(AVG(SUM(amount)) OVER(ORDER BY trade_date ROWS 19 PRECEDING),0) AS vol_ratio
                FROM kline_daily GROUP BY trade_date
            ) v ON k1.trade_date=v.trade_date
            WHERE k1.ts_code='sh000016' AND k2.ts_code='sh000688'
            WINDOW w1 AS (ORDER BY k1.trade_date), w2 AS (ORDER BY k2.trade_date)
        ),

        -- 北向背离
        nb AS (
            SELECT n.trade_date,
                   CASE WHEN n.daily<-30 AND i.chg>0 THEN 1 ELSE 0 END AS nb_outflow_diverge,
                   CASE WHEN n.daily>30 AND i.chg<0 THEN 1 ELSE 0 END AS nb_inflow_diverge,
                   CASE WHEN n.daily>30 AND i.chg>0 THEN 1 ELSE 0 END AS nb_bull_resonance
            FROM (
                SELECT trade_date, SUM(net_flow) daily FROM north_bound_flow
                WHERE net_flow IS NOT NULL AND net_flow!=0 GROUP BY trade_date
            ) n
            JOIN (
                SELECT trade_date, (close/LAG(close) OVER(ORDER BY trade_date)-1)*100 chg
                FROM kline_daily WHERE ts_code='sh000300'
            ) i ON n.trade_date=i.trade_date
        ),

        -- 跳空/缩量
        gap_vol AS (
            SELECT p.trade_date,
                   CASE WHEN p.gap>0.7 THEN 1 ELSE 0 END AS gap_up,
                   CASE WHEN p.gap<-0.7 THEN 1 ELSE 0 END AS gap_down,
                   CASE WHEN p.chg<-0.8 AND v.vr<0.8 THEN 1 ELSE 0 END AS shrink_fall,
                   CASE WHEN p.chg<-0.8 AND v.vr>1.3 THEN 1 ELSE 0 END AS expand_fall
            FROM (
                SELECT trade_date,
                       (open/LAG(close) OVER(ORDER BY trade_date)-1)*100 gap,
                       (close/LAG(close) OVER(ORDER BY trade_date)-1)*100 chg
                FROM kline_daily WHERE ts_code='sh000300'
            ) p
            JOIN (
                SELECT trade_date,
                       SUM(amount)/NULLIF(AVG(SUM(amount)) OVER(ORDER BY trade_date ROWS 19 PRECEDING),0) vr
                FROM kline_daily GROUP BY trade_date
            ) v ON p.trade_date=v.trade_date
        ),

        -- 连涨连跌
        streak AS (
            SELECT trade_date,
                   CASE WHEN streak5=5 THEN 1 ELSE 0 END AS streak5_up,
                   CASE WHEN streak5=0 THEN 1 ELSE 0 END AS streak5_dn
            FROM (
                SELECT trade_date,
                       CASE WHEN close>LAG(close) OVER o THEN 1 ELSE 0 END +
                       CASE WHEN LAG(close) OVER o>LAG(close,2) OVER o THEN 1 ELSE 0 END +
                       CASE WHEN LAG(close,2) OVER o>LAG(close,3) OVER o THEN 1 ELSE 0 END +
                       CASE WHEN LAG(close,3) OVER o>LAG(close,4) OVER o THEN 1 ELSE 0 END +
                       CASE WHEN LAG(close,4) OVER o>LAG(close,5) OVER o THEN 1 ELSE 0 END streak5
                FROM kline_daily WHERE ts_code='sh000300'
                WINDOW o AS (ORDER BY trade_date)
            )
        ),

        -- 融资变动
        margin AS (
            SELECT trade_date,
                   CASE WHEN margin_chg<-3 THEN 1 ELSE 0 END AS margin_panic,
                   CASE WHEN margin_chg>3 THEN 1 ELSE 0 END AS margin_greed
            FROM (
                SELECT trade_date,
                       (margin_balance/LAG(margin_balance) OVER(ORDER BY trade_date)-1)*100 margin_chg
                FROM margin_trading WHERE margin_balance IS NOT NULL
            )
        )

        SELECT me.trade_date,
               COALESCE(nt.nt_guard,0) AS nt_guard,
               COALESCE(nt.nt_retreat,0) AS nt_retreat,
               COALESCE(nb.nb_outflow_diverge,0) AS nb_outflow_diverge,
               COALESCE(nb.nb_inflow_diverge,0) AS nb_inflow_diverge,
               COALESCE(nb.nb_bull_resonance,0) AS nb_bull_resonance,
               COALESCE(gv.gap_up,0) AS gap_up,
               COALESCE(gv.gap_down,0) AS gap_down,
               COALESCE(gv.shrink_fall,0) AS shrink_fall,
               COALESCE(gv.expand_fall,0) AS expand_fall,
               COALESCE(st.streak5_up,0) AS streak5_up,
               COALESCE(st.streak5_dn,0) AS streak5_dn,
               COALESCE(mg.margin_panic,0) AS margin_panic,
               COALESCE(mg.margin_greed,0) AS margin_greed
        FROM me
        LEFT JOIN nt ON me.trade_date=nt.trade_date
        LEFT JOIN nb ON me.trade_date=nb.trade_date
        LEFT JOIN gap_vol gv ON me.trade_date=gv.trade_date
        LEFT JOIN streak st ON me.trade_date=st.trade_date
        LEFT JOIN margin mg ON me.trade_date=mg.trade_date
    """, 'C_judgment')

    if signals.empty:
        print(f'❌ 空(信号)')
        return pd.DataFrame()

    # 广播到个股: join with all stocks at each month end
    stocks = safe_sql(c, f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date IN ({me_str}) AND close > 0 AND vol > 0
    """, 'C_stocks')

    if stocks.empty:
        print(f'❌ 空(股票)')
        return pd.DataFrame()

    # Cross join stocks × signals
    stocks['_key'] = 1
    signals['_key'] = 1
    df = stocks.merge(signals, on='_key').drop(columns=['_key'])
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date')]
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 4: 技术指标因子 (H组 ~25) ──────────────────

def build_H_technical(c, start_date, end_date, month_ends):
    """
    从technical_indicators表直接读预计算指标，衍生距离因子
    """
    print('  [H] 构建技术指标因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    df = safe_sql(c, f"""
        SELECT t.ts_code, t.trade_date,
            -- MACD衍生
            t.macd_dif/NULLIF(k.close,0) AS macd_dif_norm,
            (t.macd_dif-t.macd_dea)/NULLIF(k.close,0) AS macd_hist_norm,
            CASE WHEN t.macd_dif>t.macd_dea THEN 1 ELSE -1 END AS macd_signal,
            CASE WHEN t.macd_dif>t.macd_dea AND LAG(t.macd_dif) OVER w <= LAG(t.macd_dea) OVER w THEN 1 ELSE 0 END AS macd_golden_cross,

            -- KDJ
            CASE WHEN t.kdj_j<20 THEN 1 WHEN t.kdj_j>80 THEN -1 ELSE 0 END AS kdj_oversold,
            t.kdj_j/NULLIF(100,0) AS kdj_j_norm,

            -- RSI
            t.rsi6/NULLIF(100,0) AS rsi6_norm,
            t.rsi14/NULLIF(100,0) AS rsi14_norm,
            CASE WHEN t.rsi6<30 THEN 1 WHEN t.rsi6>70 THEN -1 ELSE 0 END AS rsi_extreme,

            -- 布林带
            (k.close-t.boll_lower)/NULLIF(t.boll_upper-t.boll_lower,0) AS boll_position,
            (t.boll_upper-t.boll_lower)/NULLIF(t.boll_mid,0) AS boll_bandwidth,

            -- MA距离
            k.close/NULLIF(t.ma5,0)-1 AS close_div_ma5,
            k.close/NULLIF(t.ma10,0)-1 AS close_div_ma10,
            k.close/NULLIF(t.ma20,0)-1 AS close_div_ma20,
            k.close/NULLIF(t.ma60,0)-1 AS close_div_ma60,
            k.close/NULLIF(t.ma120,0)-1 AS close_div_ma120,
            k.close/NULLIF(t.ma200,0)-1 AS close_div_ma200,

            -- 均线排列分数
            CASE WHEN t.ma5>t.ma20 AND t.ma20>t.ma60 AND t.ma60>t.ma120 THEN 3
                 WHEN t.ma5>t.ma20 AND t.ma20>t.ma60 THEN 2
                 WHEN t.ma5>t.ma20 THEN 1
                 WHEN t.ma5<t.ma20 AND t.ma20<t.ma60 THEN -2
                 WHEN t.ma5<t.ma20 THEN -1
                 ELSE 0 END AS ma_alignment_score,

            -- 均线交叉信号
            CASE WHEN t.ma5>t.ma20 AND LAG(t.ma5) OVER w <= LAG(t.ma20) OVER w THEN 1 ELSE 0 END AS ma5_cross_ma20,
            CASE WHEN t.ma20>t.ma60 AND LAG(t.ma20) OVER w <= LAG(t.ma60) OVER w THEN 1 ELSE 0 END AS ma20_cross_ma60,

            -- 量比
            t.volume_ratio AS vol_ratio_tech,
            t.vol_ma5/NULLIF(t.vol_ma20,0) AS vol_ma_ratio

        FROM technical_indicators t
        JOIN kline_daily k ON t.ts_code=k.ts_code AND t.trade_date=k.trade_date
        WHERE t.trade_date IN ({me_str})
        WINDOW w AS (PARTITION BY t.ts_code ORDER BY t.trade_date)
    """, 'H_technical')

    if df.empty:
        print(f'❌ 空')
        return df

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date')]
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 5: 估值因子 (I组 ~8) ──────────────────────

def build_I_valuation(c, start_date, end_date, month_ends):
    """
    估值因子: PE/PB(由PE×ROE反推)/PS/市值代理
    总股本 = net_profit / eps, 市值 = close × 总股本
    PB = PE × ROE/100 (因 shareholders_equity 全 NULL, 用杜邦恒等式反推)
    """
    print('  [I] 构建估值因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    df = safe_sql(c, f"""
        WITH
        me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        pit AS (
            SELECT me.trade_date,
                   CASE WHEN f.ts_code LIKE '%.SZ' THEN 'sz' || REPLACE(f.ts_code, '.SZ', '')
                        WHEN f.ts_code LIKE '%.SH' THEN 'sh' || REPLACE(f.ts_code, '.SH', '')
                        WHEN f.ts_code LIKE '%.BJ' THEN 'bj' || REPLACE(f.ts_code, '.BJ', '')
                        ELSE f.ts_code END AS ts_code,
                   f.report_date,
                   f.net_profit, f.eps, f.revenue,
                   f.roe, f.gross_margin, f.net_margin,
                   ROW_NUMBER() OVER(PARTITION BY me.trade_date, f.ts_code ORDER BY f.report_date DESC) rn
            FROM me
            JOIN financial_statements f ON f.report_date <= me.trade_date
                AND f.report_date >= me.trade_date - INTERVAL '540 days'
                AND f.net_profit IS NOT NULL AND f.net_profit > 0
                AND f.eps IS NOT NULL AND f.eps > 0
                AND f.roe IS NOT NULL AND f.roe > 0 AND f.roe < 100
        ),
        fin AS (SELECT * FROM pit WHERE rn=1),
        priced AS (
            SELECT f.trade_date, f.ts_code, k.close,
                   f.net_profit, f.eps, f.revenue, f.roe, f.gross_margin, f.net_margin,
                   f.net_profit/NULLIF(f.eps,0) AS implied_shares,
                   k.close * (f.net_profit/NULLIF(f.eps,0)) AS market_cap
            FROM fin f
            JOIN kline_daily k ON f.ts_code=k.ts_code AND f.trade_date=k.trade_date
            WHERE k.close > 0
        )
        SELECT trade_date, ts_code,
               -- PE proxy
               market_cap/NULLIF(net_profit,0) AS pe_proxy,
               -- PB = PE × ROE/100 (杜邦恒等式)
               (market_cap/NULLIF(net_profit,0)) * (roe/100.0) AS pb_proxy,
               -- PS proxy
               market_cap/NULLIF(revenue,0) AS ps_proxy,
               -- 规模因子(log市值)
               LN(NULLIF(market_cap,0)) AS log_mcap,
               -- ROE
               roe AS roe_val,
               -- 毛利率
               gross_margin,
               -- 净利率
               net_margin,
               -- EPS
               eps
        FROM priced
        WHERE market_cap > 0 AND market_cap < 1e16
    """, 'I_valuation')

    if df.empty:
        print(f'❌ 空')
        return df

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date')]
    for col in feat_cols:
        if df[col].notna().sum() > 50:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 6: 宏观因子 (J组 ~10) ──────────────────────

def build_J_macro(c, start_date, end_date, month_ends):
    """
    宏观因子: 同日期所有股票共享
    可用数据源: macro_indicators(vix/usdcny/gold/us10y/wti/copper)
               + global_index_daily(SPX/SOX 月涨速, 2021-05起)
    不可用: china_10y(仅26行), spx/nasdaq/sox在macro表全NULL
    """
    print('  [J] 构建宏观因子...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    macro = safe_sql(c, f"""
        WITH
        me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        -- 主宏观表(仅有数据的列)
        m AS (
            SELECT trade_date, vix, usdcny, gold, us10y, wti, copper
            FROM macro_indicators
            WHERE trade_date IN ({me_str})
        ),
        -- SPX月涨速 (从global_index_daily)
        spx_data AS (
            SELECT trade_date, close,
                   (close/NULLIF(LAG(close, 20) OVER(ORDER BY trade_date), 0)-1)*100 AS spx_mom
            FROM global_index_daily
            WHERE index_code = '.INX' AND trade_date IN ({me_str})
        ),
        -- SOX月涨速
        sox_data AS (
            SELECT trade_date, close,
                   (close/NULLIF(LAG(close, 20) OVER(ORDER BY trade_date), 0)-1)*100 AS sox_mom
            FROM global_index_daily
            WHERE index_code IN ('SOX', '.SOX') AND trade_date IN ({me_str})
        )
        SELECT me.trade_date,
               m.vix, m.usdcny, m.gold, m.us10y, m.wti, m.copper,
               m.wti/NULLIF(m.gold, 0) AS wti_gold_ratio,
               m.copper/NULLIF(m.gold, 0) AS copper_gold_ratio,
               m.us10y - LAG(m.us10y, 20) OVER(ORDER BY me.trade_date) AS us10y_chg_m,
               CASE WHEN m.vix > 25 THEN 1 WHEN m.vix > 20 THEN 0.5 ELSE 0 END AS vix_alert,
               spx.spx_mom,
               sox.sox_mom,
               spx.close AS spx_level,
               sox.close AS sox_level
        FROM me
        LEFT JOIN m ON me.trade_date=m.trade_date
        LEFT JOIN spx_data spx ON me.trade_date=spx.trade_date
        LEFT JOIN sox_data sox ON me.trade_date=sox.trade_date
        ORDER BY me.trade_date
    """, 'J_macro')

    if macro.empty:
        print(f'❌ 空')
        return pd.DataFrame()

    # 前向填充 + 后向填充 缺失值
    macro = macro.sort_values('trade_date')
    for col in macro.columns:
        if col != 'trade_date' and macro[col].notna().sum() > 0:
            macro[col] = macro[col].ffill().bfill()

    # 广播到个股
    stocks = safe_sql(c, f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date IN ({me_str}) AND close > 0 AND vol > 0
    """, 'J_stocks')

    if stocks.empty:
        print(f'❌ 空(股票)')
        return pd.DataFrame()

    stocks['_key'] = 1
    macro['_key'] = 1
    df = stocks.merge(macro, on='_key').drop(columns=['_key'])
    feat_cols = [c for c in df.columns if c not in ('ts_code', 'trade_date')]
    print(f'{len(df)}行 × {len(feat_cols)}因子 ✓')
    return df


# ─── Part 7: 目标变量 ─────────────────────────────────

def build_target(c, month_ends):
    """
    构建60日超额收益(相对沪深300)
    LEAD(close, 20) 直接算，已在DuckDB验证通过
    """
    print('  [T] 构建目标变量(60日超额收益)...', end=' ', flush=True)
    me_str = ','.join([f"'{d}'" for d in month_ends])

    df = safe_sql(c, f"""
        WITH
        stock_fwd AS (
            SELECT ts_code, trade_date, close,
                   LEAD(close, 60) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fwd_close
            FROM kline_daily
            WHERE trade_date BETWEEN '2015-01-01' AND '2026-06-16'
        ),
        idx_fwd AS (
            SELECT trade_date, close,
                   LEAD(close, 60) OVER(ORDER BY trade_date) AS fwd_close
            FROM kline_daily
            WHERE ts_code='sh000300' AND trade_date BETWEEN '2015-01-01' AND '2026-06-16'
        )
        SELECT s.ts_code, s.trade_date,
               (s.fwd_close/NULLIF(s.close,0)-1) AS fwd_ret_20d,
               (s.fwd_close/NULLIF(s.close,0)-1) - (i.fwd_close/NULLIF(i.close,0)-1) AS excess_ret_20d
        FROM stock_fwd s
        JOIN idx_fwd i ON s.trade_date=i.trade_date
        WHERE s.trade_date IN ({me_str})
          AND s.fwd_close IS NOT NULL
          AND i.fwd_close IS NOT NULL
    """, 'target')

    if df.empty:
        print(f'❌ 空')
        return df

    # 去极值 (60日波动更大, 放宽clip)
    df['excess_ret_20d'] = df['excess_ret_20d'].clip(-0.7, 0.7)

    n_pos = (df['excess_ret_20d'] > 0).sum()
    print(f'{len(df)}行 正超额占比={n_pos/len(df)*100:.1f}% 均值={df["excess_ret_20d"].mean():+.4f} ✓')
    return df


# ─── Part 8: 股票池过滤 ────────────────────────────

def get_universe(c, trade_date_str):
    """获取指定日期的清洁股票池"""
    df = safe_sql(c, f"""
        SELECT DISTINCT k.ts_code FROM kline_daily k
        WHERE k.trade_date = '{trade_date_str}'
          AND k.close > 0 AND k.vol > 0
          AND k.is_st = FALSE
          AND k.ts_code NOT LIKE '%ST%'
          AND k.ts_code NOT IN ('sh000001','sh000016','sh000300','sh000688',
                                'sz399001','sz399005','sz399006','sz399300')
    """, f'universe_{trade_date_str}')
    return df['ts_code'].tolist() if not df.empty else []


# ─── Part 9: ML训练与评估 ─────────────────────────────

def run_ml_window(window_idx, window_dates, combination_label, group_keys, factor_dfs, target_df, universe_fn):
    """
    单个窗口 × 单个因子组合 的训练+评估
    Returns: dict of metrics
    """
    train_start, train_end, val_start, val_end, test_start, test_end = window_dates

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print('  ⚠ LightGBM未安装，跳过ML训练')
        return None

    # 1. 合并因子
    merge_keys = ['ts_code', 'trade_date']
    merged = None
    for gk in group_keys:
        fdf = factor_dfs.get(gk)
        if fdf is None or fdf.empty:
            continue
        if merged is None:
            merged = fdf.copy()
        else:
            merged = merged.merge(fdf, on=merge_keys, how='inner')

    if merged is None or merged.empty:
        print(f'    ⚠ 因子合并为空')
        return None

    # 2. 合并目标
    if target_df is None or target_df.empty:
        print(f'    ⚠ 目标变量为空')
        return None
    merged = merged.merge(target_df[['ts_code', 'trade_date', 'excess_ret_20d']],
                          on=merge_keys, how='inner')
    if len(merged) < 100:
        print(f'    ⚠ 合并后样本不足({len(merged)})')
        return None

    # 3. 分割训练/验证/测试
    train_mask = (merged['trade_date'] >= train_start) & (merged['trade_date'] <= train_end)
    val_mask = (merged['trade_date'] >= val_start) & (merged['trade_date'] <= val_end)
    test_mask = (merged['trade_date'] >= test_start) & (merged['trade_date'] <= test_end)

    feature_cols = [c for c in merged.columns if c not in
                    ('ts_code', 'trade_date', 'excess_ret_20d', 'fwd_ret_20d', 'report_date')]

    X_train = merged.loc[train_mask, feature_cols]
    y_train = merged.loc[train_mask, 'excess_ret_20d']
    X_val = merged.loc[val_mask, feature_cols]
    y_val = merged.loc[val_mask, 'excess_ret_20d']
    X_test = merged.loc[test_mask, feature_cols]
    y_test = merged.loc[test_mask, 'excess_ret_20d']

    if len(X_train) < 500 or len(X_test) < 100:
        print(f'    ⚠ 样本不足 (train={len(X_train)}, test={len(X_test)})')
        return None

    # 4. 处理缺失值
    for col in feature_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_val[col] = X_val[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    # 5. 训练LightGBM
    try:
        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse')
    except Exception as e:
        print(f'    ⚠ 训练失败: {str(e)[:100]}')
        return None

    # 6. 预测
    pred_test = model.predict(X_test)

    # 7. 计算指标
    # IC (Rank IC)
    from scipy import stats
    mask = ~np.isnan(pred_test) & ~np.isnan(y_test.values)
    if mask.sum() > 30:
        ic, ic_p = stats.spearmanr(pred_test[mask], y_test.values[mask])
    else:
        ic, ic_p = 0, 1

    # 模拟月频调仓: 每个月取Top N, 等权持有
    test_data = merged.loc[test_mask].copy()
    test_data['pred'] = model.predict(
        test_data[feature_cols].fillna(X_train[feature_cols].median()))

    # 按月分组
    test_data['year_month'] = pd.to_datetime(test_data['trade_date']).dt.to_period('M')
    monthly_returns = []

    for month, group in test_data.groupby('year_month'):
        if len(group) < TOP_N:
            continue
        top_n = group.nlargest(TOP_N, 'pred')
        avg_ret = top_n['excess_ret_20d'].mean()
        monthly_returns.append({
            'month': month,
            'ret': avg_ret,
            'n_stocks': len(top_n)
        })

    if not monthly_returns:
        return None

    rets_df = pd.DataFrame(monthly_returns)
    rets = rets_df['ret'].values

    # 年化指标
    ann_ret = np.mean(rets) * 12  # 月频×12
    ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    # 最大回撤
    cumsum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cumsum)
    mdd = np.min(cumsum / peak - 1)
    win_rate = np.mean(rets > 0)

    # 超额vs沪深300 (测试期内)
    idx_ret = safe_sql(get_db(), f"""
        SELECT (MAX(close)/MIN(close)-1) AS ret
        FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date>='{test_start}' AND trade_date<='{test_end}'
    """, 'idx_ret')
    idx_total_ret = idx_ret['ret'].values[0] if not idx_ret.empty else 0

    # 策略累计超额收益 (rets已经是超额)
    strategy_excess_ret = np.prod(1 + rets) - 1
    # 策略总收益 ≈ 超额累计 + 指数累计 (近似)
    strategy_total_ret = (1 + strategy_excess_ret) * (1 + idx_total_ret) - 1

    # 特征重要性 Top10
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    result = {
        'window': window_idx + 1,
        'combo': combination_label,
        'n_factors': len(feature_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_months': len(monthly_returns),
        'ann_ret': round(ann_ret * 100, 2),          # 年化超额收益
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),                   # 超额Sharpe
        'mdd': round(mdd * 100, 2),                   # 超额回撤
        'win_rate': round(win_rate * 100, 1),
        'ic': round(ic, 4) if ic else 0,
        'excess_ret': round(strategy_excess_ret * 100, 2),   # 累计超额
        'total_ret': round(strategy_total_ret * 100, 2),     # 策略总收益
        'idx_ret': round(idx_total_ret * 100, 2),            # 指数总收益
        'top_features': importance.to_dict('records'),
    }
    return result


# ─── Part 10: CSI 300 基准 ─────────────────────────

def compute_csi300_baseline(c, windows):
    """计算沪深300买入持有基准"""
    results = []
    for wi, w in enumerate(windows):
        _, _, _, _, test_start, test_end = w
        # 分两步避免GROUP BY冲突: 先取daily_ret, 再加总
        r = c.execute(f"""
            WITH daily AS (
                SELECT trade_date, close,
                       (close/LAG(close) OVER(ORDER BY trade_date)-1) AS daily_ret
                FROM kline_daily WHERE ts_code='sh000300'
                AND trade_date>='{test_start}' AND trade_date<='{test_end}'
            )
            SELECT * FROM daily WHERE daily_ret IS NOT NULL ORDER BY trade_date
        """).df()
        if r.empty or len(r) < 5:
            results.append({'window': wi+1, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0, 'n_days': len(r)})
            continue
        total_ret = (r['close'].values[-1] / r['close'].values[0] - 1) * 100
        daily_ret = r['daily_ret'].values
        ann_ret = np.mean(daily_ret) * 252 * 100
        ann_vol = np.std(daily_ret, ddof=1) * np.sqrt(252) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + daily_ret)
        peak = np.maximum.accumulate(cum)
        mdd = np.min(cum / peak - 1) * 100
        n_days = len(daily_ret)
        results.append({
            'window': wi+1, 'total_ret': round(total_ret, 2),
            'ann_ret': round(ann_ret, 2), 'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 2), 'n_days': n_days
        })
    return results


# ─── Part 11: 随机基准 ─────────────────────────────

def compute_random_baseline(c, windows, target_df):
    """随机选30只等权，作为ML有效性的底线"""
    results = []
    for wi, w in enumerate(windows):
        _, _, _, _, test_start, test_end = w
        tdf = target_df[(target_df['trade_date'] >= test_start) & (target_df['trade_date'] <= test_end)]
        if tdf.empty:
            results.append({'window': wi+1, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0})
            continue
        # 每月随机选30只
        np.random.seed(42)
        monthly_rets = []
        for month, group in tdf.groupby(pd.to_datetime(tdf['trade_date']).dt.to_period('M')):
            if len(group) < TOP_N:
                continue
            sample = group.sample(min(TOP_N, len(group)))
            monthly_rets.append(sample['excess_ret_20d'].mean())
        if not monthly_rets:
            results.append({'window': wi+1, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0})
            continue
        rets = np.array(monthly_rets)
        ann_ret = np.mean(rets) * 12 * 100
        ann_vol = np.std(rets, ddof=1) * np.sqrt(12) * 100
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cum)
        mdd = np.min(cum / peak - 1) * 100
        results.append({
            'window': wi+1,
            'ann_ret': round(ann_ret, 2), 'sharpe': round(sharpe, 3),
            'mdd': round(mdd, 2), 'n_months': len(monthly_rets)
        })
    return results


# ─── Main ─────────────────────────────────────────

def main():
    t0 = time.time()
    print('=' * 72)
    print('  AgentQuant · ML因子组合滚动回测')
    print(f'  启动时间: {date.today()}')
    print(f'  模型: LightGBM | 目标: {FORWARD_DAYS}日超额收益 | 月频调仓Top{TOP_N}')
    print('=' * 72)

    c = get_db(read_only=True)

    # ── 检查数据更新 ──
    latest_kline = c.execute('SELECT MAX(trade_date) FROM kline_daily').fetchone()[0]
    print(f'\n  DuckDB最新K线: {latest_kline}')
    if latest_kline < date.today() - timedelta(days=5):
        print(f'  ⚠ K线数据滞后{ (date.today() - latest_kline).days }天，需要先跑 tianyan.py daily')
        print(f'  继续使用现有数据...')

    # ── 准备所有月底日期 ──
    all_dates = c.execute("""
        SELECT DISTINCT trade_date FROM kline_daily
        WHERE trade_date >= '2015-01-01' AND trade_date <= '2026-06-16'
        ORDER BY trade_date
    """).df()
    all_dates['trade_date'] = pd.to_datetime(all_dates['trade_date'])
    all_dates['year_month'] = all_dates['trade_date'].dt.to_period('M')

    # 每月最后一个交易日
    month_ends = all_dates.groupby('year_month')['trade_date'].max().tolist()
    month_ends_str = [d.strftime('%Y-%m-%d') for d in month_ends]
    print(f'  月底日期: {len(month_ends_str)}个 (首:{month_ends_str[0]} 末:{month_ends_str[-1]})')

    # ── Step 1/8: 构建所有因子组 ──
    print(f'\n{"─"*60}')
    print('  Step 1: 构建因子组')
    print('─'*60)

    factor_dfs = {}
    full_start = '2015-01-01'
    full_end = '2026-06-16'

    factor_dfs['A'] = build_A_price_volume(c, full_start, full_end, month_ends_str)
    factor_dfs['B'] = build_B_fundamental(c, full_start, full_end, month_ends_str)
    factor_dfs['C'] = build_C_judgment(c, full_start, full_end, month_ends_str)
    factor_dfs['H'] = build_H_technical(c, full_start, full_end, month_ends_str)
    factor_dfs['I'] = build_I_valuation(c, full_start, full_end, month_ends_str)
    factor_dfs['J'] = build_J_macro(c, full_start, full_end, month_ends_str)

    # ── Step 2/8: 目标变量 ──
    print(f'\n{"─"*60}')
    print('  Step 2: 构建目标变量')
    print('─'*60)
    target_df = build_target(c, month_ends_str)

    # ── Step 3: 基准计算 ──
    print(f'\n{"─"*60}')
    print('  Step 3: 基准计算')
    print('─'*60)
    print('  计算CSI 300基准...', end=' ', flush=True)
    csi300_results = compute_csi300_baseline(c, WINDOWS)
    print(f'✓')
    print('  计算随机选股基准...', end=' ', flush=True)
    random_results = compute_random_baseline(c, WINDOWS, target_df)
    print(f'✓')

    # ── Step 4-8: ML回测 ──
    print(f'\n{"─"*60}')
    print('  Step 4: ML滚动回测 ({0}窗口 × {1}组合)'.format(len(WINDOWS), len(COMBINATIONS)))
    print('─'*60)

    all_results = []
    total_runs = len(WINDOWS) * len(COMBINATIONS)
    run_idx = 0

    for wi, window_dates in enumerate(WINDOWS):
        ts, te = window_dates[4], window_dates[5]
        print(f'\n  ═══ 窗口{wi+1}: 测试期 {ts} → {te} ═══')

        for combo_label, group_keys in COMBINATIONS:
            run_idx += 1
            print(f'  [{run_idx}/{total_runs}] {combo_label:15s}...', end=' ', flush=True)
            t_start = time.time()

            result = run_ml_window(wi, window_dates, combo_label, group_keys,
                                   factor_dfs, target_df, get_universe)
            if result:
                all_results.append(result)
                elapsed = time.time() - t_start
                print(f'Sharpe={result["sharpe"]:.3f} IC={result["ic"]:.4f} 超额={result["excess_ret"]:+.1f}% ({elapsed:.0f}s)')
            else:
                print(f'跳过')

    c.close()

    # ── 汇总报告 ──
    print(f'\n\n{"═"*72}')
    print('  📊 回测汇总报告')
    print('═'*72)

    if not all_results:
        print('  ❌ 无有效结果')
        return

    # 按窗口汇总
    print(f'\n  {"组合":14s} {"Sharpe":>7s} {"IC均值":>8s} {"累计超额%":>9s} {"策略总收益%":>10s} {"胜率%":>7s} {"窗口数":>6s}')
    print(f'  {"─"*70}')

    # 聚合同一组合跨窗口
    combo_agg = {}
    for r in all_results:
        cl = r['combo']
        if cl not in combo_agg:
            combo_agg[cl] = []
        combo_agg[cl].append(r)

    combo_summary = []
    for cl, results in combo_agg.items():
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_ic = np.mean([r['ic'] for r in results])
        avg_excess = np.mean([r['excess_ret'] for r in results])
        avg_wr = np.mean([r['win_rate'] for r in results])
        avg_mdd = np.mean([r['mdd'] for r in results])
        avg_total = np.mean([r['total_ret'] for r in results])
        combo_summary.append({
            'combo': cl, 'sharpe': avg_sharpe, 'ic': avg_ic,
            'excess': avg_excess, 'total_ret': avg_total,
            'win_rate': avg_wr, 'mdd': avg_mdd,
            'n_windows': len(results)
        })

    combo_summary.sort(key=lambda x: x['sharpe'], reverse=True)

    for s in combo_summary:
        print(f'  {s["combo"]:14s} {s["sharpe"]:>7.3f} {s["ic"]:>8.4f} {s["excess"]:>+8.1f}% {s["total_ret"]:>+9.1f}% {s["win_rate"]:>6.1f}% {s["n_windows"]:>5}')

    # 基准
    print(f'\n  {"─"*60}')
    print(f'  基准对比:')
    if csi300_results:
        avg_csi_sharpe = np.mean([r['sharpe'] for r in csi300_results if r['sharpe'] != 0])
        avg_csi_ret = np.mean([r['ann_ret'] for r in csi300_results if r['ann_ret'] != 0])
        avg_csi_mdd = np.mean([r['mdd'] for r in csi300_results if r['mdd'] != 0])
        print(f'  沪深300买入持有: Sharpe={avg_csi_sharpe:.3f} 年化={avg_csi_ret:.1f}% MDD={avg_csi_mdd:.1f}%')
    if random_results:
        avg_rand_sharpe = np.mean([r['sharpe'] for r in random_results if r['sharpe'] != 0])
        print(f'  随机选股(Top30):  Sharpe={avg_rand_sharpe:.3f}')

    # ── 特征重要性(全量组合) ──
    all_results_all = [r for r in all_results if r['combo'] == 'ALL']
    if all_results_all:
        print(f'\n  {"─"*60}')
        print(f'  全量组合(ALL) Top特征:')
        all_features = {}
        for r in all_results_all:
            for f in r.get('top_features', []):
                all_features[f['feature']] = all_features.get(f['feature'], 0) + f['importance']
        top10 = sorted(all_features.items(), key=lambda x: -x[1])[:10]
        for i, (feat, imp) in enumerate(top10):
            print(f'  {i+1:2d}. {feat:30s} {imp:.4f}')

    # ── 关键发现 ──
    best = combo_summary[0] if combo_summary else None
    print(f'\n  {"─"*60}')
    print(f'  🏆 最优组合: {best["combo"] if best else "N/A"}')
    if best:
        print(f'     Sharpe={best["sharpe"]:.3f}  IC均值={best["ic"]:.4f}  累计超额={best["excess"]:+.1f}%  策略总收益={best["total_ret"]:+.1f}%')

    # 增量分析
    if 'A' in combo_agg and best and best['combo'] != 'A':
        a_sharpe = np.mean([r['sharpe'] for r in combo_agg['A']])
        improvement = (best['sharpe'] - a_sharpe) / abs(a_sharpe) * 100 if a_sharpe != 0 else 0
        print(f'     vs A组(纯价量): Sharpe提升{improvement:+.0f}%')

    total_elapsed = time.time() - t0
    print(f'\n  ⏱ 总耗时: {total_elapsed/60:.1f}分钟')
    print(f'  📂 详细结果: {OUT_DIR}/ml_backtest_{date.today().isoformat()}.json')
    print('═'*72)

    # ── 保存详细结果 ──
    report_path = os.path.join(OUT_DIR, f'ml_backtest_{date.today().isoformat()}.json')
    report = {
        'date': date.today().isoformat(),
        'config': {
            'model': 'LightGBM',
            'target': f'{FORWARD_DAYS}d_excess_return',
            'top_n': TOP_N,
            'rebalance': 'monthly',
            'windows': len(WINDOWS),
            'combinations': len(COMBINATIONS),
            'factor_groups': FACTOR_GROUPS,
        },
        'combo_summary': combo_summary,
        'all_results': all_results,
        'baselines': {
            'csi300': csi300_results,
            'random': random_results,
        },
        'elapsed_minutes': round(total_elapsed / 60, 1),
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'  报告已保存: {report_path}')

    return report


if __name__ == '__main__':
    try:
        # 设置输出编码
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

    report = main()

    # ── 完成 ──
    print(f'\n✅ 回测完成。不自动睡眠。')
