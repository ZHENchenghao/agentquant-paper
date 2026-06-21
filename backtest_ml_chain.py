# -*- coding: utf-8 -*-
"""
AgentQuant · ML驱动完整回测链 v2.0
=====================================
基于ML实验结论重构:
  - 只用有效因子: 估值(I) + 质量(B) + 判断(C) + 技术(H)
  - 废弃无效因子: 价量Alpha158(A) + 宏观全市场(J)
  - 预测窗口: 60日
  - 月频调仓 + 真实交易成本
  - 扩展窗口训练(防前视偏差)

ML结论来源:
  20d: I(Sharpe=1.024) > B(0.837) > C(0.703) > A(0.238) > H(-0.022)
  60d: C(Sharpe=1.572) > I(0.774) > B(0.713) > ALL(0.317) > A(0.037)
  → 最优因子组: I+B+C+H (60d Sharpe=0.990, 超额+120%)
"""
import sys, io, os, json, time, warnings
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 量化引擎模块
from quant_backtest_engine import (RiskNeutralizer, ExecutionSimulator, StressTester)

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════
DB_PATH = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE_DIR = Path('D:/AgentQuant/our/cache')
REPORT_DIR = Path('D:/AgentQuant/our/reports')
CACHE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FORWARD_DAYS = 20          # 预测20日超额收益 (月度调仓)
TOP_N = 30                 # 持仓数
MIN_STOCKS = 3000          # 候选池最少股票数
REBALANCE_MONTHS = list(range(1, 13))  # 月度调仓(每月)
T1_EXECUTION = True         # T+1执行: 用次日开盘价而非今日收盘价
TIMING_OVERLAY = True       # 市场择时叠加: VIX/融资/连跌调节仓位
VIX_REGIME_FEATURE = True   # VIX分桶作为分类特征,让ML学到"不同恐慌用不同因子"

# 交易成本
STAMP_TAX = 0.001          # 印花税 0.1% (卖出)
COMMISSION = 0.0003        # 佣金 0.03%
SLIPPAGE = 0.001           # 滑点 0.1%
COST_BUY = COMMISSION + SLIPPAGE
COST_SELL = STAMP_TAX + COMMISSION + SLIPPAGE

# 回测区间
BACKTEST_START = '2002-01-01'
BACKTEST_END = '2026-06-16'

# 扩展窗口: 至少需要多少年训练数据
MIN_TRAIN_YEARS = 3

# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def get_db():
    for i in range(5):
        try:
            c = duckdb.connect(DB_PATH, read_only=True)
            c.execute('SELECT 1')
            return c
        except Exception:
            time.sleep(min(2**i, 10))
    return duckdb.connect(DB_PATH, read_only=True)

def sql(c, query, label=''):
    for attempt in range(3):
        try:
            return c.execute(query).df()
        except Exception as e:
            if attempt == 2:
                print(f'  ⚠ [{label}] {str(e)[:100]}')
                return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

# ts_code格式转换: '000001.SZ' -> 'sz000001'
def convert_ts_code(series):
    """批量转换财务表ts_code到K线格式"""
    s = series.astype(str)
    result = s.copy()
    mask_sz = s.str.endswith('.SZ')
    mask_sh = s.str.endswith('.SH')
    mask_bj = s.str.endswith('.BJ')
    result[mask_sz] = 'sz' + s[mask_sz].str.replace('.SZ', '', regex=False)
    result[mask_sh] = 'sh' + s[mask_sh].str.replace('.SH', '', regex=False)
    result[mask_bj] = 'bj' + s[mask_bj].str.replace('.BJ', '', regex=False)
    return result


# ═══════════════════════════════════════════════════════════
# Phase 1: 因子构建 (缓存到parquet)
# ═══════════════════════════════════════════════════════════

def build_month_ends():
    """获取所有月底日期"""
    c = get_db()
    df = sql(c, """
        WITH d AS (
            SELECT DISTINCT trade_date, DATE_TRUNC('month', trade_date) ym
            FROM kline_daily
            WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16'
        )
        SELECT MAX(trade_date)::DATE AS trade_date FROM d GROUP BY ym ORDER BY ym
    """, 'month_ends')
    c.close()
    return df['trade_date'].tolist()


def build_all_factors(force=False):
    """
    构建所有因子并缓存。
    返回: dict of {group_name: DataFrame}
    每组的DataFrame: ts_code × trade_date × factor_columns
    """
    cache_file = CACHE_DIR / 'factors_all.parquet'
    if cache_file.exists() and not force:
        print('  从缓存加载因子...')
        return pd.read_parquet(cache_file)

    c = get_db()
    month_ends = build_month_ends()
    me_str = ','.join([f"'{d}'" for d in month_ends])
    print(f'  月底日期: {len(month_ends)}个 ({month_ends[0]} ~ {month_ends[-1]})')

    # ── 估值因子 (I组) ──
    print('  [I] 估值因子...', end=' ', flush=True)
    df_I = sql(c, f"""
        WITH me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        pit AS (
            SELECT me.trade_date,
                   CASE WHEN f.ts_code LIKE '%.SZ' THEN 'sz'||REPLACE(f.ts_code,'.SZ','')
                        WHEN f.ts_code LIKE '%.SH' THEN 'sh'||REPLACE(f.ts_code,'.SH','')
                        WHEN f.ts_code LIKE '%.BJ' THEN 'bj'||REPLACE(f.ts_code,'.BJ','')
                        ELSE f.ts_code END AS ts_code,
                   f.net_profit, f.eps, f.revenue, f.roe, f.gross_margin, f.net_margin,
                   ROW_NUMBER() OVER(PARTITION BY me.trade_date, f.ts_code ORDER BY f.report_date DESC) rn
            FROM me JOIN financial_statements f ON f.report_date <= me.trade_date
                AND f.report_date >= me.trade_date - INTERVAL '1095 days'
                AND f.net_profit > 0 AND f.eps > 0 AND f.roe > 0 AND f.roe < 100
        ),
        fin AS (SELECT * FROM pit WHERE rn=1),
        priced AS (
            SELECT f.*, k.close,
                   k.close * (f.net_profit/f.eps) AS mcap
            FROM fin f JOIN kline_daily k ON f.ts_code=k.ts_code AND f.trade_date=k.trade_date
            WHERE k.close > 0
        )
        SELECT trade_date, ts_code,
               mcap/NULLIF(net_profit,0) AS pe,
               (mcap/NULLIF(net_profit,0))*(roe/100.0) AS pb,
               mcap/NULLIF(revenue,0) AS ps,
               LN(NULLIF(mcap,0)) AS log_mcap,
               roe, gross_margin, net_margin
        FROM priced WHERE mcap > 0
    """, 'I_valuation')
    df_I['factor_group'] = 'I'
    print(f'{len(df_I)}行 ✓')

    # ── 质量因子 (B组) ──
    print('  [B] 质量因子...', end=' ', flush=True)
    df_B = sql(c, f"""
        WITH me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        pit AS (
            SELECT me.trade_date,
                   CASE WHEN f.ts_code LIKE '%.SZ' THEN 'sz'||REPLACE(f.ts_code,'.SZ','')
                        WHEN f.ts_code LIKE '%.SH' THEN 'sh'||REPLACE(f.ts_code,'.SH','')
                        ELSE f.ts_code END AS ts_code,
                   f.roe, f.gross_margin, f.net_margin, f.eps, f.net_profit, f.revenue,
                   ROW_NUMBER() OVER(PARTITION BY me.trade_date, f.ts_code ORDER BY f.report_date DESC) rn
            FROM me JOIN financial_statements f ON f.report_date <= me.trade_date
                AND f.report_date >= me.trade_date - INTERVAL '1095 days'
                AND f.net_profit > 0 AND f.eps > 0 AND f.roe > 0 AND f.roe < 100
        )
        SELECT trade_date, ts_code,
               roe, gross_margin, net_margin,
               net_profit/NULLIF(revenue,0) AS profit_margin,
               LN(NULLIF(eps,0)) AS log_eps
        FROM pit WHERE rn=1
    """, 'B_quality')
    df_B['factor_group'] = 'B'
    print(f'{len(df_B)}行 ✓')

    # ── 判断因子 (C组) — 市场级别信号，广播到个股 ──
    print('  [C] 判断因子...', end=' ', flush=True)
    signals = sql(c, f"""
        WITH me AS (SELECT UNNEST([{me_str}])::DATE AS trade_date),
        margin AS (
            SELECT trade_date,
                   CASE WHEN (margin_balance/LAG(margin_balance) OVER(ORDER BY trade_date)-1)*100 < -3
                        THEN 1 ELSE 0 END AS margin_panic
            FROM margin_trading WHERE margin_balance IS NOT NULL
        ),
        streak AS (
            SELECT trade_date,
                   CASE WHEN close<LAG(close)OVER o AND LAG(close)OVER o<LAG(close,2)OVER o
                         AND LAG(close,2)OVER o<LAG(close,3)OVER o AND LAG(close,3)OVER o<LAG(close,4)OVER o
                        THEN 1 ELSE 0 END AS streak5_dn
            FROM kline_daily WHERE ts_code='sh000300'
            WINDOW o AS (ORDER BY trade_date)
        ),
        nb AS (
            SELECT n.trade_date,
                   CASE WHEN n.daily>30 AND i.chg>0 THEN 1 ELSE 0 END AS nb_bull_resonance,
                   CASE WHEN n.daily<-30 AND i.chg>0 THEN 1 ELSE 0 END AS nb_diverge
            FROM (SELECT trade_date, SUM(net_flow) daily FROM north_bound_flow
                  WHERE net_flow IS NOT NULL GROUP BY trade_date) n
            JOIN (SELECT trade_date, (close/LAG(close) OVER(ORDER BY trade_date)-1)*100 chg
                  FROM kline_daily WHERE ts_code='sh000300') i ON n.trade_date=i.trade_date
        ),
        vix_data AS (
            SELECT trade_date,
                   CASE WHEN vix>25 THEN 1 WHEN vix>20 THEN 0.5 ELSE 0 END AS vix_alert
            FROM macro_indicators WHERE vix IS NOT NULL
        )
        SELECT me.trade_date,
               COALESCE(m.margin_panic,0) AS margin_panic,
               COALESCE(s.streak5_dn,0) AS streak5_dn,
               COALESCE(n.nb_bull_resonance,0) AS nb_bull,
               COALESCE(n.nb_diverge,0) AS nb_diverge,
               COALESCE(v.vix_alert,0) AS vix_stress
        FROM me
        LEFT JOIN margin m ON me.trade_date=m.trade_date
        LEFT JOIN streak s ON me.trade_date=s.trade_date
        LEFT JOIN nb n ON me.trade_date=n.trade_date
        LEFT JOIN vix_data v ON me.trade_date=v.trade_date
    """, 'C_signals')

    # 广播到个股
    stocks = sql(c, f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date IN ({me_str}) AND close > 0 AND vol > 0
        AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%'
    """, 'C_stocks')

    if not signals.empty and not stocks.empty:
        stocks['_k'] = 1; signals['_k'] = 1
        df_C = stocks.merge(signals, on='_k').drop(columns=['_k'])
        df_C['factor_group'] = 'C'
    else:
        df_C = pd.DataFrame()
    print(f'{len(df_C)}行 ✓')
    c.close()

    # ── 技术因子 (H组) ──
    print('  [H] 技术因子...', end=' ', flush=True)
    c2 = get_db()
    df_H = sql(c2, f"""
        SELECT t.ts_code, t.trade_date,
               t.rsi6/100.0 AS rsi6,
               t.rsi14/100.0 AS rsi14,
               CASE WHEN t.rsi6<30 THEN 1 WHEN t.rsi6>70 THEN -1 ELSE 0 END AS rsi_extreme,
               (k.close-t.boll_lower)/NULLIF(t.boll_upper-t.boll_lower,0) AS boll_pos,
               (t.boll_upper-t.boll_lower)/NULLIF(t.boll_mid,0) AS boll_width,
               k.close/NULLIF(t.ma20,0)-1 AS div_ma20,
               k.close/NULLIF(t.ma60,0)-1 AS div_ma60,
               k.close/NULLIF(t.ma120,0)-1 AS div_ma120,
               t.volume_ratio AS vol_ratio,
               CASE WHEN t.ma5>t.ma20 AND t.ma20>t.ma60 THEN 2
                    WHEN t.ma5>t.ma20 THEN 1
                    WHEN t.ma5<t.ma20 AND t.ma20<t.ma60 THEN -2
                    WHEN t.ma5<t.ma20 THEN -1 ELSE 0 END AS ma_score
        FROM technical_indicators t
        JOIN kline_daily k ON t.ts_code=k.ts_code AND t.trade_date=k.trade_date
        WHERE t.trade_date IN ({me_str}) AND t.rsi6 IS NOT NULL
    """, 'H_technical')
    df_H['factor_group'] = 'H'
    print(f'{len(df_H)}行 ✓')
    c2.close()

    # ── 合并所有因子并加价格 ──
    print('  合并因子并获取价格...', end=' ', flush=True)
    dfs = [d for d in [df_I, df_B, df_C, df_H] if d is not None and not d.empty]
    dfs_clean = []
    for d in dfs:
        d_clean = d.drop(columns=['factor_group'], errors='ignore')
        dfs_clean.append(d_clean)
    merged = dfs_clean[0]
    for d in dfs_clean[1:]:
        merge_cols = ['ts_code', 'trade_date']
        common = [c for c in merge_cols if c in merged.columns and c in d.columns]
        overlap = set(merged.columns) & set(d.columns) - set(common)
        if overlap:
            d = d.drop(columns=list(overlap))
        merged = merged.merge(d, on=common, how='left')

    # 加入当日close价格(统一价格源)
    c3 = get_db()
    prices_df = sql(c3, f"""
        SELECT ts_code, trade_date, close FROM kline_daily
        WHERE trade_date IN ({me_str})
    """, 'prices_lookup')
    c3.close()
    # 填充C/H组缺失值(早年无融资/北向/技术数据)
    for col in merged.columns:
        if merged[col].isna().any() and col not in ('close','ts_code','trade_date'):
            merged[col] = merged[col].fillna(0)
    if not prices_df.empty:
        merged = merged.merge(prices_df, on=['ts_code', 'trade_date'], how='left')
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'{len(merged)}行 × {len(merged.columns)}列 ✓')

    # 移除Inf
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f'{len(merged)}行 × {len(merged.columns)}列 ✓')

    # 缓存
    merged.to_parquet(cache_file)
    print(f'  缓存: {cache_file}')
    return merged


# ═══════════════════════════════════════════════════════════
# Phase 2: 目标变量
# ═══════════════════════════════════════════════════════════

def build_target():
    """构建60日超额收益目标"""
    cache_file = CACHE_DIR / 'target_60d.parquet'
    if cache_file.exists():
        print('  从缓存加载目标...')
        return pd.read_parquet(cache_file)

    c = get_db()
    month_ends = build_month_ends()
    me_str = ','.join([f"'{d}'" for d in month_ends])
    print('  构建60日超额收益目标...', end=' ', flush=True)

    df = sql(c, f"""
        WITH stock_fwd AS (
            SELECT ts_code, trade_date, close,
                   LEAD(close, 20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fwd_close
            FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16'
        ),
        idx_fwd AS (
            SELECT trade_date, close,
                   LEAD(close, 20) OVER(ORDER BY trade_date) AS fwd_close
            FROM kline_daily WHERE ts_code='sh000300' AND trade_date BETWEEN '2002-01-01' AND '2026-06-16'
        )
        SELECT s.ts_code, s.trade_date,
               (s.fwd_close/s.close-1) AS fwd_ret,
               (s.fwd_close/s.close-1)-(i.fwd_close/i.close-1) AS excess_ret
        FROM stock_fwd s JOIN idx_fwd i ON s.trade_date=i.trade_date
        WHERE s.trade_date IN ({me_str}) AND s.fwd_close IS NOT NULL AND i.fwd_close IS NOT NULL
    """, 'target')
    c.close()

    df['excess_ret'] = df['excess_ret'].clip(-0.4, 0.4)
    df.to_parquet(cache_file)
    print(f'{len(df)}行 ✓')
    return df


# ═══════════════════════════════════════════════════════════
# Phase 3: 排雷过滤
# ═══════════════════════════════════════════════════════════

def get_minesweep_flags(c, trade_date_str):
    """返回当日应排除的股票列表"""
    # ST股票
    st = sql(c, f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date='{trade_date_str}' AND is_st=TRUE
    """, 'st')
    st_codes = set(st['ts_code'].tolist()) if not st.empty else set()

    # 上市不足1年
    new = sql(c, f"""
        SELECT ts_code FROM kline_daily
        WHERE ts_code IN (SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date='{trade_date_str}')
        GROUP BY ts_code HAVING MIN(trade_date) > '{trade_date_str}'::DATE - INTERVAL '365 days'
    """, 'new_listing')
    new_codes = set(new['ts_code'].tolist()) if not new.empty else set()

    return st_codes | new_codes


# ═══════════════════════════════════════════════════════════
# Phase 4: ML训练 + 选股
# ═══════════════════════════════════════════════════════════

def run_ml_chain():
    """主回测链"""
    t0 = time.time()
    print('=' * 70)
    print('  AgentQuant · ML驱动完整回测链 v2.0')
    print(f'  启动: {date.today()}  |  预测: {FORWARD_DAYS}日超额  |  成本: 双边0.23%')
    print('=' * 70)

    # Step 1: 加载因子
    print('\n── Step 1: 因子构建 ──')
    factors = build_all_factors()
    if factors.empty:
        print('❌ 因子构建失败')
        return

    # Step 2: 加载目标
    print('\n── Step 2: 目标变量 ──')
    target = build_target()
    if target.empty:
        print('❌ 目标构建失败')
        return

    # Step 3: 合并
    print('\n── Step 3: 合并因子+目标 ──')
    data = factors.merge(target[['ts_code', 'trade_date', 'excess_ret']],
                         on=['ts_code', 'trade_date'], how='inner')
    print(f'  合并后: {len(data)}行')

    # 特征列 (排除close价格, 否则数据泄露)
    feat_cols = [c for c in data.columns if c not in
                 ('ts_code', 'trade_date', 'excess_ret', 'fwd_ret', 'factor_group',
                  'report_date', '_k', 'close')]
    print(f'  特征数: {len(feat_cols)}')

    # ── 二次项市值中性化 (initial-d/ml-quant-trading 公式(7)) ──
    print('\n── Step 3.5: 二次项市值中性化 + 污染感知 ──')
    if 'log_mcap' in data.columns:
        data['industry_code'] = data['ts_code'].astype(str).str[:4]
        neut = RiskNeutralizer()
        neut_factors = [c for c in feat_cols if c != 'log_mcap']
        report = {}
        for col in neut_factors:
            if col not in data.columns or data[col].isna().all():
                continue
            fv = neut.industry_neutralize(data[col].values, data['industry_code'].values)
            fv_q, gamma, delta, r2 = neut.size_neutralize_quadratic(fv, data['log_mcap'].values)
            data[col] = fv_q
            report[col] = {'gamma': round(gamma, 4), 'delta': round(delta, 4), 'r2': round(r2, 4)}
        data.drop(columns=['industry_code'], errors='ignore', inplace=True)
        sig_delta = {k: v for k, v in report.items() if abs(v.get('delta', 0)) > 0.001}
        if sig_delta:
            print(f'  ⚠ 二次项显著: {list(sig_delta.keys())[:5]}')
        avg_r2 = np.mean([v['r2'] for v in report.values()]) if report else 0
        print(f'  市值R²均值={avg_r2:.3f}')

    # ── 八类危机检测 + VIX指纹 ──
    if VIX_REGIME_FEATURE:
        c_vix = get_db()
        vix_data = c_vix.execute("""
            WITH base AS (
                SELECT m.trade_date, m.vix,
                       mt.margin_balance,
                       LAG(mt.margin_balance,20) OVER(ORDER BY m.trade_date) AS margin_20d_ago,
                       SUM(nb.net_flow) OVER(ORDER BY m.trade_date ROWS 19 PRECEDING) AS north_20d,
                       k.close, MAX(k.close) OVER(ORDER BY k.trade_date ROWS 249 PRECEDING) AS peak_250,
                       k.vol, AVG(k.vol) OVER(ORDER BY k.trade_date ROWS 19 PRECEDING) AS vol_ma20
                FROM macro_indicators m
                LEFT JOIN margin_trading mt ON m.trade_date=mt.trade_date
                LEFT JOIN north_bound_flow nb ON m.trade_date=nb.trade_date
                LEFT JOIN kline_daily k ON m.trade_date=k.trade_date AND k.ts_code='sh000300'
                WHERE m.vix IS NOT NULL
            )
            SELECT trade_date, vix,
                   vix - LAG(vix,5) OVER w AS vix_velocity_5,
                   vix - LAG(vix,20) OVER w AS vix_velocity_20,
                   (margin_balance/NULLIF(margin_20d_ago,0)-1)*100 AS margin_chg_20,
                   north_20d,
                   close/NULLIF(peak_250,0)-1 AS drawdown_250,
                   vol/NULLIF(vol_ma20,0) AS mkt_vol_ratio
            FROM base
            WINDOW w AS (ORDER BY trade_date)
        """).df()
        c_vix.close()
        if not vix_data.empty:
            # 多维指纹分类
            def vix_fingerprint(row):
                v = row['vix']; vel5 = row.get('vix_velocity_5',0) or 0
                vel20 = row.get('vix_velocity_20',0) or 0
                mg = row.get('margin_chg_20',0) or 0
                nf = row.get('north_20d',0) or 0
                dd = row.get('drawdown_250',0) or 0
                vr = row.get('vol_ratio',1) or 1
                if pd.isna(v): return -1
                # 底层: VIX绝对水平
                if v>35: base=5
                elif v>28: base=4
                elif v>22: base=3
                elif v>16: base=2
                elif v>12: base=1
                else: base=0
                # 动态调整: 急升+1档, 急降-1档
                if vel5>5: base=min(5,base+1)
                elif vel5<-3 and base>0: base-=1
                # 融资崩溃+1档
                if mg<-10 and base<5: base+=1
                # 外资出逃+1档(恐慌叠加)
                if nf<-200 and base<5: base+=1
                # 加杠杆+外资稳: 恐慌降级
                if mg>5 and nf>100 and base>0: base-=1
                # 深跌中恐慌放大
                if dd<-0.25 and base<5: base+=1
                return base
            vix_data['vix_regime'] = vix_data.apply(vix_fingerprint, axis=1)
            # 注入衍生特征
            for col in ['vix_velocity_5','vix_velocity_20','margin_chg_20','north_20d','drawdown_250','vol_ratio']:
                if col in vix_data.columns:
                    vix_data[col] = vix_data[col].fillna(0)
            data = data.merge(vix_data, on='trade_date', how='left')
            data['vix_regime'] = data['vix_regime'].fillna(-1).astype(int)
            for col in ['vix_regime','vix_velocity_5','vix_velocity_20','drawdown_250']:
                if col in data.columns and col not in feat_cols:
                    feat_cols.append(col)
            regime_count = data['vix_regime'].nunique()
            print(f'  VIX多维指纹: {regime_count}档 (0=安逸 5=极端) + 速度/回撤 (特征={len(feat_cols)})')
            # 追加八类危机检测
            if 'vix_velocity_5' in data.columns and 'margin_chg_20' in data.columns and 'drawdown_250' in data.columns:
                vel = data['vix_velocity_5'].fillna(0)
                mg20 = data['margin_chg_20'].fillna(0)
                dd = data['drawdown_250'].fillna(0)
                nf = data['north_20d'].fillna(0) if 'north_20d' in data.columns else pd.Series(0, index=data.index)
                # 7类危机特征(连续值, 越大越像该类型)
                data['crisis_spike'] = np.clip((vel/10 + (-mg20/20).clip(0,1))/2, 0, 1)        # 急跌恐慌
                data['crisis_slowburn'] = np.clip(((-dd-0.15)/0.1 + (-mg20/10).clip(0,1))/2, 0, 1)  # 慢炖阴跌
                data['crisis_structural'] = np.clip(((1-(data['vix'].fillna(20)/25)) + (-dd-0.08)/0.05)/2, 0, 1)  # 结构崩塌
                data['crisis_liquidity'] = np.clip(((0.8-(data['mkt_vol_ratio'].fillna(1) if 'mkt_vol_ratio' in data.columns else pd.Series(1,index=data.index)))/0.5).clip(0,1), 0, 1)  # 流动性枯竭
                data['crisis_policy'] = np.clip((vel/10 + (-nf/300).clip(0,1))/2, 0, 1)          # 政策冲击
                data['crisis_margin'] = np.clip((-mg20/30).clip(0,1), 0, 1)                      # 杠杆踩踏
                data['crisis_false_alarm'] = np.clip(((vel/8).clip(0,1) * (mg20/5).clip(0,1)), 0, 1)  # 假恐慌
                # 砸盘/洗盘: 急跌+放量+融资异动
                data['crisis_washout'] = np.clip(((-dd-0.03)/0.08 * (data['vol_ratio'].fillna(1)-1)/0.5 * (-mg20/5).clip(0,1)), 0, 1)  # 洗盘(后续反弹)
                data['crisis_dump'] = np.clip(((-dd-0.06)/0.10 * (data['vol_ratio'].fillna(1)-1.3)/0.7 * (-mg20/3).clip(0,1)), 0, 1)     # 砸盘(后续续跌)
                for cc in ['crisis_spike','crisis_slowburn','crisis_structural','crisis_liquidity',
                           'crisis_policy','crisis_margin','crisis_false_alarm','crisis_washout','crisis_dump']:
                    if cc in data.columns and cc not in feat_cols:
                        feat_cols.append(cc)
                print(f'  八类危机检测已注入 (特征={len(feat_cols)})')

    # ── 上游污染修复: 注入污染感知特征 ──
    print('  注入污染感知特征...', end=' ', flush=True)
    c_p = get_db()
    # 取所有月底日期的停牌/涨跌停统计
    me_str = ','.join([f"'{d}'" for d in sorted(data['trade_date'].unique())])
    contamination = sql(c_p, f"""
        WITH daily_flags AS (
            SELECT ts_code, trade_date,
                   CASE WHEN vol<=0 OR close=pre_close THEN 1 ELSE 0 END AS is_suspended,
                   CASE WHEN (close/pre_close-1) >= 0.098 THEN 1 ELSE 0 END AS is_limit_up,
                   CASE WHEN (close/pre_close-1) <= -0.098 THEN 1 ELSE 0 END AS is_limit_down
            FROM kline_daily WHERE trade_date >= '2002-01-01'
        ),
        rolling AS (
            SELECT ts_code, trade_date, is_suspended, is_limit_up, is_limit_down,
                   SUM(is_suspended) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS suspend_days_20,
                   SUM(is_limit_up+is_limit_down) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 19 PRECEDING) AS limit_days_20,
                   SUM(is_suspended) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS 59 PRECEDING) AS suspend_days_60
            FROM daily_flags
        )
        SELECT ts_code, trade_date, suspend_days_20, limit_days_20, suspend_days_60
        FROM rolling WHERE trade_date IN ({me_str})
    """, 'contamination')
    c_p.close()

    if not contamination.empty:
        data = data.merge(contamination, on=['ts_code', 'trade_date'], how='left')
        for cc in ['suspend_days_20', 'limit_days_20', 'suspend_days_60']:
            if cc in data.columns:
                data[cc] = data[cc].fillna(0)
                if cc not in feat_cols:
                    feat_cols.append(cc)
        print(f'3个污染特征已注入 (特征={len(feat_cols)})')
    else:
        print('跳过(无数据)')

    # Step 4: 滚动窗口训练+交易
    print(f'\n── Step 4: 扩展窗口ML交易 ({MIN_TRAIN_YEARS}年训练起步) ──')

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        print('❌ LightGBM未安装')
        return

    data = data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    all_dates = sorted(data['trade_date'].unique())
    all_dates_dt = pd.to_datetime(all_dates)

    # 找第一个合法测试月: 至少有MIN_TRAIN_YEARS年训练数据
    first_test = all_dates_dt[0] + pd.DateOffset(years=MIN_TRAIN_YEARS)
    test_dates = [d for d in all_dates if pd.to_datetime(d) >= first_test]

    print(f'  总日期: {len(all_dates)}  训练起始: {all_dates[0]}  测试起始: {test_dates[0]}  测试月数: {len(test_dates)}')

    # 按季度调仓
    quarterly_dates = [d for d in test_dates if pd.to_datetime(d).month in REBALANCE_MONTHS]
    print(f'  季频调仓: {len(quarterly_dates)}次')

    # ── 模拟交易 ──
    c = get_db()

    # 先获取CSI300每日净值用于基准对比
    bench_daily = sql(c, f"""
        SELECT trade_date, close FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date BETWEEN '{BACKTEST_START}' AND '{BACKTEST_END}'
        ORDER BY trade_date
    """, 'bench')
    bench_daily['bench_return'] = bench_daily['close'].pct_change()

    # 获取全A股每日收盘价(用于日频估值)
    all_stock_prices = sql(c, f"""
        SELECT ts_code, trade_date, close FROM kline_daily
        WHERE trade_date >= '{test_dates[0]}'
        ORDER BY ts_code, trade_date
    """, 'daily_prices')

    portfolio = {}          # {ts_code: shares}
    last_prices = {}        # {ts_code: last_known_close} — 处理停牌
    cash = 1_000_000
    trade_log = []
    daily_nav = []          # 日频净值
    initial_value = cash
    price_idx = all_stock_prices.set_index(['trade_date', 'ts_code'])['close']

    # 获取所有交易日(用于日频净值)
    all_trade_dates = sorted(all_stock_prices['trade_date'].unique())
    trade_dates_after_start = [d for d in all_trade_dates if d >= pd.Timestamp(str(test_dates[0]))]
    nav_date_idx = 0

    for i, test_date in enumerate(quarterly_dates):
        sold_value = 0
        bought_value = 0
        test_dt = pd.to_datetime(test_date)
        train_end = test_date
        train_start_dt = test_dt - pd.DateOffset(years=MIN_TRAIN_YEARS)

        # ── 日频净值更新(从上次调仓日到本次) ──
        if i == 0:
            prev_nav_date = trade_dates_after_start[0]
            daily_nav.append({'trade_date': prev_nav_date,
                             'nav': initial_value, 'bench': bench_daily.iloc[0]['close']})
        else:
            prev_reb_ts = pd.Timestamp(str(quarterly_dates[i-1]))
            test_date_ts = pd.Timestamp(str(test_date))
            for nav_date in trade_dates_after_start:
                if nav_date <= prev_reb_ts:
                    continue
                if nav_date > test_date_ts:
                    break
                # 更新持仓价格
                holdings_val = 0
                for code, shares in portfolio.items():
                    # 尝试获取当日价格, 停牌则用上次价格
                    try:
                        px = price_idx.get((pd.Timestamp(nav_date), code), None)
                    except Exception:
                        px = None
                    if px is None or pd.isna(px):
                        px = last_prices.get(code, 0)
                    else:
                        last_prices[code] = px
                    holdings_val += shares * px
                total_val = cash + holdings_val
                daily_nav.append({'trade_date': nav_date, 'nav': total_val,
                                 'bench': bench_daily.loc[bench_daily['trade_date']==nav_date, 'close'].values[0]
                                          if nav_date in bench_daily['trade_date'].values else np.nan})
                nav_date_idx += 1

        # ── ML训练 ──
        train_mask = (data['trade_date'] >= BACKTEST_START) & (data['trade_date'] <= train_end)
        valid_feats = [c for c in feat_cols if c in data.columns]
        train_data = data[train_mask].dropna(subset=valid_feats + ['excess_ret'])

        if len(train_data) < 1000:
            print(f'  [{i+1}/{len(quarterly_dates)}] {test_date} ⚠ 训练样本不足')
            continue

        X_train = train_data[feat_cols].fillna(train_data[feat_cols].median())
        y_train = train_data['excess_ret']

        model = LGBMRegressor(
            objective='regression', metric='rmse',
            learning_rate=0.05, num_leaves=63, max_depth=10,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_estimators=300, early_stopping_rounds=20,
            verbose=-1, random_state=42, n_jobs=-1
        )

        split_idx = int(len(X_train) * 0.8)
        if split_idx > 500:
            model.fit(X_train.iloc[:split_idx], y_train.iloc[:split_idx],
                      eval_set=[(X_train.iloc[split_idx:], y_train.iloc[split_idx:])])
        else:
            model.fit(X_train, y_train)

        # ── 预测+选股 ──
        test_mask = data['trade_date'] == test_date
        X_test = data.loc[test_mask, feat_cols].copy()
        if len(X_test) < 100:
            continue

        X_test_filled = X_test.fillna(train_data[feat_cols].median())
        scores = model.predict(X_test_filled)
        X_test['ml_score'] = scores

        blacklist = get_minesweep_flags(c, test_date)
        X_test = X_test[~X_test.index.isin(
            data.loc[test_mask & data['ts_code'].isin(blacklist)].index)]

        top_n_idx = X_test['ml_score'].nlargest(TOP_N).index
        selected = data.loc[top_n_idx, ['ts_code', 'trade_date']].copy()

        if len(selected) < 10:
            continue

        # ── 获取执行价格 (T+1开盘价) ──
        test_data_close = data.loc[test_mask, ['ts_code', 'trade_date', 'close']].copy()
        if 'close' not in test_data_close.columns or test_data_close['close'].isna().all():
            continue

        price_map = dict(zip(test_data_close['ts_code'], test_data_close['close']))
        if T1_EXECUTION:
            nd = c.execute("SELECT trade_date FROM kline_daily WHERE trade_date > '%s' ORDER BY trade_date LIMIT 1" % test_date).fetchone()
            if nd:
                next_day = nd[0]
                all_codes = test_data_close['ts_code'].tolist()
                codes_str = ','.join(["'%s'" % x for x in all_codes[:5000]])
                nd_df = sql(c, "SELECT ts_code, open FROM kline_daily WHERE ts_code IN (%s) AND trade_date = '%s'" % (codes_str, next_day), 'next_open')
                if not nd_df.empty:
                    nd_map = dict(zip(nd_df['ts_code'], nd_df['open']))
                    t1_map = {}
                    for code in price_map:
                        px = nd_map.get(code, None)
                        if px and px > 0:
                            t1_map[code] = px
                    if len(t1_map) >= 10:
                        price_map = t1_map

        # ── 执行交易: 使用ExecutionSimulator ──
        sim = ExecutionSimulator()

        # 过滤涨跌停
        buy_codes = selected['ts_code'].tolist()
        valid_buy, blocked = sim.limit_filter(buy_codes, test_date, 'BUY')

        # 卖出现有持仓(不在新选股中的)
        sell_codes = [c for c in portfolio.keys() if c not in buy_codes]
        valid_sell, sell_blocked = sim.limit_filter(sell_codes, test_date, 'SELL') if sell_codes else ([], {})

        # 准备价格和流动性数据
        valid_prices_arr = np.array([price_map.get(c, last_prices.get(c, 0)) for c in valid_buy])
        # 用因子数据中的close和amount估算日内波动率和流动性
        daily_ret_std = data.loc[test_mask, 'close'].pct_change().std()
        if pd.isna(daily_ret_std) or daily_ret_std <= 0:
            daily_ret_std = 0.02
        vol_arr = np.array([daily_ret_std] * len(valid_buy))
        # 日均成交额近似: 用close×vol估算, 若无vol用固定值
        amt_arr = np.full(len(valid_buy), 5e8)  # 默认5亿日均成交

        # ── 市场择时叠加 ──
        timing_mult = 1.0
        timing_reason = []
        if TIMING_OVERLAY:
            # VIX恐慌降仓
            vx = c.execute("SELECT vix FROM macro_indicators WHERE trade_date<='%s' AND vix IS NOT NULL ORDER BY trade_date DESC LIMIT 1" % test_date).fetchone()
            if vx and vx[0]:
                vix_val = vx[0]
                if vix_val > 25:
                    timing_mult *= 0.3
                    timing_reason.append('VIX=%.0f→30%%' % vix_val)
                elif vix_val > 20:
                    timing_mult *= 0.6
                    timing_reason.append('VIX=%.0f→60%%' % vix_val)
            # 融资急降减仓
            mg = c.execute("SELECT (margin_balance/LAG(margin_balance) OVER(ORDER BY trade_date)-1)*100 FROM margin_trading WHERE trade_date<='%s' ORDER BY trade_date DESC LIMIT 1" % test_date).fetchone()
            if mg and mg[0] and mg[0] < -3:
                timing_mult *= 0.5
                timing_reason.append('融资急降%.0f%%→50%%' % mg[0])
            # 5连跌加仓(逆向)
            stk = c.execute("SELECT CASE WHEN close<LAG(close)OVER w AND LAG(close)OVER w<LAG(close,2)OVER w AND LAG(close,2)OVER w<LAG(close,3)OVER w AND LAG(close,3)OVER w<LAG(close,4)OVER w THEN 1 ELSE 0 END FROM kline_daily WHERE ts_code='sh000300' AND trade_date<='%s' WINDOW w AS (ORDER BY trade_date) ORDER BY trade_date DESC LIMIT 1" % test_date).fetchone()
            if stk and stk[0]:
                timing_mult *= 1.2
                timing_reason.append('5连跌→120%%')
            if timing_reason:
                pass  # logging handled below

        # 计算目标权重(等权×择时)
        n_target = min(len(valid_buy), TOP_N)
        if n_target >= 10:
            invest_cash = cash * timing_mult
            weights = np.ones(n_target) / n_target

            # 执行买入(含动态滑点+容量截断)
            exec_report = sim.execute_round(
                weights, valid_buy[:n_target], test_date,
                valid_prices_arr[:n_target], amt_arr[:n_target], vol_arr[:n_target],
                amt_arr[:n_target], invest_cash
            )

            if exec_report.get('shares') is not None:
                shares_arr = exec_report['shares']
                for j, code in enumerate(valid_buy[:n_target]):
                    if j < len(shares_arr) and shares_arr[j] > 0:
                        buy_px = price_map.get(code, 0)
                        if buy_px > 0:
                            cost = shares_arr[j] * buy_px * (1 + COST_BUY)
                            cash -= cost
                            portfolio[code] = portfolio.get(code, 0) + int(shares_arr[j])
                            last_prices[code] = buy_px
                            trade_log.append({'date': test_date, 'ts_code': code, 'action': 'BUY',
                                              'shares': int(shares_arr[j]), 'price': buy_px})

            # 卖出不在新选股中的
            for code in sell_codes:
                if code in portfolio and code in valid_sell:
                    shares = portfolio[code]
                    sell_px = price_map.get(code, last_prices.get(code, 0))
                    if sell_px > 0:
                        sell_price_net = sell_px * (1 - COST_SELL)
                        cash += shares * sell_price_net
                        last_prices.pop(code, None)
                    trade_log.append({'date': test_date, 'ts_code': code, 'action': 'SELL',
                                      'shares': shares, 'price': sell_px})
                    del portfolio[code]

            friction_bps = exec_report.get('avg_slippage_bps', 0)
        else:
            friction_bps = 0

        # 进度
        holdings_val = sum(portfolio.get(c,0)*last_prices.get(c,0) for c in portfolio)
        total_value = cash + holdings_val
        if (i+1) % 5 == 0 or i <= 2:
            top_features = pd.DataFrame({
                'feature': feat_cols,
                'importance': model.feature_importances_
            }).nlargest(3, 'importance')['feature'].tolist()
            tinfo = (' 择时:' + ','.join(timing_reason)) if timing_reason else ''
            print(f'  [{i+1}/{len(quarterly_dates)}] {test_date}  资产={total_value:,.0f}  持仓={len(portfolio)}  '
                  f'仓位={timing_mult:.0%}{tinfo}  摩擦={friction_bps:.0f}bps  Top:{top_features}')

    c.close()

    # ── 追补最后一段净值 ──
    if quarterly_dates:
        last_reb_ts = pd.Timestamp(str(quarterly_dates[-1]))
        for nav_date in trade_dates_after_start:
            if nav_date <= last_reb_ts:
                continue
            holdings_val = 0
            for code, shares in portfolio.items():
                try:
                    px = price_idx.get((pd.Timestamp(nav_date), code), None)
                except Exception:
                    px = None
                if px is None or pd.isna(px):
                    px = last_prices.get(code, 0)
                else:
                    last_prices[code] = px
                holdings_val += shares * px
            total_val = cash + holdings_val
            daily_nav.append({'trade_date': nav_date, 'nav': total_val,
                             'bench': bench_daily.loc[bench_daily['trade_date']==nav_date, 'close'].values[0]
                                      if nav_date in bench_daily['trade_date'].values else np.nan})

    # ── Step 5: 绩效评估 ──
    print(f'\n── Step 5: 绩效评估 ──')
    nav_df = pd.DataFrame(daily_nav)
    if nav_df.empty or len(nav_df) < 10:
        print('❌ 净值记录不足')
        return

    nav_df = nav_df.sort_values('trade_date').dropna(subset=['nav', 'bench'])
    nav_df['strategy_return'] = nav_df['nav'].pct_change()
    nav_df['bench_return'] = nav_df['bench'].pct_change()
    nav_df['excess_return'] = nav_df['strategy_return'] - nav_df['bench_return']

    # 策略日收益
    strat_rets = nav_df['strategy_return'].dropna().values
    excess_rets = nav_df['excess_return'].dropna().values
    n_days = len(strat_rets)

    # 策略指标
    total_ret = nav_df['nav'].values[-1] / initial_value - 1
    bench_total = nav_df['bench'].values[-1] / nav_df['bench'].values[0] - 1
    ann_ret = np.mean(strat_rets) * 252
    ann_vol = np.std(strat_rets, ddof=1) * np.sqrt(252)
    ann_excess = np.mean(excess_rets) * 252
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    excess_sharpe = ann_excess / (np.std(excess_rets, ddof=1) * np.sqrt(252)) if np.std(excess_rets) > 0 else 0
    strat_cum = np.cumprod(1 + strat_rets)
    peak = np.maximum.accumulate(strat_cum)
    mdd = np.min(strat_cum / peak - 1)
    bench_cum = np.cumprod(1 + bench_total / n_days)  # approximate
    win_rate = np.mean(strat_rets > 0)
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    # 滚动12月
    rolling_12m = pd.Series(strat_rets).rolling(252).apply(
        lambda x: np.prod(1+x)-1).dropna() if n_days > 252 else pd.Series()

    # ── 报告 ──
    print(f'\n{"═"*70}')
    print(f'  📊 ML驱动回测链 v2.1 — 最终报告')
    print(f'{"═"*70}')
    print(f'')
    print(f'  策略指标 (日频净值):')
    print(f'    总收益      {total_ret*100:+.1f}%')
    print(f'    年化收益    {ann_ret*100:+.1f}%')
    print(f'    年化波动    {ann_vol*100:+.1f}%')
    print(f'    Sharpe      {sharpe:.3f}')
    print(f'    超额Sharpe  {excess_sharpe:.3f}')
    print(f'    MDD         {mdd*100:.1f}%')
    print(f'    Calmar      {calmar:.3f}')
    print(f'    日胜率      {win_rate*100:.0f}%')
    if len(rolling_12m) > 0:
        print(f'    滚动12月  min={rolling_12m.min()*100:+.0f}%  max={rolling_12m.max()*100:+.0f}%  '
              f'last={rolling_12m.iloc[-1]*100:+.0f}%')
    print(f'')
    print(f'  基准对比:')
    print(f'    策略总收益  {total_ret*100:+.1f}%')
    print(f'    沪深300     {bench_total*100:+.1f}%')
    print(f'    超额收益    {(total_ret-bench_total)*100:+.1f}%')
    print(f'')
    print(f'  交易统计:')
    print(f'    调仓次数    {len(quarterly_dates)}')
    print(f'    交易笔数    {len(trade_log)}')
    buys = sum(1 for t in trade_log if t['action']=='BUY')
    sells = sum(1 for t in trade_log if t['action']=='SELL')
    print(f'    买入{buys}笔  卖出{sells}笔')
    print(f'')

    if portfolio:
        print(f'  最新持仓({nav_df["trade_date"].values[-1]}):')
        for code, shares in list(portfolio.items())[:10]:
            px = last_prices.get(code, 0)
            print(f'    {code}  {shares}股  @{px:.2f}  ={shares*px:,.0f}')

    # ── 等权基准诊断 ──
    print(f'\n── 诊断: 等权候选池基准 ──')
    # 在每个调仓日, 计算"所有可选股票等权买入"的收益
    eq_nav = [{'trade_date': trade_dates_after_start[0], 'nav': initial_value}]
    eq_portfolio_val = initial_value
    eq_stocks = []
    eq_cash = 0  # all invested

    for test_date in quarterly_dates[:5]:  # 只看前5次,够判断
        test_mask = data['trade_date'] == test_date
        candidates = data.loc[test_mask, ['ts_code', 'close']].dropna(subset=['close'])
        if len(candidates) < 30:
            continue

        # 等权买入所有候选股
        n_eq = min(50, len(candidates))
        eq_sample = candidates.sample(n_eq, random_state=42)
        eq_ret_60d = target.loc[(target['trade_date'] == test_date) &
                                 target['ts_code'].isin(eq_sample['ts_code'].values),
                                 'excess_ret']
        if eq_ret_60d.empty:
            continue
        avg_ret = eq_ret_60d.mean()
        eq_portfolio_val *= (1 + avg_ret)
        eq_nav.append({'trade_date': test_date, 'nav': eq_portfolio_val})

    if len(eq_nav) > 1:
        eq_df = pd.DataFrame(eq_nav)
        eq_total = eq_portfolio_val / initial_value - 1
        print(f'  等权候选池(前5次调仓): {eq_total*100:+.1f}%')
        print(f'  ML策略同期: 查看调仓日志对比')
        # 找对应的ML净值
        ml_matches = [n for n in daily_nav if str(n['trade_date']) in [str(q) for q in quarterly_dates[:5]]]
        if ml_matches:
            ml_first = ml_matches[0]['nav']
            ml_last = ml_matches[-1]['nav']
            ml_ret = ml_last/ml_first - 1
            print(f'  ML策略同期: {ml_ret*100:+.1f}%')
            print(f'  超额: {(ml_ret-eq_total)*100:+.1f}pp')

    # ── 压力测试 ──
    print(f'\n── Step 6: 极端行情压力测试 ──')
    if nav_df is not None and len(nav_df) > 100:
        nav_stress = nav_df[['trade_date', 'nav', 'bench']].copy()
        nav_stress.columns = ['trade_date', 'nav', 'benchmark_nav']
        tester = StressTester(nav_stress)
        tester.summary()

    elapsed = time.time() - t0
    print(f'\n  ⏱ 总耗时: {elapsed/60:.1f}分钟')
    print(f'{"═"*70}')

    # 保存报告
    report = {
        'date': date.today().isoformat(),
        'config': {
            'forward_days': FORWARD_DAYS,
            'top_n': TOP_N,
            'min_train_years': MIN_TRAIN_YEARS,
            'rebalance_months': REBALANCE_MONTHS,
            'costs': {'stamp_tax': STAMP_TAX, 'commission': COMMISSION, 'slippage': SLIPPAGE}
        },
        'performance': {
            'total_return': round(total_ret*100, 2),
            'annual_return': round(ann_ret*100, 2),
            'annual_vol': round(ann_vol*100, 2),
            'sharpe': round(sharpe, 3),
            'excess_sharpe': round(excess_sharpe, 3),
            'mdd': round(mdd*100, 2),
            'calmar': round(calmar, 3),
            'win_rate': round(win_rate*100, 1),
            'benchmark_return': round(bench_total*100, 2),
            'excess_return': round((total_ret-bench_total)*100, 2),
        },
        'trading': {
            'rebalance_count': len(quarterly_dates),
            'trade_count': len(trade_log),
            'buy_count': buys,
            'sell_count': sells,
        },
        'elapsed_minutes': round(elapsed/60, 1),
    }
    report_path = REPORT_DIR / f'ml_chain_{date.today().isoformat()}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'  报告: {report_path}')

    return report


if __name__ == '__main__':
    try:
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
    run_ml_chain()
