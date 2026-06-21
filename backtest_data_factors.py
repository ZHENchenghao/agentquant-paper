# -*- coding: utf-8 -*-
"""
AgentQuant · 数据因子独立回测
==============================
不改V3策略, 单独验证每个数据维度的增量贡献:
  1. 北向资金: 择时信号? 趋势确认? 还是无用?
  2. 经营现金流质量: OCF/NP 排雷效果
  3. 商誉占比: 高商誉是否跑输
  4. 应收账款: 应收暴增预警
  5. 概念炒作: 涨太快没业绩 → 后续表现
"""
import duckdb, pandas as pd, numpy as np
from datetime import date, timedelta

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB, read_only=True)


# ═══════════════════════════════════════
# 1. 北向资金 — 三种用法对比
# ═══════════════════════════════════════

def backtest_northbound():
    """
    北向三种用法:
    A: 择时 — 北向流入>50亿 → 次日买沪深300, 持1天
    B: 趋势确认 — 北向+沪深300同向 → 持5天
    C: 极端信号 — 北向>100亿或<-100亿 → 反向或跟随
    """
    c = cdb()

    # 基础数据: 北向日净流入 + 沪深300日收益
    df = c.execute("""
        WITH nb AS (
            SELECT trade_date, SUM(net_flow) daily_flow
            FROM north_bound_flow WHERE net_flow != 0 AND trade_date >= '2015-01-01'
            GROUP BY trade_date
        ),
        idx AS (
            SELECT trade_date, close,
                   (LEAD(close,1) OVER w - close)/close*100 ret_1d,
                   (LEAD(close,5) OVER w - close)/close*100 ret_5d,
                   (LEAD(close,20) OVER w - close)/close*100 ret_20d
            FROM kline_daily WHERE ts_code='sh000300' AND trade_date >= '2015-01-01'
            WINDOW w AS (ORDER BY trade_date)
        )
        SELECT nb.trade_date, nb.daily_flow,
               idx.ret_1d, idx.ret_5d, idx.ret_20d
        FROM nb JOIN idx ON nb.trade_date = idx.trade_date
        WHERE idx.ret_1d IS NOT NULL
        ORDER BY nb.trade_date
    """).df()
    c.close()

    if df.empty:
        return {'error': 'no data'}

    results = {}

    # A: 择时 — 流入>50亿次日买
    buy_days = df[df['daily_flow'] > 50]
    sell_days = df[df['daily_flow'] < -50]
    neutral = df[(df['daily_flow'] >= -50) & (df['daily_flow'] <= 50)]

    results['A_择时_流入>50亿次日'] = {
        '信号天数': len(buy_days),
        '次日均收益%': round(buy_days['ret_1d'].mean(), 3),
        '胜率%': round((buy_days['ret_1d'] > 0).mean() * 100, 1),
        '基准均收益%': round(df['ret_1d'].mean(), 3),
        '超额': round(buy_days['ret_1d'].mean() - df['ret_1d'].mean(), 3),
    }
    results['A_择时_流出>50亿次日'] = {
        '信号天数': len(sell_days),
        '次日均收益%': round(sell_days['ret_1d'].mean(), 3),
        '胜率%': round((sell_days['ret_1d'] > 0).mean() * 100, 1),
    }

    # B: 趋势确认 — 北向+当日同向 → 持5天
    df['nb_dir'] = np.where(df['daily_flow'] > 0, 1, -1)
    df['idx_dir'] = np.where(df['ret_1d'] > 0, 1, -1)  # 当日方向
    # 同向信号: 北向和指数同日同向
    same_dir = df[df['nb_dir'] == df['idx_dir']]
    opp_dir = df[df['nb_dir'] != df['idx_dir']]

    results['B_确认_同向5日'] = {
        '信号天数': len(same_dir),
        '5日均收益%': round(same_dir['ret_5d'].mean(), 3),
        '胜率%': round((same_dir['ret_5d'] > 0).mean() * 100, 1),
    }
    results['B_背离_反向5日'] = {
        '信号天数': len(opp_dir),
        '5日均收益%': round(opp_dir['ret_5d'].mean(), 3),
        '胜率%': round((opp_dir['ret_5d'] > 0).mean() * 100, 1),
    }

    # C: 极端 — >100亿 或 <-100亿
    extreme_in = df[df['daily_flow'] > 100]
    extreme_out = df[df['daily_flow'] < -100]
    results['C_极端流入>100亿'] = {
        '信号天数': len(extreme_in),
        '5日均收益%': round(extreme_in['ret_5d'].mean(), 3) if len(extreme_in) > 0 else None,
        '胜率%': round((extreme_in['ret_5d'] > 0).mean() * 100, 1) if len(extreme_in) > 0 else None,
    }
    results['C_极端流出>100亿'] = {
        '信号天数': len(extreme_out),
        '5日均收益%': round(extreme_out['ret_5d'].mean(), 3) if len(extreme_out) > 0 else None,
        '胜率%': round((extreme_out['ret_5d'] > 0).mean() * 100, 1) if len(extreme_out) > 0 else None,
    }

    # D: 累计北向趋势 — 北向20日累计流入 vs 沪深300 20日收益
    df['nb_20d'] = df['daily_flow'].rolling(20).sum()
    df['nb_20d_dir'] = np.where(df['nb_20d'] > 0, 1, -1)
    pos_20d = df[df['nb_20d_dir'] == 1]
    neg_20d = df[df['nb_20d_dir'] == -1]

    results['D_趋势_20日累计流入'] = {
        '信号天数': len(pos_20d),
        '20日均收益%': round(pos_20d['ret_20d'].mean(), 3),
        '胜率%': round((pos_20d['ret_20d'] > 0).mean() * 100, 1),
    }
    results['D_趋势_20日累计流出'] = {
        '信号天数': len(neg_20d),
        '20日均收益%': round(neg_20d['ret_20d'].mean(), 3),
        '胜率%': round((neg_20d['ret_20d'] > 0).mean() * 100, 1),
    }

    # 基准
    results['基准'] = {
        '总天数': len(df),
        '日均收益%': round(df['ret_1d'].mean(), 3),
        '日胜率%': round((df['ret_1d'] > 0).mean() * 100, 1),
        '5日均收益%': round(df['ret_5d'].mean(), 3),
        '20日均收益%': round(df['ret_20d'].mean(), 3),
    }

    return results


# ═══════════════════════════════════════
# 2. 排雷因子 — 财报质量验证
# ═══════════════════════════════════════

def backtest_fundamentals():
    """
    验证: 经营现金流质量 / 商誉占比 / 应收堆积 是否预测后续收益
    """
    c = cdb()

    # 取年报数据 + 后续收益 (用最近交易日匹配)
    df = c.execute("""
        WITH fin AS (
            SELECT ts_code, report_date,
                   net_profit, operating_cf, revenue, accounts_receivable,
                   CASE WHEN net_profit>0 AND operating_cf IS NOT NULL AND ABS(operating_cf) > 0
                        THEN operating_cf/NULLIF(net_profit,0) END ocf_ratio,
                   CASE WHEN revenue>0 AND accounts_receivable IS NOT NULL AND ABS(accounts_receivable) > 0
                        THEN accounts_receivable/NULLIF(revenue,0) END ar_ratio
            FROM financial_statements
            WHERE report_type='annual' AND net_profit IS NOT NULL
              AND report_date >= '2018-12-31'
        ),
        ret AS (
            SELECT ts_code, trade_date, close,
                   (LEAD(close,60) OVER w - close)/close*100 ret_60d,
                   (LEAD(close,120) OVER w - close)/close*100 ret_120d,
                   (LEAD(close,250) OVER w - close)/close*100 ret_250d
            FROM kline_daily WHERE trade_date >= '2019-01-01'
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        )
        SELECT fin.ts_code, fin.report_date,
               fin.ocf_ratio, fin.ar_ratio,
               ret.trade_date kdate, ret.ret_60d, ret.ret_120d, ret.ret_250d
        FROM fin JOIN ret ON fin.ts_code=ret.ts_code
            AND ret.trade_date >= fin.report_date
            AND ret.trade_date <= (fin.report_date + INTERVAL 150 DAY)::DATE
        WHERE fin.ocf_ratio IS NOT NULL OR fin.ar_ratio IS NOT NULL
        QUALIFY ROW_NUMBER() OVER(PARTITION BY fin.ts_code, fin.report_date ORDER BY ret.trade_date) = 1
    """).df()
    c.close()

    if df.empty:
        return {'error': 'no fundamental data'}

    results = {'总样本': len(df)}

    # OCF质量分组
    if 'ocf_ratio' in df.columns and df['ocf_ratio'].notna().sum() > 0:
        ocf_ok = df[df['ocf_ratio'] > 0.5]      # 现金流健康
        ocf_bad = df[df['ocf_ratio'] < 0.3]     # 纸面利润
        ocf_neg = df[df['ocf_ratio'] < 0]        # 现金流为负

        results['OCF健康(>0.5)'] = {
            '样本': len(ocf_ok),
            '60日收益%': round(ocf_ok['ret_60d'].mean(), 2) if len(ocf_ok)>0 else None,
            '120日收益%': round(ocf_ok['ret_120d'].mean(), 2) if len(ocf_ok)>0 else None,
            '250日收益%': round(ocf_ok['ret_250d'].mean(), 2) if len(ocf_ok)>0 else None,
        }
        results['OCF纸面(<0.3)'] = {
            '样本': len(ocf_bad),
            '60日收益%': round(ocf_bad['ret_60d'].mean(), 2) if len(ocf_bad)>0 else None,
            '120日收益%': round(ocf_bad['ret_120d'].mean(), 2) if len(ocf_bad)>0 else None,
            '250日收益%': round(ocf_bad['ret_250d'].mean(), 2) if len(ocf_bad)>0 else None,
        }
        results['OCF负值'] = {
            '样本': len(ocf_neg),
            '60日收益%': round(ocf_neg['ret_60d'].mean(), 2) if len(ocf_neg)>0 else None,
            '120日收益%': round(ocf_neg['ret_120d'].mean(), 2) if len(ocf_neg)>0 else None,
        }

    # 应收分组
    if 'ar_ratio' in df.columns and df['ar_ratio'].notna().sum() > 0:
        ar_ok = df[df['ar_ratio'] < 0.3]
        ar_high = df[(df['ar_ratio'] >= 0.3) & (df['ar_ratio'] < 0.6)]
        ar_extreme = df[df['ar_ratio'] >= 0.6]

        results['AR正常(<30%)'] = {
            '样本': len(ar_ok),
            '60日收益%': round(ar_ok['ret_60d'].mean(), 2) if len(ar_ok)>0 else None,
            '120日收益%': round(ar_ok['ret_120d'].mean(), 2) if len(ar_ok)>0 else None,
        }
        results['AR偏高(30-60%)'] = {
            '样本': len(ar_high),
            '60日收益%': round(ar_high['ret_60d'].mean(), 2) if len(ar_high)>0 else None,
            '120日收益%': round(ar_high['ret_120d'].mean(), 2) if len(ar_high)>0 else None,
        }
        results['AR极端(>60%)'] = {
            '样本': len(ar_extreme),
            '60日收益%': round(ar_extreme['ret_60d'].mean(), 2) if len(ar_extreme)>0 else None,
            '120日收益%': round(ar_extreme['ret_120d'].mean(), 2) if len(ar_extreme)>0 else None,
        }

    return results


# ═══════════════════════════════════════
# 3. 商誉 — 减值风险验证
# ═══════════════════════════════════════

def backtest_goodwill():
    """商誉占比 vs 后续收益 + 最大回撤"""
    c = cdb()
    df = c.execute("""
        WITH gw AS (
            SELECT ts_code, report_date, goodwill_pct FROM goodwill_detail WHERE goodwill_pct IS NOT NULL
        ),
        start_price AS (
            SELECT gw.ts_code, gw.goodwill_pct, k.close start_close, k.trade_date start_date
            FROM gw JOIN kline_daily k ON gw.ts_code=k.ts_code
            WHERE k.trade_date >= gw.report_date AND k.trade_date <= (gw.report_date + INTERVAL 30 DAY)::DATE
            QUALIFY ROW_NUMBER() OVER(PARTITION BY gw.ts_code ORDER BY k.trade_date) = 1
        ),
        end_price AS (
            SELECT sp.ts_code, sp.goodwill_pct, sp.start_close, sp.start_date,
                   k2.close end_close, k2.trade_date end_date
            FROM start_price sp
            JOIN kline_daily k2 ON sp.ts_code=k2.ts_code
            WHERE k2.trade_date >= (sp.start_date + INTERVAL 120 DAY)::DATE
            QUALIFY ROW_NUMBER() OVER(PARTITION BY sp.ts_code ORDER BY k2.trade_date) = 1
        )
        SELECT ts_code, goodwill_pct,
               (end_close/start_close - 1)*100 ret_120d
        FROM end_price WHERE end_close IS NOT NULL
    """).df()
    c.close()

    if df.empty:
        return {'error': 'no goodwill data, need more backfill'}

    results = {'总样本': len(df)}

    gw_low = df[df['goodwill_pct'] < 10]
    gw_mid = df[(df['goodwill_pct'] >= 10) & (df['goodwill_pct'] < 30)]
    gw_high = df[df['goodwill_pct'] >= 30]

    for label, group in [('商誉<10%', gw_low), ('商誉10-30%', gw_mid), ('商誉>30%', gw_high)]:
        if len(group) > 0:
            results[label] = {
                '样本': len(group),
                '120日收益%': round(group['ret_120d'].mean(), 2),
                '中位收益%': round(group['ret_120d'].median(), 2),
                '最差%': round(group['ret_120d'].min(), 2),
                '胜率%': round((group['ret_120d'] > 0).mean() * 100, 1),
            }

    return results


# ═══════════════════════════════════════
# 主入口
# ═══════════════════════════════════════

if __name__ == '__main__':
    import json

    print('=' * 60)
    print('  数据因子独立回测 — 不改V3策略')
    print('=' * 60)

    # 1. 北向
    print('\n--- 1. 北向资金 ---')
    nb = backtest_northbound()
    if 'error' not in nb:
        print(f'  基准(2234天): 日均{nb["基准"]["日均收益%"]:.3f}%  胜率{nb["基准"]["日胜率%"]:.0f}%')
        for k, v in nb.items():
            if k != '基准' and isinstance(v, dict) and '信号天数' in v:
                extra = ''
                if '超额' in v:
                    extra = f'  超额{v["超额"]:.3f}%'
                print(f'  {k}: {v["信号天数"]}次  收益{v.get("次日均收益%",v.get("5日均收益%",v.get("20日均收益%","?")))}%  胜率{v.get("胜率%","?")}%{extra}')
    else:
        print(f'  ERR: {nb["error"]}')

    # 2. 排雷因子
    print('\n--- 2. 排雷因子 ---')
    fin = backtest_fundamentals()
    if 'error' not in fin:
        for k, v in fin.items():
            if isinstance(v, dict):
                print(f'  {k}: {v.get("样本","?")}只  60日{v.get("60日收益%","?")}%  120日{v.get("120日收益%","?")}%')
    else:
        print(f'  ERR: {fin["error"]}')

    # 3. 商誉
    print('\n--- 3. 商誉 ---')
    gw = backtest_goodwill()
    if 'error' not in gw:
        for k, v in gw.items():
            if isinstance(v, dict):
                print(f'  {k}: {v.get("样本","?")}只  120日{v.get("120日收益%","?")}%  最差{v.get("最差%","?")}%  胜率{v.get("胜率%","?")}%')
    else:
        print(f'  ERR: {gw["error"]}')

    print('\n' + '=' * 60)
