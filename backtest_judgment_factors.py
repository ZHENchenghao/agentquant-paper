# -*- coding: utf-8 -*-
"""
AgentQuant · 判断因子回测
=========================
不是数学公式, 是模式识别。逐个验证: 国家队/穿透/北向极端/概念炒作
"""
import duckdb, pandas as pd, numpy as np
from datetime import date, timedelta

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB, read_only=True)


# ═══════════════════════════════════════
# 因子1: 国家队护盘 — 该跟还是该跑?
# ═══════════════════════════════════════

def backtest_national_team():
    """
    国家队护盘日: 上证50涨>1% AND 科创50涨<0.5% AND 全市场缩量
    问题: 这种日子之后, 沪深300怎么走?
    跟: 国家队扛着, 跟进去 → 5日/20日收益?
    跑: 国家队在掩护出货, 应该跑
    """
    c = cdb()
    df = c.execute("""
        WITH idx AS (
            SELECT k1.trade_date,
                   (k1.close/LAG(k1.close) OVER w1 - 1)*100 chg50,
                   (k2.close/LAG(k2.close) OVER w2 - 1)*100 chg688,
                   (k3.close/LAG(k3.close) OVER w3 - 1)*100 chg300,
                   k3.close idx_close,
                   (LEAD(k3.close,1) OVER w3/k3.close-1)*100 ret1d,
                   (LEAD(k3.close,5) OVER w3/k3.close-1)*100 ret5d,
                   (LEAD(k3.close,20) OVER w3/k3.close-1)*100 ret20d,
                   (LEAD(k3.close,60) OVER w3/k3.close-1)*100 ret60d
            FROM kline_daily k1
            JOIN kline_daily k2 ON k1.trade_date=k2.trade_date
            JOIN kline_daily k3 ON k1.trade_date=k3.trade_date
            WHERE k1.ts_code='sh000016' AND k2.ts_code='sh000688' AND k3.ts_code='sh000300'
            WINDOW w1 AS (ORDER BY k1.trade_date),
                   w2 AS (ORDER BY k2.trade_date),
                   w3 AS (ORDER BY k3.trade_date)
        ),
        vol AS (
            SELECT trade_date, SUM(amount)/1e8 total_vol,
                   AVG(SUM(amount)/1e8) OVER(ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) avg20
            FROM kline_daily GROUP BY trade_date
        ),
        nb AS (
            SELECT trade_date, SUM(net_flow) daily FROM north_bound_flow
            WHERE net_flow != 0 GROUP BY trade_date
        )
        SELECT idx.trade_date, idx.chg50, idx.chg688, idx.chg300,
               vol.total_vol/vol.avg20 vol_ratio,
               COALESCE(nb.daily,0) nb_flow,
               -- 国家队信号
               CASE WHEN idx.chg50 > 1.0 AND idx.chg688 < 0.5 AND vol.total_vol/vol.avg20 < 0.8
                    THEN '国家队护盘'
                    WHEN idx.chg50 < -1.0 AND idx.chg688 > -0.5
                    THEN '国家队撤退'
                    ELSE '正常' END signal,
               idx.ret1d, idx.ret5d, idx.ret20d, idx.ret60d
        FROM idx JOIN vol ON idx.trade_date=vol.trade_date
        LEFT JOIN nb ON idx.trade_date=nb.trade_date
        WHERE idx.trade_date >= '2015-01-01'
    """).df()
    c.close()

    results = {}
    # 护盘日
    nt_days = df[df['signal'] == '国家队护盘']
    exit_days = df[df['signal'] == '国家队撤退']
    normal = df[df['signal'] == '正常']

    results['国家队护盘(上证50涨>1%+科创涨<0.5%+缩量)'] = {
        '天数': len(nt_days),
        '1日收益': round(nt_days['ret1d'].mean(), 2) if len(nt_days)>0 else None,
        '5日收益': round(nt_days['ret5d'].mean(), 2) if len(nt_days)>0 else None,
        '20日收益': round(nt_days['ret20d'].mean(), 2) if len(nt_days)>0 else None,
        '胜率5日': round((nt_days['ret5d']>0).mean()*100, 1) if len(nt_days)>0 else None,
    }
    results['国家队撤退(上证50跌>1%+科创跌<0.5%)'] = {
        '天数': len(exit_days),
        '5日收益': round(exit_days['ret5d'].mean(), 2) if len(exit_days)>0 else None,
        '20日收益': round(exit_days['ret20d'].mean(), 2) if len(exit_days)>0 else None,
    }
    results['基准(正常日)'] = {
        '天数': len(normal),
        '5日收益': round(normal['ret5d'].mean(), 2),
        '胜率5日': round((normal['ret5d']>0).mean()*100, 1),
    }
    return results


# ═══════════════════════════════════════
# 因子2: 北向极端 vs 温和流出
# ═══════════════════════════════════════

def backtest_northbound_extreme():
    """北向流出: 温和(-50亿) vs 极端(>100亿) — 该跑还是该买?"""
    c = cdb()
    df = c.execute("""
        WITH nb AS (
            SELECT trade_date, SUM(net_flow) daily FROM north_bound_flow
            WHERE net_flow != 0 GROUP BY trade_date
        ),
        idx AS (
            SELECT trade_date,
                   (LEAD(close,5) OVER w/close-1)*100 ret5d,
                   (LEAD(close,20) OVER w/close-1)*100 ret20d,
                   (LEAD(close,60) OVER w/close-1)*100 ret60d
            FROM kline_daily WHERE ts_code='sh000300'
            WINDOW w AS (ORDER BY trade_date)
        )
        SELECT nb.trade_date, nb.daily,
               CASE WHEN nb.daily > 100 THEN '极端流入(>100亿)'
                    WHEN nb.daily > 50 THEN '大幅流入(50-100亿)'
                    WHEN nb.daily > 0 THEN '小幅流入'
                    WHEN nb.daily > -50 THEN '温和流出(0-50亿)'
                    WHEN nb.daily > -100 THEN '大幅流出(50-100亿)'
                    ELSE '极端流出(>100亿)' END signal,
               idx.ret5d, idx.ret20d, idx.ret60d
        FROM nb JOIN idx ON nb.trade_date=idx.trade_date
        WHERE nb.trade_date >= '2015-01-01' AND idx.ret5d IS NOT NULL
    """).df()
    c.close()

    results = {}
    for label in ['极端流入(>100亿)','大幅流入(50-100亿)','小幅流入',
                  '温和流出(0-50亿)','大幅流出(50-100亿)','极端流出(>100亿)']:
        sub = df[df['signal'] == label]
        if len(sub) > 3:
            results[label] = {
                '天数': len(sub),
                '5日收益': round(sub['ret5d'].mean(), 2),
                '20日收益': round(sub['ret20d'].mean(), 2),
                '60日收益': round(sub['ret60d'].mean(), 2) if sub['ret60d'].notna().any() else None,
                '胜率5日': round((sub['ret5d']>0).mean()*100, 1),
            }
    results['全样本均值'] = {
        '天数': len(df),
        '5日收益': round(df['ret5d'].mean(), 2),
        '胜率5日': round((df['ret5d']>0).mean()*100, 1),
    }
    return results


# ═══════════════════════════════════════
# 因子3: 概念炒作甄别
# ═══════════════════════════════════════

def backtest_concept_hype():
    """
    概念炒作: 30日涨>30% AND ROE<5%
    这种股票后续60日表现?
    """
    c = cdb()
    df = c.execute("""
        WITH prices AS (
            SELECT ts_code, trade_date, close,
                   (close/LAG(close,30) OVER w - 1)*100 chg30,
                   (LEAD(close,60) OVER w/close-1)*100 ret60d
            FROM kline_daily WHERE trade_date >= '2018-01-01'
            WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
        ),
        fin AS (
            SELECT ts_code, roe, report_date FROM financial_statements
            WHERE report_type='annual' AND roe IS NOT NULL
        )
        SELECT p.trade_date, p.chg30, f.roe, p.ret60d,
               CASE WHEN p.chg30>30 AND (f.roe IS NULL OR f.roe<5) THEN '炒作嫌疑'
                    WHEN p.chg30>30 AND f.roe>=15 THEN '业绩支撑的涨'
                    WHEN p.chg30<-20 THEN '超跌'
                    ELSE '正常' END signal
        FROM prices p
        LEFT JOIN fin f ON p.ts_code=f.ts_code
            AND f.report_date <= p.trade_date
            AND f.report_date >= p.trade_date - INTERVAL 1 YEAR
        WHERE p.chg30 IS NOT NULL AND p.ret60d IS NOT NULL
    """).df()
    c.close()

    results = {}
    for label in ['炒作嫌疑', '业绩支撑的涨', '超跌', '正常']:
        sub = df[df['signal'] == label]
        if len(sub) > 10:
            results[label] = {
                '样本数': len(sub),
                '60日均收益': round(sub['ret60d'].mean(), 2),
                '中位收益': round(sub['ret60d'].median(), 2),
                '胜率': round((sub['ret60d']>0).mean()*100, 1),
                '最差': round(sub['ret60d'].min(), 2),
            }
    return results


# ═══════════════════════════════════════
# 因子4: 穿透判断 — 事件前定价
# ═══════════════════════════════════════

def backtest_pre_pricing():
    """
    WTI暴跌>5%: 当天沪深300涨=利好已定价 → 后续表现?
    WTI暴跌>5%: 当天沪深300跌=恐慌超卖 → 后续表现?
    """
    c = cdb()
    df = c.execute("""
        WITH drops AS (
            SELECT trade_date, (wti/LAG(wti) OVER w - 1)*100 chg
            FROM macro_indicators WHERE wti IS NOT NULL
            WINDOW w AS (ORDER BY trade_date) QUALIFY chg < -5
        ),
        idx AS (
            SELECT trade_date,
                   (close/LAG(close) OVER w - 1)*100 chg,
                   (LEAD(close,5) OVER w/close-1)*100 ret5d,
                   (LEAD(close,20) OVER w/close-1)*100 ret20d
            FROM kline_daily WHERE ts_code='sh000300'
            WINDOW w AS (ORDER BY trade_date)
        )
        SELECT d.trade_date,
               CASE WHEN i.chg>0 THEN '利好已定价(当日涨)' ELSE '恐慌超卖(当日跌)' END scenario,
               i.ret5d, i.ret20d
        FROM drops d JOIN idx i ON d.trade_date=i.trade_date
        WHERE i.ret5d IS NOT NULL
    """).df()
    c.close()

    results = {}
    for label in ['利好已定价(当日涨)', '恐慌超卖(当日跌)']:
        sub = df[df['scenario'] == label]
        if len(sub) > 3:
            results[label] = {
                '次数': len(sub),
                '5日收益': round(sub['ret5d'].mean(), 2),
                '20日收益': round(sub['ret20d'].mean(), 2),
                '胜率5日': round((sub['ret5d']>0).mean()*100, 1),
            }
    return results


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print('=' * 60)
    print('  判断因子回测: 非数学, 模式识别')
    print('=' * 60)

    print('\n--- 1. 国家队: 护盘时该跟还是该跑? ---')
    for k, v in backtest_national_team().items():
        print(f'  {k}: {v["天数"]}天')
        for mk, mv in v.items():
            if mk != '天数':
                print(f'    {mk}: {mv}')

    print('\n--- 2. 北向: 极端流出是恐慌还是机会? ---')
    for k, v in backtest_northbound_extreme().items():
        print(f'  {k}: {v["天数"]}天  5日{v.get("5日收益","?")}%  胜率{v.get("胜率5日","?")}%')

    print('\n--- 3. 概念炒作: 涨30%但没业绩的后60天? ---')
    for k, v in backtest_concept_hype().items():
        print(f'  {k}: {v["样本数"]}只  60日{v["60日均收益"]}%  胜率{v["胜率"]}%  最差{v["最差"]}%')

    print('\n--- 4. 穿透: 利好已定价 vs 恐慌超卖 ---')
    for k, v in backtest_pre_pricing().items():
        print(f'  {k}: {v["次数"]}次  5日{v["5日收益"]}%  20日{v["20日收益"]}%  胜率{v["胜率5日"]}%')
