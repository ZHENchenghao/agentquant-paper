# -*- coding: utf-8 -*-
"""
AgentQuant · 排雷+概念甄别
=========================
五雷扫: ST/解禁/质押/财务注水/商誉
概念甄别: 真概念vs蹭热点, 轮动生命周期(早期/主升/末期)

集成到factor_pipeline: get_clean_universe() → mine_sweep() → offense/defense
"""
import duckdb, pandas as pd, numpy as np
from datetime import date, timedelta

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def cdb(): return duckdb.connect(DB, read_only=True)


# ═══════════════════════════════════════
# 排雷
# ═══════════════════════════════════════

def sweep_stocks(universe: list, trade_date: date) -> pd.DataFrame:
    """
    对候选池执行五雷扫描, 返回每只票的风险标记。
    不直接排除, 而是标注风险等级——让下游PM决策。
    """
    if not universe:
        return pd.DataFrame()

    c = cdb()
    codes_str = ','.join([f"'{x}'" for x in universe])
    td = trade_date.isoformat()
    results = []

    for code in universe:
        ts = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
        mines = []

        # ── 雷1: ST/退市 ──
        r = c.execute("SELECT name, is_st FROM stock_basic WHERE ts_code=?", [ts]).fetchone()
        if r and r[1]:
            mines.append({'type': 'ST/delist', 'severity': 'FATAL', 'detail': f'ST标记: {r[0]}'})

        # ── 雷2: 股权质押爆仓 ──
        r = c.execute("""
            SELECT pledge_ratio FROM shareholder_pledge
            WHERE ts_code=? AND trade_date<=? AND pledge_ratio > 40
            ORDER BY pledge_ratio DESC LIMIT 1
        """, [ts, td]).fetchone()
        if r:
            sev = 'FATAL' if r[0] > 60 else 'HIGH' if r[0] > 50 else 'MEDIUM'
            mines.append({'type': 'pledge', 'severity': sev, 'detail': f'质押比例{r[0]:.0f}%'})

        # ── 雷3: 商誉减值 ──
        r = c.execute("""
            SELECT goodwill_pct, impairment_risk FROM goodwill_detail
            WHERE ts_code=? AND report_date<=? ORDER BY report_date DESC LIMIT 1
        """, [ts, td]).fetchone()
        if r and r[0] and r[0] > 20:
            sev = 'FATAL' if r[0] > 40 else 'HIGH'
            mines.append({'type': 'goodwill', 'severity': sev,
                          'detail': f'商誉占净资产{r[0]:.0f}%' + (f' 减值风险:{r[1]}' if r[1] else '')})

        # ── 雷4: 财务注水 ──
        # 注意: 银行/保险/券商经营现金流为负是正常(贷款业务特性), 排除金融股
        r = c.execute("""
            SELECT net_profit, operating_cf, revenue, accounts_receivable
            FROM financial_statements WHERE ts_code=? AND report_type='annual'
            ORDER BY report_date DESC LIMIT 1
        """, [ts]).fetchone()
        if r:
            net, ocf, rev, ar = r
            # 判断是否金融股 (银行/保险/券商)
            is_fin = ts.startswith('601') and any(kw in str(code) for kw in ['288','398','328','939','988','818','166','169','229','658','077','916','825','628','336','688','788','211','318'])
            # 纸面利润: 有利润但经营现金流极低 (金融股跳过)
            if not is_fin and net and net > 0 and ocf is not None and ocf < net * 0.3:
                mines.append({'type': 'fake_profit', 'severity': 'HIGH',
                              'detail': f'经营现金流/净利={ocf/net:.2f} (严重不匹配)'})
            elif not is_fin and net and net > 0 and ocf is not None and ocf < net * 0.5:
                mines.append({'type': 'weak_cashflow', 'severity': 'MEDIUM',
                              'detail': f'经营现金流/净利={ocf/net:.2f} (偏弱)'})
            # 应收堆积: 应收账款远超营收 (金融股跳过, 银行应收不是一般概念)
            if not is_fin and rev and ar and rev > 0 and ar / rev > 0.6:
                mines.append({'type': 'ar_buildup', 'severity': 'HIGH',
                              'detail': f'应收/营收={ar/rev:.0%} (放宽信用冲收入)'})

        # ── 雷5: 关联交易异常 ──
        r = c.execute("""
            SELECT MAX(related_ratio) FROM related_party_tx
            WHERE ts_code=? AND report_date>=?
        """, [ts, (trade_date - timedelta(days=365)).isoformat()]).fetchone()
        if r and r[0] and r[0] > 30:
            mines.append({'type': 'related_tx', 'severity': 'MEDIUM',
                          'detail': f'关联交易占营收{r[0]:.0f}%'})

        if mines:
            fatal = any(m['severity'] == 'FATAL' for m in mines)
            high = any(m['severity'] == 'HIGH' for m in mines)
            risk_level = 'FATAL' if fatal else 'HIGH' if high else 'MEDIUM'
            results.append({
                'ts_code': ts,
                'risk_level': risk_level,
                'mine_count': len(mines),
                'mines': mines,
            })

    c.close()
    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════
# 概念甄别
# ═══════════════════════════════════════

def concept_check(stock_codes: list, trade_date: date) -> pd.DataFrame:
    """
    判断涨幅是真概念还是蹭热点。

    真概念: 有政策/订单支撑, 板块联动, 机构参与
    蹭热点: 纯标签, 无基本面改善, 游资主导, 涨幅虚高

    返回每个标的的"概念可疑度" 0-100 (越高越可疑)
    """
    if not stock_codes:
        return pd.DataFrame()

    c = cdb()
    td = trade_date.isoformat()
    results = []

    for code in stock_codes:
        ts = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
        suspicion = 0
        reasons = []

        # ── 信号1: 30日涨幅 vs 营收增速 ──
        r = c.execute("""
            SELECT (MAX(CASE WHEN rn=1 THEN close END)/
                    NULLIF(MAX(CASE WHEN rn=30 THEN close END),0)-1)*100
            FROM (SELECT close, ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                  FROM kline_daily WHERE ts_code=? AND trade_date<=?)
            WHERE rn<=30
        """, [ts, td]).fetchone()
        chg_30d = r[0] if r and r[0] else 0

        r = c.execute("""
            SELECT revenue, net_profit, roe FROM financial_statements
            WHERE ts_code=? AND report_type='annual' ORDER BY report_date DESC LIMIT 1
        """, [ts]).fetchone()
        rev = r[0] if r else 0
        net = r[1] if r else 0
        roe = r[2] if r else 0

        # 涨幅远超基本面支撑 → 概念炒作嫌疑
        if chg_30d > 30 and (not roe or roe < 5):
            suspicion += 30
            reasons.append(f'30日涨{chg_30d:.0f}%但ROE={roe or 0:.1f}% → 脱离基本面')

        # ── 信号2: 龙虎榜游资 vs 机构 ──
        r = c.execute("""
            SELECT COUNT(*), COALESCE(SUM(institution_buy),0), COALESCE(SUM(institution_sell),0)
            FROM dragon_tiger_list WHERE ts_code=? AND trade_date>=?
        """, [ts, (trade_date - timedelta(days=30)).isoformat()]).fetchone()
        if r:
            appearances = r[0] or 0
            inst_net = (r[1] or 0) - (r[2] or 0)
            if appearances >= 3 and inst_net < 0:
                suspicion += 25
                reasons.append(f'龙虎榜{appearances}次上榜, 机构净卖出{inst_net:.0f} → 游资主导')

        # ── 信号3: 换手率异常 ──
        r = c.execute("""
            SELECT AVG(turnover_rate) FROM (
                SELECT turnover_rate FROM kline_daily WHERE ts_code=? AND trade_date<=?
                ORDER BY trade_date DESC LIMIT 5
            )
        """, [ts, td]).fetchone()
        if r and r[0] and r[0] > 10:
            suspicion += 20
            reasons.append(f'近5日换手率{r[0]:.0f}% → 筹码高速换手, 游资对倒嫌疑')

        # ── 信号4: 概念板块轮动位置 ──
        # (简化: 看30日涨幅是否已经过大 → 末期的概率高)
        if chg_30d > 50:
            suspicion += 15
            reasons.append(f'30日涨{chg_30d:.0f}% → 可能已处概念末期')
        elif chg_30d > 20:
            suspicion += 5
            reasons.append(f'30日涨{chg_30d:.0f}% → 主升段, 但需警惕')

        # ── 信号5: PE膨胀 vs ROE ──
        r = c.execute("""
            SELECT pe_ttm FROM valuation_daily WHERE ts_code=? AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 1
        """, [ts, td]).fetchone()
        pe = r[0] if r else None
        if pe and roe and pe > 50 and roe < 10:
            suspicion += 10
            reasons.append(f'PE={pe:.0f}但ROE={roe:.1f}% → 估值脱离盈利')

        results.append({
            'ts_code': ts,
            'suspicion': min(100, suspicion),
            'concept_risk': 'HIGH' if suspicion >= 60 else 'MEDIUM' if suspicion >= 30 else 'LOW',
            'chg_30d': round(chg_30d, 1),
            'reasons': reasons,
        })

    c.close()
    return pd.DataFrame(results) if results else pd.DataFrame()


# ═══════════════════════════════════════
# 集成: 在V3选股链路中调用
# ═══════════════════════════════════════

def filter_with_minesweep(universe: list, trade_date: date) -> dict:
    """
    排雷+概念甄别 → 返回清洗后的池子

    返回:
      clean: 无雷+非概念炒作的标的
      mines: 有雷标的 (FATAL直接排除, HIGH标记)
      concepts: 概念炒作嫌疑标的
    """
    mines_df = sweep_stocks(universe, trade_date)
    concept_df = concept_check(universe, trade_date)

    fatal_codes = set()
    high_risk = []
    concept_suspects = []

    if not mines_df.empty:
        for _, row in mines_df.iterrows():
            code = row['ts_code']
            if row['risk_level'] == 'FATAL':
                fatal_codes.add(code)
            else:
                high_risk.append({'code': code, 'risk': row['risk_level'], 'mines': row['mines']})

    if not concept_df.empty:
        for _, row in concept_df.iterrows():
            if row['concept_risk'] == 'HIGH':
                concept_suspects.append({'code': row['ts_code'], 'suspicion': row['suspicion']})

    fatal_codes_set = {c.split('.')[0] for c in fatal_codes}
    clean = [c for c in universe if c not in fatal_codes_set]

    return {
        'clean': clean,
        'mines_flagged': high_risk,
        'concept_suspects': concept_suspects,
        'fatal_excluded': list(fatal_codes_set),
    }


if __name__ == '__main__':
    # 快速测试: 对前一天的进攻池排雷
    import sys
    sys.path.insert(0, '.')
    from factor_pipeline import cdb as fp_db, get_clean_universe, calc_offense_score, calc_defense_score

    c = fp_db()
    td = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
    u = get_clean_universe(c, td)
    offense = calc_offense_score(c, td, u)
    top_codes = offense.head(15)['ts_code'].tolist() if not offense.empty else []
    c.close()

    print(f"日期: {td}")
    print(f"候选: {len(u)} → offense top15: {len(top_codes)}")
    print()

    result = filter_with_minesweep(top_codes, td)
    print(f"FATAL排除: {result['fatal_excluded']}")
    print(f"排雷标记: {len(result['mines_flagged'])}只")
    for m in result['mines_flagged']:
        print(f"  {m['code']} [{m['risk']}]")
        for mm in m['mines']:
            print(f"    - {mm['type']}: {mm['detail']}")
    print(f"概念炒作嫌疑: {len(result['concept_suspects'])}只")
    for c in result['concept_suspects']:
        print(f"  {c['code']} 可疑度{c['suspicion']}")
    print(f"清洁池: {result['clean']}")
