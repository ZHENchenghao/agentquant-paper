# -*- coding: utf-8 -*-
"""
AgentQuant · 宏观分析模块
=========================
从DuckDB读取宏观数据 → 六维分析 → 风控门禁 → 宏观裁决

维度:
  1. WTI油价 (成本端/通胀传导)
  2. 美10Y利率 (全球资产定价锚)
  3. 中国市场状态 (O'Neil + 量价分析)
  4. 国家队检测 (五重证据)
  5. 资金流 (北向/主力/成交额)
  6. 宏观事件日历

输出: MacroVerdict (风控门禁 + 仓位上限 + 纠错线)
"""
import sys, os, io, json
sys.path.insert(0, 'D:/AgentQuant/our')
import duckdb, numpy as np
from datetime import date, timedelta
from dataclasses import dataclass, field, asdict

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB, read_only=True)

# ═══════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════

@dataclass
class MacroSnapshot:
    """宏观快照"""
    trade_date: str = ''
    wti: float = None           # WTI原油
    wti_20d_high: float = None  # 20日最高
    wti_20d_low: float = None   # 20日最低
    wti_change_5d: float = None # 5日涨跌%
    us10y: float = None         # 美10年国债
    us10y_20d_high: float = None
    china_10y: float = None
    usdcny: float = None
    spread_cn_us: float = None
    gold: float = None
    vix: float = None
    shibor_on: float = None
    south_net: float = None     # 南向资金(亿)

@dataclass
class MarketState:
    """中国市场状态"""
    trade_date: str = ''
    sh000300: float = None      # 沪深300收盘
    sh000300_ma20: float = None # 沪深300 20日均线
    sh000300_ma60: float = None
    sh000300_vol_5d: float = None  # 5日均量
    sh000300_vol_20d: float = None
    sh000016: float = None      # 上证50
    sh000688: float = None      # 科创50
    sz399006: float = None      # 创业板
    total_amount: float = None  # 全市场成交额(亿)
    total_amount_20d: float = None
    advance_count: int = None   # 上涨家数
    decline_count: int = None   # 下跌家数
    # 派生指标
    trend: str = 'NEUTRAL'      # BULL/BEAR/NEUTRAL
    vol_ratio: float = 1.0      # 量比(今日/20日均)
    breadth: float = 0.5        # 涨跌比

@dataclass
class NationalTeam:
    """国家队检测"""
    active: bool = False
    confidence: float = 0.0     # 0-100
    evidence: list = field(default_factory=list)  # 证据链

@dataclass
class CapitalFlow:
    """资金流 (铁律#17: None=数据不可用)"""
    north_net_5d: float = None     # 北向5日净流入(亿), None=数据不可用
    north_net_20d: float = None
    south_net_5d: float = 0
    total_amount_trend: str = 'STABLE'  # SURGE/SHRINK/STABLE
    main_force_direction: str = 'NEUTRAL'  # NO_DATA=北向不可用

@dataclass
class MacroVerdict:
    """宏观裁决"""
    risk_gate: str = 'NEUTRAL'    # ATTACK / CAUTION / DEFENSE / CRISIS
    position_cap: float = 1.0     # 仓位上限 (0~1.0)
    wti_score: float = 50
    us10y_score: float = 50
    market_score: float = 50
    ntl_team_score: float = 50
    flow_score: float = 50
    event_score: float = 50
    composite: float = 50        # 综合评分 (0=极空 100=极多)
    reasons: list = field(default_factory=list)
    correction_line: str = ''

# ═══════════════════════════════════════════
# 数据采集
# ═══════════════════════════════════════════

def get_macro_snapshot() -> MacroSnapshot:
    """从DuckDB读取宏观快照"""
    c = cdb()
    snap = MacroSnapshot()

    # 最新交易日
    r = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()
    td = r[0]
    snap.trade_date = str(td)

    # macro_indicators 最新行
    r = c.execute("""
        SELECT wti, us10y, china_10y, usdcny, gold, vix, shibor_on,
               south_net, spread_cn_us
        FROM macro_indicators
        WHERE trade_date <= ? ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    if r:
        snap.wti = r[0]; snap.us10y = r[1]; snap.china_10y = r[2]
        snap.usdcny = r[3]; snap.gold = r[4]; snap.vix = r[5]
        snap.shibor_on = r[6]; snap.south_net = r[7]
        snap.spread_cn_us = r[8]

    # WTI 20日范围 + 5日变化
    if snap.wti:
        r = c.execute("""
            SELECT MAX(wti), MIN(wti) FROM (
                SELECT wti FROM macro_indicators
                WHERE wti IS NOT NULL AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 20
            )
        """, [td.isoformat()]).fetchone()
        if r: snap.wti_20d_high, snap.wti_20d_low = r[0], r[1]
        r = c.execute("""
            SELECT wti FROM macro_indicators
            WHERE wti IS NOT NULL AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 5 OFFSET 1
        """, [td.isoformat()]).fetchall()
        if r and len(r) >= 4 and snap.wti:
            old_wti = r[-1][0]
            if old_wti and old_wti > 0:
                snap.wti_change_5d = round((snap.wti / old_wti - 1) * 100, 1)

    # US10Y 20日范围
    if snap.us10y:
        r = c.execute("""
            SELECT MAX(us10y) FROM (
                SELECT us10y FROM macro_indicators
                WHERE us10y IS NOT NULL AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 20
            )
        """, [td.isoformat()]).fetchone()
        if r: snap.us10y_20d_high = r[0]

    # VIX fallback: 从macro_indicators中取最新非空值
    if snap.vix is None:
        r = c.execute("""
            SELECT vix FROM macro_indicators
            WHERE vix IS NOT NULL AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1
        """, [td.isoformat()]).fetchone()
        if r: snap.vix = r[0]

    c.close()
    return snap


def get_market_state(td: date) -> MarketState:
    """中国市场状态分析"""
    c = cdb()
    st = MarketState(trade_date=str(td))

    # 指数收盘
    for code, attr in [('sh000300', 'sh000300'), ('sh000016', 'sh000016'),
                        ('sh000688', 'sh000688'), ('sz399006', 'sz399006')]:
        r = c.execute("SELECT close FROM kline_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                      [code, td.isoformat()]).fetchone()
        if r: setattr(st, attr, r[0])

    # 沪深300均线
    r = c.execute("""
        SELECT AVG(close) FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 20
        )
    """, [td.isoformat()]).fetchone()
    if r: st.sh000300_ma20 = r[0]

    r = c.execute("""
        SELECT AVG(close) FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 60
        )
    """, [td.isoformat()]).fetchone()
    if r: st.sh000300_ma60 = r[0]

    # 量比
    r = c.execute("""
        SELECT AVG(amount) FROM (
            SELECT amount FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 5
        )
    """, [td.isoformat()]).fetchone()
    if r: st.sh000300_vol_5d = r[0]

    r = c.execute("""
        SELECT AVG(amount) FROM (
            SELECT amount FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 20
        )
    """, [td.isoformat()]).fetchone()
    if r: st.sh000300_vol_20d = r[0]

    if st.sh000300_vol_5d and st.sh000300_vol_20d and st.sh000300_vol_20d > 0:
        st.vol_ratio = round(st.sh000300_vol_5d / st.sh000300_vol_20d, 2)

    # 全市场成交额(亿)
    r = c.execute("""
        SELECT SUM(amount)/1e8 FROM kline_daily WHERE trade_date=?
    """, [td.isoformat()]).fetchone()
    if r and r[0]: st.total_amount = round(r[0], 1)

    r = c.execute("""
        SELECT AVG(total) FROM (
            SELECT SUM(amount)/1e8 total FROM kline_daily
            WHERE trade_date<=? GROUP BY trade_date ORDER BY trade_date DESC LIMIT 20
        )
    """, [td.isoformat()]).fetchone()
    if r and r[0]: st.total_amount_20d = round(r[0], 1)

    # 趋势判断: O'Neil方法
    if st.sh000300 and st.sh000300_ma20:
        above_ma20 = st.sh000300 > st.sh000300_ma20
        above_ma60 = st.sh000300 > st.sh000300_ma60 if st.sh000300_ma60 else None

        if above_ma20 and above_ma60:
            st.trend = 'BULL'      # 确认上升趋势
        elif not above_ma20 and not above_ma60:
            st.trend = 'BEAR'      # 确认下降趋势
        else:
            st.trend = 'NEUTRAL'   # 均线矛盾

    # 涨跌比
    r = c.execute("""
        SELECT COUNT(*) FILTER(WHERE change_pct>0),
               COUNT(*) FILTER(WHERE change_pct<0)
        FROM kline_daily WHERE trade_date=?
    """, [td.isoformat()]).fetchone()
    if r:
        st.advance_count = r[0]
        st.decline_count = r[1]
        total_ud = (r[0] or 0) + (r[1] or 0)
        st.breadth = round((r[0] or 0) / total_ud, 2) if total_ud > 0 else 0.5

    c.close()
    return st


def detect_national_team(st: MarketState) -> NationalTeam:
    """国家队五重证据检测"""
    nt = NationalTeam()
    score = 0

    # 证据1: 权重独拉 (上证50 vs 科创50 涨幅差>1%)
    if st.sh000016 and st.sh000688:
        r50 = cdb().execute("""
            SELECT (MAX(CASE WHEN rn=1 THEN close END)/
                    NULLIF(MAX(CASE WHEN rn=2 THEN close END),0)-1)*100
            FROM (SELECT close, ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                  FROM kline_daily WHERE ts_code='sh000016' AND trade_date<=? ORDER BY trade_date DESC LIMIT 2)
        """, [st.trade_date]).fetchone()
        r688 = cdb().execute("""
            SELECT (MAX(CASE WHEN rn=1 THEN close END)/
                    NULLIF(MAX(CASE WHEN rn=2 THEN close END),0)-1)*100
            FROM (SELECT close, ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                  FROM kline_daily WHERE ts_code='sh000688' AND trade_date<=? ORDER BY trade_date DESC LIMIT 2)
        """, [st.trade_date]).fetchone()
        try:
            chg50 = float(r50[0]) if r50 and r50[0] else 0
            chg688 = float(r688[0]) if r688 and r688[0] else 0
            if chg50 - chg688 > 0.008:  # >0.8%差异
                score += 20
                nt.evidence.append(f'权重独拉: 上证50 {chg50:+.2%} vs 科创50 {chg688:+.2%}')
        except: pass

    # 证据2: 上证50放量 vs 全市场缩量
    if st.sh000300_vol_5d and st.sh000300_vol_20d and st.total_amount and st.total_amount_20d:
        idx_vol_ratio = st.sh000300_vol_5d / st.sh000300_vol_20d
        mkt_vol_ratio = st.total_amount / st.total_amount_20d
        if idx_vol_ratio > 1.3 and mkt_vol_ratio < 0.8:
            score += 20
            nt.evidence.append(f'量价背离: 沪深300量比{idx_vol_ratio:.1f}x vs 全市场{mkt_vol_ratio:.1f}x')

    # 证据3: 北向 vs 指数背离 (同花顺源 ts_code='NORTH')
    if st.sh000300:
        from datetime import date as dt_date
        r = cdb().execute("""
            SELECT MAX(trade_date) FROM north_bound_flow WHERE ts_code='NORTH' AND net_flow != 0
        """).fetchone()
        north_last = r[0] if r else None
        st_date = dt_date.fromisoformat(st.trade_date) if isinstance(st.trade_date, str) else st.trade_date
        north_ok = north_last and (st_date - north_last).days < 5 if north_last else False

        if north_ok:
            nb = cdb().execute("""
                SELECT net_flow FROM north_bound_flow
                WHERE ts_code='NORTH' AND trade_date=?
            """, [st.trade_date]).fetchone()
            nb_today = nb[0] if nb else 0
            if abs(nb_today) < 2 and st.trend == 'BULL':
                score += 15
                nt.evidence.append(f'北向近似挂零({nb_today:.1f}亿)但沪深300>MA20')
        else:
            nt.evidence.append('⚠️ 北向历史不足(需积累≥5天), 跳过北向-指数背离检测')

    # 证据4: 涨跌比<0.4但指数涨 (少数大票拉指数)
    if st.breadth < 0.45 and st.sh000300 and st.sh000300_ma20:
        r = cdb().execute("""
            SELECT (MAX(CASE WHEN rn=1 THEN close END)/
                    NULLIF(MAX(CASE WHEN rn=2 THEN close END),0)-1)
            FROM (SELECT close, ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                  FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 2)
        """, [st.trade_date]).fetchone()
        if r and r[0] and r[0] > 0:
            score += 15
            nt.evidence.append(f'涨跌比{st.breadth:.0%}但沪深300收涨 → 少数大票拉指数')

    # 证据5: 全市场成交极度萎缩
    if st.total_amount and st.total_amount_20d and st.total_amount < st.total_amount_20d * 0.7:
        score += 10
        nt.evidence.append(f'全市场缩量至{st.total_amount:.0f}亿 (vs 20日均{st.total_amount_20d:.0f}亿)')

    nt.confidence = score
    nt.active = score >= 40
    return nt


def get_capital_flow(td: date) -> CapitalFlow:
    """资金流分析"""
    c = cdb()
    cf = CapitalFlow()

    # 北向5日/20日累计 (同花顺hexin.cn源, ts_code='NORTH')
    r = c.execute("""
        SELECT MAX(trade_date) FROM north_bound_flow WHERE ts_code='NORTH' AND net_flow != 0
    """).fetchone()
    north_last_real = r[0] if r else None
    north_stale = north_last_real and (td - north_last_real).days > 5 if north_last_real else True

    if north_stale:
        cf.north_net_5d = None
        cf.north_net_20d = None
        cf.main_force_direction = 'NO_DATA'
    else:
        r = c.execute("""
            SELECT COALESCE(SUM(net_flow),0) FROM (
                SELECT net_flow FROM north_bound_flow
                WHERE ts_code='NORTH' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 5
            )
        """, [td.isoformat()]).fetchone()
        if r: cf.north_net_5d = round(r[0], 1)

        r = c.execute("""
            SELECT COALESCE(SUM(net_flow),0) FROM (
                SELECT net_flow FROM north_bound_flow
                WHERE ts_code='NORTH' AND trade_date <= ? ORDER BY trade_date DESC LIMIT 20
            )
        """, [td.isoformat()]).fetchone()
        if r: cf.north_net_20d = round(r[0], 1)

    # 南向5日
    r = c.execute("""
        SELECT COALESCE(SUM(south_net),0) FROM (
            SELECT south_net FROM macro_indicators
            WHERE south_net IS NOT NULL AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 5
        )
    """, [td.isoformat()]).fetchone()
    if r: cf.south_net_5d = round(r[0], 1)

    # 全市场成交额趋势
    r = c.execute("""
        SELECT AVG(total) FROM (
            SELECT SUM(amount)/1e8 total FROM kline_daily
            WHERE trade_date<=? GROUP BY trade_date ORDER BY trade_date DESC LIMIT 5
        )
    """, [td.isoformat()]).fetchone()
    r20 = c.execute("""
        SELECT AVG(total) FROM (
            SELECT SUM(amount)/1e8 total FROM kline_daily
            WHERE trade_date<=? GROUP BY trade_date ORDER BY trade_date DESC LIMIT 20
        )
    """, [td.isoformat()]).fetchone()
    if r and r[0] and r20 and r20[0]:
        ratio = r[0] / r20[0]
        if ratio > 1.3: cf.total_amount_trend = 'SURGE'
        elif ratio < 0.7: cf.total_amount_trend = 'SHRINK'
        else: cf.total_amount_trend = 'STABLE'

    # 主力方向(简化: 北向+全市场量能)
    if cf.north_net_5d is not None:
        if cf.north_net_5d > 50 and cf.total_amount_trend == 'SURGE':
            cf.main_force_direction = 'BULLISH'
        elif cf.north_net_5d < -50:
            cf.main_force_direction = 'BEARISH'

    c.close()
    return cf


# ═══════════════════════════════════════════
# 裁决引擎
# ═══════════════════════════════════════════

def calc_wti_score(snap: MacroSnapshot) -> tuple:
    """WTI评分: 油价低→利好(成本低), 油价高→利空"""
    reasons = []
    if snap.wti is None:
        return 50, reasons

    wti = snap.wti
    score = 50
    # <$70 = 利好 (80分), $70-85 = 中性(60分), $85-95 = 偏空(40分), >$95 = 利空(20分)
    if wti < 70:
        score = 80; reasons.append(f'WTI=${wti:.1f}<70 → 成本压力低, 利好制造业')
    elif wti < 80:
        score = 65; reasons.append(f'WTI=${wti:.1f} 70-80区间 → 中性偏利好')
    elif wti < 85:
        score = 55; reasons.append(f'WTI=${wti:.1f} 80-85区间 → 中性')
    elif wti < 95:
        score = 40; reasons.append(f'WTI=${wti:.1f} 85-95区间 → 成本压力偏大')
    else:
        score = 20; reasons.append(f'WTI=${wti:.1f}>95 → 高油价压制制造业利润')

    # 暴跌加成
    if snap.wti_change_5d is not None and snap.wti_change_5d < -5:
        score = min(100, score + 10)
        reasons.append(f'WTI 5日暴跌{snap.wti_change_5d:+.1f}% → 短期利好(进口成本骤降)')

    return score, reasons


def calc_us10y_score(snap: MacroSnapshot) -> tuple:
    """美10Y评分: 利率高→杀估值, 利率低→利好成长"""
    reasons = []
    if snap.us10y is None:
        return 50, reasons

    y = snap.us10y
    score = 50
    if y < 4.0:
        score = 75; reasons.append(f'美10Y={y:.2f}%<4.0 → 估值压力轻')
    elif y < 4.3:
        score = 60; reasons.append(f'美10Y={y:.2f}% 4.0-4.3 → 偏利好')
    elif y < 4.5:
        score = 50; reasons.append(f'美10Y={y:.2f}% 4.3-4.5 → 中性')
    elif y < 4.8:
        score = 35; reasons.append(f'美10Y={y:.2f}% 4.5-4.8 → 杀成长估值')
    else:
        score = 20; reasons.append(f'美10Y={y:.2f}%>4.8 → 严重压制权益估值')

    return score, reasons


def calc_market_score(st: MarketState, nt: NationalTeam) -> tuple:
    """市场状态评分"""
    reasons = []
    score = 50

    if st.trend == 'BULL':
        score += 20; reasons.append('沪深300>MA20+MA60 → O\'Neil确认上升趋势')
    elif st.trend == 'BEAR':
        score -= 20; reasons.append('沪深300<MA20+MA60 → O\'Neil确认下降趋势')

    if st.vol_ratio > 1.3:
        score += 10; reasons.append(f'放量({st.vol_ratio:.1f}x) → 增量资金进场')
    elif st.vol_ratio < 0.7:
        score -= 10; reasons.append(f'缩量({st.vol_ratio:.1f}x) → 资金观望')

    if st.breadth > 0.6:
        score += 10; reasons.append(f'涨跌比{st.breadth:.0%} → 普涨, 健康')
    elif st.breadth < 0.4:
        score -= 10; reasons.append(f'涨跌比{st.breadth:.0%} → 多数票下跌')

    if nt.active:
        score -= 15
        reasons.append(f'国家队疑似护盘(置信度{nt.confidence:.0f}/100) → 自然买盘不足')
        for e in nt.evidence:
            reasons.append(f'  └ {e}')

    if st.total_amount and st.total_amount_20d and st.total_amount < st.total_amount_20d * 0.6:
        score -= 10
        reasons.append(f'全市场成交{st.total_amount:.0f}亿 → 极度缩量(20日均{st.total_amount_20d:.0f}亿)')

    return max(0, min(100, score)), reasons


def calc_flow_score(cf: CapitalFlow) -> tuple:
    """资金流评分 (铁律#17: 数据不可用时标注, 不造假)"""
    reasons = []
    score = 50

    if cf.main_force_direction == 'NO_DATA':
        score = 50  # 中性, 不偏多不偏空
        reasons.append('[!] 北向资金数据不可用 (最后真实数据: 2024-08-16, 东财API屏蔽)')
        reasons.append('-> 资金流评分退化为中性, 以北向缺失前的历史数据和南向/全市场量能为辅')
    else:
        if cf.north_net_5d is not None and cf.north_net_5d > 100:
            score += 20; reasons.append(f'北向5日净流入+{cf.north_net_5d:.0f}亿 → 外资积极')
        elif cf.north_net_5d is not None and cf.north_net_5d > 30:
            score += 10; reasons.append(f'北向5日+{cf.north_net_5d:.0f}亿 → 温和流入')
        elif cf.north_net_5d is not None and cf.north_net_5d < -100:
            score -= 20; reasons.append(f'北向5日净流出{cf.north_net_5d:.0f}亿 → 外资撤离')
        elif cf.north_net_5d is not None and cf.north_net_5d < -30:
            score -= 10; reasons.append(f'北向5日-{abs(cf.north_net_5d):.0f}亿 → 温和流出')

    if cf.total_amount_trend == 'SURGE':
        score += 10; reasons.append('全市场放量 → 资金进场')
    elif cf.total_amount_trend == 'SHRINK':
        score -= 10; reasons.append('全市场缩量 → 资金观望')

    if cf.south_net_5d > 50:
        score -= 5; reasons.append(f'南向{cf.south_net_5d:.0f}亿 → 资金南下港股分流')

    return max(0, min(100, score)), reasons


def calc_event_score(snap: MacroSnapshot) -> tuple:
    """事件风险评分 (简化版: 基于日历+数据推演)"""
    reasons = []
    score = 50

    # FOMC周影响
    td = date.today()
    # 6/16-17 FOMC周
    if date(2026, 6, 15) <= td <= date(2026, 6, 18):
        score -= 10
        reasons.append('⚠️ FOMC周(6/16-17沃什首秀) → 政策不确定性高 → 防御性折价')

    # 陆家嘴论坛
    if date(2026, 6, 16) <= td <= date(2026, 6, 20):
        score += 5
        reasons.append('陆家嘴论坛(6/17) → 政策催化预期')

    # WTI暴跌后反弹风险
    if snap.wti_change_5d is not None and snap.wti_change_5d < -5:
        score -= 5
        reasons.append(f'WTI 5日急跌{snap.wti_change_5d:+.1f}% → 历史规律: 急跌后3日内反弹概率>60%')

    # VIX
    if snap.vix:
        if snap.vix > 30:
            score -= 15; reasons.append(f'VIX={snap.vix:.1f}>30 → 恐慌模式')
        elif snap.vix > 25:
            score -= 5; reasons.append(f'VIX={snap.vix:.1f}>25 → 偏高')
        elif snap.vix < 15:
            score += 5; reasons.append(f'VIX={snap.vix:.1f}<15 → 极度平静')

    return max(0, min(100, score)), reasons


def determine_risk_gate(composite: float, snap: MacroSnapshot, st: MarketState) -> str:
    """四重门裁决"""
    # CRISIS: WTI>$95 OR US10Y>4.8% OR VIX>35
    if snap.wti and snap.wti > 95:
        return 'CRISIS'
    if snap.us10y and snap.us10y > 4.8:
        return 'CRISIS'
    if snap.vix and snap.vix > 35:
        return 'CRISIS'

    # DEFENSE: composite<35 OR BEAR trend
    if composite < 35 or st.trend == 'BEAR':
        return 'DEFENSE'

    # CAUTION: composite 35-55
    if composite < 55:
        return 'CAUTION'

    return 'ATTACK'


def position_cap_from_gate(gate: str) -> float:
    """门禁→仓位上限"""
    return {'ATTACK': 0.8, 'CAUTION': 0.5, 'DEFENSE': 0.3, 'CRISIS': 0.1}[gate]


# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

def run() -> MacroVerdict:
    """跑完整宏观分析 → 返回裁决"""
    snap = get_macro_snapshot()
    td = date.fromisoformat(snap.trade_date) if snap.trade_date else date.today()
    st = get_market_state(td)
    nt = detect_national_team(st)
    cf = get_capital_flow(td)

    # 五维评分
    wti_score, wti_reasons = calc_wti_score(snap)
    us10y_score, us10y_reasons = calc_us10y_score(snap)
    mkt_score, mkt_reasons = calc_market_score(st, nt)
    flow_score, flow_reasons = calc_flow_score(cf)
    event_score, event_reasons = calc_event_score(snap)

    # 综合评分 (加权平均)
    composite = round(
        wti_score * 0.20 +
        us10y_score * 0.20 +
        mkt_score * 0.25 +
        flow_score * 0.15 +
        event_score * 0.20
    )

    gate = determine_risk_gate(composite, snap, st)
    cap = position_cap_from_gate(gate)

    # 纠错线
    corrections = []
    if snap.wti and snap.wti > 90:
        corrections.append(f'WTI>${snap.wti:.0f}→降仓位至{cap*0.5:.0%}')
    if snap.us10y and snap.us10y > 4.6:
        corrections.append(f'US10Y>{snap.us10y:.2f}%→冻结进攻端')
    if st.trend == 'BEAR':
        corrections.append('沪深300<MA20→全仓防守')

    verdict = MacroVerdict(
        risk_gate=gate,
        position_cap=cap,
        wti_score=wti_score,
        us10y_score=us10y_score,
        market_score=mkt_score,
        ntl_team_score=100 - nt.confidence if nt.active else 50,
        flow_score=flow_score,
        event_score=event_score,
        composite=composite,
        reasons=(wti_reasons + us10y_reasons + mkt_reasons + flow_reasons + event_reasons),
        correction_line=' | '.join(corrections) if corrections else '无触发'
    )

    return verdict


def print_verdict(v: MacroVerdict):
    """打印裁决"""
    gate_tag = {'ATTACK': '[ATTACK]', 'CAUTION': '[CAUTION]', 'DEFENSE': '[DEFENSE]', 'CRISIS': '[CRISIS]'}
    print(f"""
{'='*60}
  AgentQuant Macro Verdict
{'='*60}
  Gate: {gate_tag.get(v.risk_gate, 'UNKNOWN')} {v.risk_gate}
  Position Cap: {v.position_cap:.0%}
  Composite: {v.composite}/100
  ---------------------------------
  Scores:
    WTI:       {v.wti_score}/100 (weight 20%)
    US10Y:     {v.us10y_score}/100 (weight 20%)
    Market:    {v.market_score}/100 (weight 25%)
    Flow:      {v.flow_score}/100 (weight 15%)
    Event:     {v.event_score}/100 (weight 20%)
  ---------------------------------
  Reasoning:
""")
    for r in v.reasons:
        print(f'    {r}')
    print(f"""
  Correction: {v.correction_line}
{'='*60}
""")


if __name__ == '__main__':
    v = run()
    print_verdict(v)
