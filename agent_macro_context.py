# -*- coding: utf-8 -*-
"""
AgentQuant · 宏观上下文注入器
=============================
从DuckDB取宏观数据 → 格式化为Agent可推理的结构化叙事
注入到TradingAgents的past_context中, 让所有agent(市场/基本面/辩论/PM)都有宏观视野

用法:
  from agent_macro_context import build_macro_context
  ctx = build_macro_context()
  # 然后注入到 agent_v3_batch.py 的 past_context
"""
import sys, os
sys.path.insert(0, 'D:/AgentQuant/our')
import duckdb
from datetime import date, timedelta

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def build_macro_context() -> str:
    """构建宏观上下文, 返回一个结构化文本块, 供Agent推理使用"""
    c = duckdb.connect(DB, read_only=True)
    td = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]

    lines = []
    lines.append("=" * 60)
    lines.append("## 宏观体制上下文 (Macro Regime Context)")
    lines.append(f"分析日期: {td}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("以下是你分析任何A股标的时必须参考的宏观环境。你的技术/基本面/辩论分析不能脱离这个背景。")
    lines.append("")

        # ── 0. 角斗博弈框架 ──
    lines.append("### 0. 角斗博弈: Fed/华尔街/白宫的三角战争")
    lines.append("")
    lines.append("你不是在分析经济数据——你是在看三股势力互相算计。")
    lines.append("")
    lines.append("**Fed的囚笼**: 通胀4.2%绑架了嘴(不能松)+国债利息绑架了手(不能真加)+白宫施压(不能怂)")
    lines.append("→ 看行动不听话: 看点阵图分布/记者会语气/OIS掉期定价 vs Fed指引")
    lines.append("")
    lines.append("**华尔街的囚笼**: FOMO怕踏空+高位怕接盘→行为高度一致→一旦转向就踩踏")
    lines.append("→ 看仓位不看看法: 货币基金余额/板块轮动/对冲基金净多头")
    lines.append("")
    lines.append("**白宫的囚笼**: 中期选举需要低油价+高股价+低失业→喊话Fed降息+外交斡旋油价")
    lines.append("→ 看行动不听话: 喊话≠能做到, Fed独立性>白宫压力")
    lines.append("")
    lines.append("**当前角力(2026/6)**: 油价$99→$77, 白宫/特朗普外交压制起作用。Fed在等FOMC。华尔街在等Fed。三角僵局——谁先动谁输。")
    lines.append("")
    lines.append("### 1. WTI原油 (成本端+通胀传导)")
    r = c.execute("""
        SELECT wti FROM macro_indicators WHERE wti IS NOT NULL AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    if r and r[0]:
        wti = r[0]
        # 20日高低
        r2 = c.execute("""
            SELECT MAX(wti), MIN(wti) FROM (
                SELECT wti FROM macro_indicators WHERE wti IS NOT NULL AND trade_date <= ?
                ORDER BY trade_date DESC LIMIT 20
            )
        """, [td.isoformat()]).fetchone()
        # 5日变化
        pts = c.execute("""
            SELECT wti FROM macro_indicators WHERE wti IS NOT NULL AND trade_date <= ?
            ORDER BY trade_date DESC LIMIT 5
        """, [td.isoformat()]).fetchall()
        chg_5d = None
        if len(pts) >= 5 and pts[0][0] and pts[-1][0]:
            chg_5d = (pts[0][0] / pts[-1][0] - 1) * 100

        regime = '🟢 低油价利好' if wti < 75 else '🟡 中性' if wti < 85 else '🔴 高油价利空'
        lines.append(f"当前: ${wti:.1f} | 20日区间: ${r2[1]:.1f}~${r2[0]:.1f} | 5日变化: {chg_5d:+.1f}%" if chg_5d else f"当前: ${wti:.1f} | 20日区间: ${r2[1]:.1f}~${r2[0]:.1f}")
        lines.append(f"体制: {regime}")
        lines.append(f"含义: WTI<75→利好航空/化工/航运(成本降); WTI>90→利好石油/煤炭; WTI单日暴跌>5%→短期提振市场情绪但3日内技术反弹概率60%+")
        lines.append("")

    # ── 2. 美10Y ──
    lines.append("### 2. 美国10年期国债 (全球资产定价锚)")
    r = c.execute("""
        SELECT us10y FROM macro_indicators WHERE us10y IS NOT NULL AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    if r and r[0]:
        us10y = r[0]
        regime = '🟢 宽松' if us10y < 4.0 else '🟡 中性' if us10y < 4.5 else '🔴 紧缩'
        lines.append(f"当前: {us10y:.2f}% | 体制: {regime}")
        lines.append(f"含义: >4.5%→杀成长股估值(科创板/创业板承压); >4.8%→系统性风险; <4.0%→利好成长; 中美利差(中10Y={_get_china_10y(c,td)}%)倒挂→北向流出压力")
        lines.append("")

    # ── 3. VIX ──
    lines.append("### 3. VIX恐慌指数")
    r = c.execute("""
        SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    vix = r[0] if r else None
    if vix:
        regime = '🟢 平静' if vix < 18 else '🟡 警惕' if vix < 25 else '🔴 恐慌' if vix < 35 else '⛔ 危机'
        lines.append(f"当前: {vix:.1f} | 体制: {regime}")
        lines.append(f"含义: <15→极度平静(可能酝酿风暴); 15-20→正常; 20-28→谨慎; >28→减仓信号; >35→危机模式")
        lines.append("")

    # ── 4. 中国市场状态 ──
    lines.append("### 4. A股市场状态 (O'Neil框架)")
    # 沪深300
    r300 = c.execute("""
        SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    ma20 = c.execute("""
        SELECT AVG(close) FROM (SELECT close FROM kline_daily WHERE ts_code='sh000300'
        AND trade_date<=? ORDER BY trade_date DESC LIMIT 20)
    """, [td.isoformat()]).fetchone()
    ma60 = c.execute("""
        SELECT AVG(close) FROM (SELECT close FROM kline_daily WHERE ts_code='sh000300'
        AND trade_date<=? ORDER BY trade_date DESC LIMIT 60)
    """, [td.isoformat()]).fetchone()

    if r300 and ma20 and ma20[0]:
        idx_close = r300[0]
        idx_ma20 = ma20[0]
        idx_ma60 = ma60[0] if ma60 else 0
        above_ma20 = idx_close > idx_ma20
        above_ma60 = idx_close > idx_ma60

        if above_ma20 and above_ma60:
            trend = 'BULL (确认上升趋势 - O\'Neil)'
        elif not above_ma20 and not above_ma60:
            trend = 'BEAR (确认下降趋势 - O\'Neil)'
        else:
            trend = 'NEUTRAL (均线矛盾 - 方向不明)'

        lines.append(f"沪深300: {idx_close:.0f} | MA20: {idx_ma20:.0f} | MA60: {idx_ma60:.0f}")
        lines.append(f"趋势: {trend}")
        lines.append(f"乖离: {(idx_close/idx_ma20-1)*100:+.1f}%(vs MA20) | {(idx_close/idx_ma60-1)*100:+.1f}%(vs MA60)" if idx_ma60 else "")

    # 量价
    r_vol = c.execute("""
        SELECT SUM(amount)/1e8 FROM kline_daily WHERE trade_date=?
    """, [td.isoformat()]).fetchone()
    r_vol20 = c.execute("""
        SELECT AVG(total) FROM (SELECT SUM(amount)/1e8 total FROM kline_daily
        WHERE trade_date<=? GROUP BY trade_date ORDER BY trade_date DESC LIMIT 20)
    """, [td.isoformat()]).fetchone()
    if r_vol and r_vol[0] and r_vol20 and r_vol20[0]:
        vol_now = r_vol[0]
        vol_avg = r_vol20[0]
        vol_ratio = vol_now / vol_avg
        regime = '放量' if vol_ratio > 1.3 else '缩量' if vol_ratio < 0.7 else '正常'
        lines.append(f"全市场成交: {vol_now:.0f}亿 | 20日均: {vol_avg:.0f}亿 | 量比: {vol_ratio:.1f}x ({regime})")
        lines.append(f"含义: 缩量上涨→国家队护盘/诱多; 放量上涨→真金白银进场; 缩量下跌→抛压枯竭; 放量下跌→恐慌出逃")

    # 涨跌比
    r_adv = c.execute("""
        SELECT COUNT(*) FILTER(WHERE change_pct>0), COUNT(*) FILTER(WHERE change_pct<0)
        FROM kline_daily WHERE trade_date=?
    """, [td.isoformat()]).fetchone()
    if r_adv:
        up_count, dn_count = r_adv[0] or 0, r_adv[1] or 0
        breadth = up_count / (up_count + dn_count) if (up_count + dn_count) > 0 else 0.5
        lines.append(f"涨跌比: {up_count}涨/{dn_count}跌 ({breadth:.0%}) | {'普涨' if breadth > 0.6 else '普跌' if breadth < 0.4 else '分化'}")

    # 上证50 vs 科创50 背离检测
    r50 = c.execute("""
        SELECT (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=2 THEN close END),0)-1)*100
        FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
        FROM kline_daily WHERE ts_code='sh000016' AND trade_date<=? ORDER BY trade_date DESC LIMIT 2)
    """, [td.isoformat()]).fetchone()
    r688 = c.execute("""
        SELECT (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=2 THEN close END),0)-1)*100
        FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
        FROM kline_daily WHERE ts_code='sh000688' AND trade_date<=? ORDER BY trade_date DESC LIMIT 2)
    """, [td.isoformat()]).fetchone()
    try:
        chg50 = float(r50[0]) if r50 and r50[0] else 0
        chg688 = float(r688[0]) if r688 and r688[0] else 0
        if chg50 - chg688 > 0.005:
            lines.append(f"⚠️ 国家队疑踪: 上证50 {chg50:+.2%} vs 科创50 {chg688:+.2%} → 大票独拉、小票不跟，典型国家队手法。不是市场自发上涨，不可追高。")
    except: pass

    lines.append("")

    # ── 5. 资金流 ──
    lines.append("### 5. 资金流 (北向/南向)")
    # 检查北向数据 (同花顺hexin.cn源, ts_code='NORTH')
    r = c.execute("SELECT MAX(trade_date) FROM north_bound_flow WHERE ts_code='NORTH' AND net_flow != 0").fetchone()
    north_last_real = r[0] if r else None

    if north_last_real and (td - north_last_real).days < 5:
        nb5 = c.execute("""
            SELECT COALESCE(SUM(net_flow),0) FROM (
                SELECT net_flow FROM north_bound_flow WHERE ts_code='NORTH' AND trade_date<=? ORDER BY trade_date DESC LIMIT 5
            )
        """, [td.isoformat()]).fetchone()
        nb20 = c.execute("""
            SELECT COALESCE(SUM(net_flow),0) FROM (
                SELECT net_flow FROM north_bound_flow WHERE ts_code='NORTH' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
            )
        """, [td.isoformat()]).fetchone()
        nb5v = nb5[0] if nb5 else 0
        nb20v = nb20[0] if nb20 else 0
        nb_dir = '外资流入' if nb5v > 30 else '外资流出' if nb5v < -30 else '外资观望'
        lines.append(f"北向5日: {nb5v:+.0f}亿 | 北向20日: {nb20v:+.0f}亿 | 数据源: 同花顺hexin.cn")
        lines.append(f"方向: {nb_dir}")
    else:
        lines.append(f"⚠️ 北向数据积累中 (当前{1 if north_last_real else 0}天, 需≥5天出方向)")
        lines.append(f"最新: {north_last_real} (同花顺hexin.cn源)")
        if north_last_real:
            nb = c.execute("SELECT net_flow FROM north_bound_flow WHERE ts_code='NORTH' AND trade_date=?", [north_last_real.isoformat()]).fetchone()
            if nb: lines.append(f"最新日净流入: {nb[0]:+.1f}亿")

    # 南向 (独立源)
    sb5 = c.execute("""
        SELECT COALESCE(SUM(south_net),0) FROM (
            SELECT south_net FROM macro_indicators WHERE south_net IS NOT NULL AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 5
        )
    """, [td.isoformat()]).fetchone()
    sb5v = sb5[0] if sb5 else 0
    if sb5v != 0:
        lines.append(f"南向5日: {sb5v:+.0f}亿 (资金南下港股)")
    lines.append("")

    # ── 6. 事件风险 ──
    lines.append("### 6. 事件风险日历")
    today = date.today()
    events = []
    if date(2026, 6, 15) <= today <= date(2026, 6, 18):
        events.append("🔴 FOMC会议(6/16-17沃什首秀) — 利率维持但点阵图可能偏鹰，美10Y有上行风险")
    if date(2026, 6, 16) <= today <= date(2026, 6, 20):
        events.append("🟡 陆家嘴论坛(6/17) — 资本市场政策催化预期")
    if today.weekday() == 4:  # Friday
        events.append("⚠️ 周末2天空窗 — 任何消息都会被放大，减仓避险是A股惯例")

    if events:
        for e in events:
            lines.append(f"  - {e}")
    else:
        lines.append("  (无重大事件)")
    lines.append("")

    # ── 7. 宏观综合结论 ──
    lines.append("### 7. 宏观综合指引")
    # Compute composite
    score = 50
    if r300 and ma20 and ma20[0] and r300[0] > ma20[0]: score += 15
    if vix and vix < 20: score += 5
    if wti < 80: score += 5
    if us10y < 4.5: score += 5
    if nb5 and nb5[0] > 0: score += 5
    if nb5 and nb5[0] < -30: score -= 10

    if score >= 70:
        guidance = "🟢 进攻环境: 宏观有利，积极寻找机会。优先进攻池(通信/电子/CPO/AI)，关注技术突破+放量确认。"
    elif score >= 50:
        guidance = "🟡 谨慎环境: 50%仓位上限。可配置防守池(银行/公用/红利)，进攻池只买Agent明确BUY且基本面可信度≥B级的标的。"
    elif score >= 35:
        guidance = "🔴 防御环境: 30%仓位上限。冻结进攻端，仅保留公用事业/银行等MA60线上的防守仓位。"
    else:
        guidance = "⛔ 危机环境: 10%仓位上限。清仓所有进攻仓位，防守仓位减至最低，现金为王。"

    lines.append(f"综合评分: {score}/100 → {guidance}")
    lines.append("")

    # ── 8. 矛盾推演框架 ──
    lines.append("### 8. 矛盾推演: 当教科书失灵")
    lines.append("")
    lines.append("宏观分析最容易犯的错误: 看到两个指标, 用教科书公式推结论, 不管现实。")
    lines.append("当指标出现'反常'组合时, 不要急于下结论, 按以下步骤推演:")
    lines.append("")
    lines.append("**步骤1: 识别矛盾**")
    lines.append("  例: 存款利率<1% + 人民币升破6.8 → 利率平价理论说应该贬值, 现实是升值")
    lines.append("  问自己: 教科书假设什么? (资本自由流动) → 中国有这个前提吗? (没有)")
    lines.append("")
    lines.append("**步骤2: 找被忽略的玩家**")
    lines.append("  利率平价关注的是'套利资金'。但在中国, 有更大的玩家:")
    lines.append("  - 出口商: 年顺差5000亿美元, 必须结汇卖出美元买入RMB → 刚性需求, 不受利率影响")
    lines.append("  - 资本管制: 每人每年5万美元额度, QFII限额 → 套利通道被堵死")
    lines.append("  → 贸易结汇量 >>> 套利资金量 → 汇率由贸易决定, 不由利差决定")
    lines.append("")
    lines.append("**步骤3: 算实际利率, 不算名义利率**")
    lines.append("  中国: 1% - CPI 1.2% = 实际利率 -0.2%")
    lines.append("  美国: 4.5% - CPI 4.2% = 实际利率 +0.3%")
    lines.append("  名义利差3.5%→实际利差仅0.5%→扣换汇成本→套利空间≈零")
    lines.append("")
    lines.append("**步骤4: 推A股传导链**")
    lines.append("  RMB升值 → 利好(航空/造纸/石化: 进口成本降)")
    lines.append("  RMB升值 → 利空(纺织/低端制造: 出口利润被汇率吃掉)")
    lines.append("  低利率 → 利好(高股息/红利: 理财资金被挤出到股市)")
    lines.append("  低利率+RMB升值 → 外资买A股成本高, 但汇率稳定=不会恐慌出逃")
    lines.append("")
    lines.append("**通用铁律: 当两个指标同时指向相反方向 → 不是你读错了, 是有一个隐藏变量在起作用。你的任务就是找到它。**")
    lines.append("")

    # ── 9. 穿透判断方法论 ──
    lines.append("### 9. 穿透判断: 不要问'利好谁'——先问'谁被错杀了'")
    lines.append("")
    lines.append("任何宏观事件发生后, 不要直接跳到'利好XX板块'。按以下三步:")
    lines.append("")
    lines.append("**步骤1: 扫描事件前N日的各板块涨跌**")
    lines.append("  如果一个板块在事件前已经涨了5%——利好早已被聪明钱定价, 你追进去是接盘")
    lines.append("  如果一个板块在事件前跌了5%——它被错杀了, 事件可能触发反弹")
    lines.append("  如果一个板块在事件前基本没动——事件对它影响不大, 别硬找理由")
    lines.append("")
    lines.append("**步骤2: 用历史数据验证**")
    lines.append("  历史上类似事件发生后, 这个板块真的涨了吗? 引用validated_rules中的回测数据")
    lines.append("  如果历史上不成立 → 不是这次不一样, 是市场共识一直错")
    lines.append("  如果历史上成立 → 标注样本量(如29次)和胜率(如52%)")
    lines.append("")
    lines.append("**步骤3: 判断持续性**")
    lines.append("  1-3天行情: 不调仓, 不值得")
    lines.append("  1-4周趋势: 可以考虑调仓")
    lines.append("  判断依据: 历史回测中信号在第几天开始衰减")
    lines.append("")
    lines.append("**穿透判断禁止:**")
    lines.append("  - 禁止说'油价跌利好航空'而不查航空前5日是否已经涨了")
    lines.append("  - 禁止只分析一个板块→必须全扫至少5个相关板块做比较")
    lines.append("  - 禁止不查历史数据就说'这次不一样'")
    lines.append("")

    # ── 10. 四维评分 + 事实推演分离 ──
    lines.append("### 10. 信号评分标准 (每个信号必须标注)")
    lines.append("")
    lines.append("| 维度 | 1分 | 3分 | 5分 |")
    lines.append("|------|-----|-----|-----|")
    lines.append("| 影响力 | 影响单只股票 | 影响整个板块 | 影响全市场 |")
    lines.append("| 时效性 | 数据>3天前 | 1-2天前 | 实时/今天 |")
    lines.append("| 确定性 | 单一来源/传闻 | 多源交叉验证 | 官方数据+盘面确认 |")
    lines.append("| 交易性 | 无法执行(停牌/流动性) | 可执行但有摩擦 | 流动性好/可精准执行 |")
    lines.append("")
    lines.append("综合分 < 10: 不参与决策 | 10-15: 参考 | 16-20: 核心依据")
    lines.append("")
    lines.append("### 11. 事实 vs 推演 (每条结论必须标注)")
    lines.append("")
    lines.append("- **事实**: 数据直接可得, 无需推断 (例: WTI=76.65, 来源=DuckDB)")
    lines.append("- **事实+推演**: 基于事实的逻辑推断, 有历史规律支撑 (例: WTI暴跌后电子20日+3.4%, 29次样本)")
    lines.append("- **猜测**: 无数据支撑的主观判断 (必须标注'置信度低', 不参与仓位决策)")
    lines.append("")
    lines.append("禁止把推演当事实陈述。禁止把猜测当推演陈述。")

    lines.append("---")
    lines.append("**重要**: 上述宏观数据必须在你的分析中体现。你的个股技术面/基本面/辩论结论必须与宏观体制一致。如果个股信号与宏观方向冲突，必须标注冲突点并解释为什么你认为个股可以逆宏观。如果多个指标出现'矛盾'组合, 不要忽略——那可能正是最关键的信号。")
    lines.append("=" * 60)

    c.close()
    return '\n'.join(lines)


def _get_china_10y(c, td) -> str:
    r = c.execute("""
        SELECT china_10y FROM macro_indicators WHERE china_10y IS NOT NULL AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    return f"{r[0]:.2f}" if r and r[0] else "?"


if __name__ == '__main__':
    ctx = build_macro_context()
    print(ctx)
