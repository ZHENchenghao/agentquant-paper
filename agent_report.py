# -*- coding: utf-8 -*-
"""
AgentQuant · 完整日报生成器
============================
组合: 宏观裁决 + V3策略信号 + Agent多空辩论 + ETF信号 + 三段论结论

用法:
  python agent_report.py              → 生成完整日报(需先有agent日志)
  python agent_report.py --quick      → 快速版(跳过agent, 只用宏观+V3)
"""
import sys, os, io, json, glob
sys.path.insert(0, 'D:/AgentQuant/our')
from datetime import date, timedelta
from agent_macro import (
    run as macro_run, get_macro_snapshot, get_market_state,
    detect_national_team, get_capital_flow,
    calc_wti_score, calc_us10y_score, calc_market_score,
    calc_flow_score, calc_event_score, determine_risk_gate,
    position_cap_from_gate,
    MacroVerdict, MarketState, NationalTeam, CapitalFlow, MacroSnapshot
)
import duckdb
DB = 'D:/FreeFinanceData/data/duckdb/finance.db'


def load_agent_results():
    """加载TradingAgents日志 → 提取完整推理"""
    log_dir = os.path.expanduser('~/.tradingagents/logs')
    # 优先取最新日期
    files = glob.glob(os.path.join(log_dir, '*/TradingAgentsStrategy_logs/full_states_log_*.json'))

    # 按日期分组, 取最新的
    date_files = {}
    for f in files:
        fn = os.path.basename(f)
        d = fn.replace('full_states_log_','').replace('.json','')
        if d not in date_files: date_files[d] = []
        date_files[d].append(f)

    if not date_files:
        return None, []

    latest_date = sorted(date_files.keys())[-1]
    latest_files = date_files[latest_date]

    results = []
    for f in sorted(latest_files):
        code = os.path.basename(os.path.dirname(os.path.dirname(f)))
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except:
            continue

        fd = str(data.get('final_trade_decision', ''))
        # 从Rating行精确解析信号
        rating = 'HOLD'
        for line in fd.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('**Rating**'):
                r = line_stripped.split('**Rating**')[1].strip().strip(':').strip().lower()
                if 'buy' in r or 'overweight' in r:
                    rating = 'BUY'
                elif 'sell' in r or 'underweight' in r:
                    rating = 'SELL'
                elif 'hold' in r:
                    rating = 'HOLD'
                break

        results.append({
            'code': code,
            'signal': rating,
            'rating_line': rating,
            'final_decision': fd,
            'market_report': data.get('market_report', ''),
            'fundamentals_report': data.get('fundamentals_report', ''),
            'investment_plan': data.get('investment_plan', ''),
            'investment_debate': data.get('investment_debate_state', {}),
            'risk_debate': data.get('risk_debate_state', {}),
        })

    return latest_date, results


def get_v3_signal():
    """V3策略当前信号"""
    c = duckdb.connect(DB, read_only=True)
    td = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]

    # 大盘趋势
    r = c.execute("""
        SELECT AVG(close), MAX(close) FROM (SELECT close FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20)
    """, [td.isoformat()]).fetchone()
    trend = 'BEAR' if r[1] and r[0] and r[1] < r[0] else 'BULL'

    # VIX
    vix_r = c.execute("""
        SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=?
        ORDER BY trade_date DESC LIMIT 1
    """, [td.isoformat()]).fetchone()
    vix = vix_r[0] if vix_r else 20

    # 强势行业Top3
    ind = c.execute(f"""
        SELECT industry, (MAX(CASE WHEN rn=1 THEN close END)/
                          NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1)*100 mom
        FROM (SELECT industry, close, ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
              FROM proxy_industry_daily WHERE trade_date>='{(td-timedelta(days=30)).isoformat()}') t
        WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8 ORDER BY mom DESC LIMIT 5
    """).fetchall()

    # ETF V3.3 信号 (通信+电子 MA60)
    etf_signals = {}
    for ind_name, ind_code in [('通信','801770'), ('电子','801080'), ('公用事业','801160'), ('银行','801780')]:
        r = c.execute("""
            SELECT close, ma60 FROM (
                SELECT close, AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) ma60
                FROM proxy_industry_daily WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1
            )
        """, [ind_code, td.isoformat()]).fetchone()
        if r:
            dev = (r[0]/r[1]-1)*100 if r[1] else 0
            etf_signals[ind_name] = {
                'close': r[0], 'ma60': r[1],
                'deviation': round(dev, 1),
                'above_ma': r[0] > r[1]
            }

    c.close()

    return {
        'trade_date': str(td),
        'trend': trend,
        'vix': vix,
        'top_industries': [(r[0], round(r[1], 1)) for r in ind],
        'etf_signals': etf_signals
    }


def generate_report(macro: MacroVerdict, v3_signal: dict, agent_date: str, agent_results: list):
    """生成完整Markdown日报"""
    td = v3_signal['trade_date']
    lines = []

    # ═══ 抬头 ═══
    lines.append(f'# AgentQuant 完整日报 | {td}')
    lines.append('')
    lines.append(f'> 生成时间: {date.today().isoformat()}')
    lines.append('')

    # ═══ 一、宏观裁决 ═══
    lines.append('---')
    lines.append('')
    lines.append('## 一、宏观裁决')
    lines.append('')
    gate_map = {'ATTACK': '🟢 进攻', 'CAUTION': '🟡 谨慎', 'DEFENSE': '🔴 防御', 'CRISIS': '⛔ 危机'}
    lines.append(f'| 项目 | 裁决 |')
    lines.append(f'|------|------|')
    lines.append(f'| 风控门禁 | **{gate_map.get(macro.risk_gate, macro.risk_gate)}** |')
    lines.append(f'| 仓位上限 | **{macro.position_cap:.0%}** |')
    lines.append(f'| 综合评分 | **{macro.composite}/100** |')
    lines.append('')

    lines.append('### 五维评分明细')
    lines.append('')
    lines.append(f'| 维度 | 评分 | 权重 |')
    lines.append(f'|------|------|------|')
    lines.append(f'| WTI油价 | {macro.wti_score}/100 | 20% |')
    lines.append(f'| 美10Y利率 | {macro.us10y_score}/100 | 20% |')
    lines.append(f'| 中国市场状态 | {macro.market_score}/100 | 25% |')
    lines.append(f'| 资金流 | {macro.flow_score}/100 | 15% |')
    lines.append(f'| 事件风险 | {macro.event_score}/100 | 20% |')
    lines.append('')

    lines.append('### 推理链')
    lines.append('')
    for r in macro.reasons:
        lines.append(f'- {r}')
    lines.append('')

    lines.append(f'**纠错线**: {macro.correction_line}')
    lines.append('')

    # ═══ 二、V3策略信号 ═══
    lines.append('---')
    lines.append('')
    lines.append('## 二、V3策略信号')
    lines.append('')
    lines.append(f'| 指标 | 值 |')
    lines.append(f'|------|-----|')
    lines.append(f'| 大盘趋势 | {v3_signal["trend"]} |')
    lines.append(f'| VIX | {v3_signal["vix"]:.1f} |')
    lines.append('')

    lines.append('### 强势行业Top5')
    lines.append('')
    for name, mom in v3_signal.get('top_industries', []):
        lines.append(f'- **{name}**: 10日动量 {mom:+.1f}%')
    lines.append('')

    lines.append('### ETF MA60乖离')
    lines.append('')
    etf = v3_signal.get('etf_signals', {})
    if etf:
        lines.append(f'| 行业 | 收盘 | MA60 | 乖离 | 状态 |')
        lines.append(f'|------|------|------|------|------|')
        for name, sig in etf.items():
            status = '✅ 线上' if sig['above_ma'] else '❌ 线下'
            lines.append(f'| {name} | {sig["close"]:.0f} | {sig["ma60"]:.0f} | {sig["deviation"]:+.1f}% | {status} |')
    lines.append('')

    # ═══ 三、Agent多空辩论 ═══
    lines.append('---')
    lines.append('')
    lines.append('## 三、16Agent多空辩论')
    lines.append('')

    if agent_results:
        buys = [r for r in agent_results if r['signal'] == 'BUY']
        holds = [r for r in agent_results if r['signal'] == 'HOLD']
        sells = [r for r in agent_results if r['signal'] == 'SELL']

        lines.append(f'| 信号 | 数量 | 标的 |')
        lines.append(f'|------|------|------|')
        lines.append(f'| ✅ BUY | {len(buys)} | {", ".join([r["code"] for r in buys]) or "无"} |')
        lines.append(f'| ⏸️ HOLD | {len(holds)} | {", ".join([r["code"] for r in holds]) or "无"} |')
        lines.append(f'| ❌ SELL | {len(sells)} | {", ".join([r["code"] for r in sells]) or "无"} |')
        lines.append('')

        # 逐票详情
        lines.append('### 逐票推理摘要')
        lines.append('')

        for r in sorted(agent_results, key=lambda x: {'BUY': 0, 'HOLD': 1, 'SELL': 2}.get(x['signal'], 3)):
            sig_emoji = {'BUY': '✅', 'HOLD': '⏸️', 'SELL': '❌'}.get(r['signal'], '❓')
            lines.append(f'#### {sig_emoji} {r["code"]} → {r["signal"]}')
            lines.append('')

            # 提取Rating和核心摘要
            fd = r.get('final_decision', '')
            summary = ''
            for line_text in fd.split('\n'):
                ls = line_text.strip()
                if ls.startswith('**Executive Summary**') or ls.startswith('**Rating**'):
                    summary = ls.replace('**', '')
                    break
            if not summary:
                # 取前200字符
                summary = fd[:200].replace('\n', ' ')

            lines.append(f'> {summary}')
            lines.append('')

            # 多空辩论裁决
            debate = r.get('investment_debate', {})
            jd = str(debate.get('judge_decision', ''))
            if jd:
                # 提取关键句
                key_lines = []
                for dl in jd.split('\n'):
                    if any(kw in dl for kw in ['综合判断', '核心矛盾', '多头最强', '空头最强', '最强论据']):
                        key_lines.append(dl.strip())
                if key_lines:
                    for kl in key_lines[:5]:
                        lines.append(f'- {kl}')
                    lines.append('')

            # 风控裁决
            risk = r.get('risk_debate', {})
            rj = str(risk.get('judge_decision', ''))
            if rj:
                for rl in rj.split('\n'):
                    if '最终建议' in rl or '仓位' in rl or '止损' in rl:
                        lines.append(f'- {rl.strip()}')
                lines.append('')

    else:
        lines.append('_无Agent数据 (需先运行 agent_v3_batch.py)_')
        lines.append('')

    # ═══ 四、交叉验证 ═══
    lines.append('---')
    lines.append('')
    lines.append('## 四、交叉验证')
    lines.append('')

    # 宏观 vs V3 一致性
    macro_bullish = macro.composite >= 50
    v3_bullish = v3_signal['trend'] == 'BULL'

    lines.append(f'| 验证项 | 结果 |')
    lines.append(f'|--------|------|')
    if macro_bullish == v3_bullish:
        lines.append(f'| 宏观 vs V3趋势 | ✅ 一致 ({macro.composite}/100 vs {v3_signal["trend"]}) |')
    else:
        lines.append(f'| 宏观 vs V3趋势 | ⚠️ 分歧 (宏观{macro.composite}/100 vs V3 {v3_signal["trend"]}) |')

    # Agent一致性
    if agent_results:
        buy_count = len([r for r in agent_results if r['signal'] == 'BUY'])
        sell_count = len([r for r in agent_results if r['signal'] == 'SELL'])
        if sell_count > buy_count * 2:
            lines.append(f'| Agent一致性 | ⚠️ 偏空 ({buy_count}买 vs {sell_count}卖) |')
        elif buy_count > sell_count * 2:
            lines.append(f'| Agent一致性 | ✅ 偏多 ({buy_count}买 vs {sell_count}卖) |')
        else:
            lines.append(f'| Agent一致性 | ⚖️ 分歧 ({buy_count}买 vs {sell_count}卖) |')

    # 风险叠加检查
    risk_count = 0
    if macro.risk_gate in ('DEFENSE', 'CRISIS'): risk_count += 2
    if macro.risk_gate == 'CAUTION': risk_count += 1
    if v3_signal['vix'] > 25: risk_count += 1
    if agent_results:
        buy_pct = len([r for r in agent_results if r['signal'] == 'BUY']) / len(agent_results)
        if buy_pct < 0.1: risk_count += 1.5  # 几乎全灭
        elif buy_pct < 0.3: risk_count += 0.5

    risk_label = '🔴 高' if risk_count >= 2.5 else '🟡 中' if risk_count >= 1 else '🟢 低'
    lines.append(f'| 风险叠加 | {risk_label} ({risk_count:.1f}) |')
    lines.append('')

    # ═══ 五、三段论裁决 ═══
    lines.append('---')
    lines.append('')
    lines.append('## 五、三段论裁决')
    lines.append('')

    # 大前提
    lines.append('### 大前提 (市场+规则)')
    lines.append('')
    gate_action = {
        'ATTACK': '宏观门禁开放，允许进攻配置',
        'CAUTION': '宏观门禁谨慎，仓位上限50%，减配进攻',
        'DEFENSE': '宏观门禁防御，仓位上限30%，冻结进攻端',
        'CRISIS': '宏观门禁危机，仓位上限10%，全仓防御'
    }
    lines.append(f'- {gate_action.get(macro.risk_gate, "")}')
    lines.append(f'- 沪深300 {v3_signal["trend"]} 趋势，VIX={v3_signal["vix"]:.1f}')
    if v3_signal.get('etf_signals', {}).get('公用事业', {}).get('above_ma'):
        lines.append(f'- 公用事业ETF > MA60 → 防守端正常')
    else:
        lines.append(f'- 公用事业ETF < MA60 → 防守端触发现金替代')
    lines.append('')

    # 小前提
    lines.append('### 小前提 (标的维度)')
    lines.append('')

    if agent_results:
        for r in agent_results:
            if r['signal'] == 'BUY':
                lines.append(f'- **{r["code"]}**: Agent建议买入')
        buy_codes = [r['code'] for r in agent_results if r['signal'] == 'BUY']
        if not buy_codes:
            lines.append('- 进攻池Agent全灭 → 0只BUY')
            lines.append('- 防守池银行股有结构性机会但Agent分歧大')
    else:
        lines.append('- Agent数据缺失，无法逐票验证')

    etf = v3_signal.get('etf_signals', {})
    for name in ['通信', '电子']:
        if name in etf:
            lines.append(f'- {name}ETF: 乖离{etf[name]["deviation"]:+.1f}% → {"趋势完好" if etf[name]["above_ma"] else "破位"}')
    lines.append('')

    # 结论
    lines.append('### 结论 (操作+点位+概率+纠错)')
    lines.append('')

    # 根据信号综合判断
    agent_buy_count = len([r for r in agent_results if r['signal'] == 'BUY']) if agent_results else 0
    agent_total = len(agent_results) if agent_results else 0
    agent_buy_ratio = agent_buy_count / agent_total if agent_total > 0 else 0
    agent_sell_ratio = len([r for r in agent_results if r['signal'] == 'SELL']) / agent_total if agent_total > 0 else 0

    if macro.risk_gate == 'CRISIS':
        action = '**清仓观望**'
        detail = '宏观危机模式，不持有任何风险资产'
        prob = '踏空概率10% vs 亏钱概率60%'
        correction = f'WTI回落至$70以下+美10Y破4.0% → 可解除危机模式'
    elif macro.risk_gate == 'DEFENSE' or agent_sell_ratio > 0.5:
        action = '**防御观望**'
        sell_count = agent_total - agent_buy_count - len([r for r in agent_results if r['signal'] == 'HOLD'])
        detail = f'Agent偏空({agent_buy_count}买/{sell_count}卖) → 不新开仓，现有防守端观察'
        prob = f'踏空概率15% vs 亏钱概率35%'
        correction = f'FOMC落地+公用事业收复MA60+Agent BUY>3只 → 可升级至谨慎试探'
    elif macro.risk_gate == 'CAUTION' and agent_buy_ratio < 0.3:
        action = '**防御观望**'
        detail = f'宏观谨慎+Agent偏空({agent_buy_count}买/{agent_total}只) → 保守为上'
        prob = '赚钱概率25% vs 亏钱概率30%'
        correction = f'FOMC落地+公用事业MA60收复+宏观评分>60 → 可考虑半仓防守'
    elif macro.risk_gate == 'CAUTION' and agent_buy_count >= 2:
        action = '**谨慎试探**'
        detail = f'不超过{macro.position_cap:.0%}仓位，优先Agent BUY标的'
        prob = f'赚钱概率40% vs 亏钱概率30%'
        buy_list = ', '.join([r['code'] for r in agent_results if r['signal'] == 'BUY'])
        correction = f'买入标的破MA60或FOMC鹰派 → 减至30%仓位'
    else:
        action = '**正常进攻**'
        detail = f'进攻ETF(通信+电子)+防守ETF(公用事业)哑铃配置'
        prob = '赚钱概率50% vs 亏钱概率25%'
        correction = '沪深300<MA20 → 封印进攻端'

    lines.append(f'| 项目 | 内容 |')
    lines.append(f'|------|------|')
    lines.append(f'| 操作 | {action} |')
    lines.append(f'| 详情 | {detail} |')
    lines.append(f'| 仓位上限 | {macro.position_cap:.0%} |')
    lines.append(f'| 概率分布 | {prob} |')
    lines.append(f'| 纠错线 | {correction} |')

    lines.append('')
    lines.append('---')
    lines.append(f'> AgentQuant | Macro + V3 + 16Agent | {td}')
    lines.append('')

    return '\n'.join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true', help='跳过Agent数据, 仅宏观+V3')
    p.add_argument('--output', type=str, default=None, help='输出路径')
    args = p.parse_args()

    # 1. 宏观裁决
    print('Running macro analysis...')
    macro = macro_run()

    # 2. V3策略信号
    print('Running V3 signal...')
    v3 = get_v3_signal()

    # 3. Agent数据
    agent_date = None
    agent_results = []
    if not args.quick:
        print('Loading agent results...')
        agent_date, agent_results = load_agent_results()
        if agent_results:
            print(f'  Loaded {len(agent_results)} agent verdicts from {agent_date}')
        else:
            print('  No agent data found (run agent_v3_batch.py first)')
    else:
        print('  Quick mode: skipping agent data')

    # 4. 生成报告
    print('Generating report...')
    report = generate_report(macro, v3, agent_date, agent_results)

    # 5. 输出
    out_path = args.output or f'D:/AgentQuant/our/agentquant_daily_{date.today().strftime("%Y%m%d")}.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'\nReport saved: {out_path}')
    print(f'  Gate: {macro.risk_gate} | Cap: {macro.position_cap:.0%} | Score: {macro.composite}/100')

    if agent_results:
        buys = [r for r in agent_results if r['signal'] == 'BUY']
        print(f'  Agent: {len(buys)} BUY / {len([r for r in agent_results if r["signal"]=="HOLD"])} HOLD / {len([r for r in agent_results if r["signal"]=="SELL"])} SELL')

    return report


if __name__ == '__main__':
    main()
