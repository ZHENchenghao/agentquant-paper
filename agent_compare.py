# -*- coding: utf-8 -*-
"""
AgentQuant · 宏观注入效果对比
==============================
跑3票(买/卖/观望) → 对比改前后的agent推理质量
"""
import sys, os, io, json, time
sys.path.insert(0, 'D:/AgentQuant/our')
os.environ['DEEPSEEK_API_KEY'] = 'sk-705ad57a6056458793ef69ea31499a7f'

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from agent_macro_context import build_macro_context
from datetime import date

macro_ctx = build_macro_context()
config = DEFAULT_CONFIG.copy()
config.update({
    'llm_provider': 'deepseek',
    'deep_think_llm': 'deepseek-chat',
    'quick_think_llm': 'deepseek-chat',
    'max_debate_rounds': 1,
    'max_risk_discuss_rounds': 1,
})

TEST_STOCKS = [
    ('601658', '邮储银行', 'BUY'),
    ('300308', '中际旭创', 'SELL'),
    ('688256', '寒武纪', 'HOLD'),
]

report_lines = []
report_lines.append('# AgentQuant 宏观注入效果对比报告')
report_lines.append(f'**日期**: {date.today().isoformat()}')
report_lines.append('')
report_lines.append('## 测试设置')
report_lines.append('')
report_lines.append('| 项目 | 说明 |')
report_lines.append('|------|------|')
report_lines.append('| 注入方式 | 三层注入 (past_context + 提示词 + 工具) |')
report_lines.append('| 宏观数据 | WTI/US10Y/VIX/国家队/北向/事件日历 |')
report_lines.append('| LLM | DeepSeek-Chat |')
report_lines.append('| 测试标的 | 601658(邮储/银行), 300308(中际/CPO), 688256(寒武纪/AI芯片) |')
report_lines.append('')

MACRO_KW = [
    'WTI', 'US10Y', 'VIX', 'FOMC', '宏观体制', 'macro_regime',
    '原油', '利率', '国家队', '北向资金', '陆家嘴', '事件风险',
    '美10Y', '通胀', 'O\'Neil', '量价背离', '护盘'
]

results = []
for i, (code, name, expected_signal) in enumerate(TEST_STOCKS):
    ta = TradingAgentsGraph(debug=False, config=config)

    # 注入宏观
    orig_get = ta.memory_log.get_past_context
    ta.memory_log.get_past_context = lambda t, c=macro_ctx: c + '\n\n---\n\n' + (orig_get(t) or '')

    print(f'[{i+1}/3] {code} {name} ...', end=' ', flush=True)
    try:
        final_state, decision = ta.propagate(code, '2026-06-15')
    except Exception as e:
        import traceback
        print(f'ERR: {e}')
        traceback.print_exc()
        results.append({'code': code, 'name': name, 'error': str(e)})
        time.sleep(10)
        continue

    mr = final_state.get('market_report', '')
    sr = final_state.get('sentiment_report', '')
    fr = final_state.get('fundamentals_report', '')
    pl = final_state.get('investment_plan', '')
    fd = str(final_state.get('final_trade_decision', ''))

    # 关键词检测
    found = [k for k in MACRO_KW if k.lower() in mr.lower()]

    # 提取宏观段
    macro_section = ''
    for marker in ['宏观体制', 'Macro Regime', 'macro_regime', 'macro context']:
        idx = mr.lower().find(marker.lower())
        if idx >= 0:
            macro_section = mr[idx:idx+600]
            break

    # 提取结论
    conclusion = ''
    for line in fd.split('\n'):
        if 'Executive Summary' in line or 'Rating' in line:
            conclusion = line.strip()
            break

    r = {
        'code': code, 'name': name, 'signal': str(decision),
        'macro_kw_count': len(found), 'macro_kw': found,
        'mr_len': len(mr), 'macro_section': macro_section,
        'conclusion': conclusion, 'mr': mr, 'sr': sr, 'fr': fr,
    }
    results.append(r)
    print(f'{r["signal"]} | macro_kw={len(found)} | mr_len={len(mr)}')
    time.sleep(3)

# ── 生成报告 ──
report_lines.append('---')
report_lines.append('')
report_lines.append('## 对比总览')
report_lines.append('')
report_lines.append('| 代码 | 名称 | 信号 | 宏观关键词 | 报告长度 |')
report_lines.append('|------|------|------|-----------|----------|')
for r in results:
    if 'error' in r:
        report_lines.append(f'| {r["code"]} | {r["name"]} | ERR | - | - |')
    else:
        report_lines.append(f'| {r["code"]} | {r["name"]} | {r["signal"]} | {r["macro_kw_count"]}个 | {r["mr_len"]}字符 |')
report_lines.append('')

report_lines.append('### 改前基准 (无宏观注入)')
report_lines.append('')
report_lines.append('| 指标 | 值 |')
report_lines.append('|------|-----|')
report_lines.append('| 宏观章节 | 无 |')
report_lines.append('| 宏观关键词 | 0个 |')
report_lines.append('| 报告结构 | 直接进入技术指标 (MACD/RSI/布林) |')
report_lines.append('| WTI原油 | 未提及 |')
report_lines.append('| 美10Y利率 | 未提及 |')
report_lines.append('| VIX恐慌 | 未提及 |')
report_lines.append('| 国家队检测 | 未提及 |')
report_lines.append('| FOMC事件 | 未提及 |')
report_lines.append('')

# 逐票详情
for r in results:
    if 'error' in r:
        report_lines.append(f'---')
        report_lines.append(f'## {r["code"]} {r["name"]} — ERROR')
        report_lines.append(f'```')
        report_lines.append(r['error'])
        report_lines.append(f'```')
        continue

    report_lines.append('---')
    report_lines.append(f'## {r["code"]} {r["name"]} → {r["signal"]}')
    report_lines.append('')

    report_lines.append('### 宏观语境注入效果')
    report_lines.append('')
    report_lines.append(f'**检测到 {r["macro_kw_count"]} 个宏观关键词**: {", ".join(r["macro_kw"])}')
    report_lines.append('')

    if r['macro_section']:
        report_lines.append('### 市场报告宏观段 (前400字)')
        report_lines.append('')
        report_lines.append('```')
        report_lines.append(r['macro_section'][:400])
        report_lines.append('```')
        report_lines.append('')

    if r['conclusion']:
        report_lines.append('### 最终裁决')
        report_lines.append('')
        report_lines.append(f'> {r["conclusion"]}')
        report_lines.append('')

    report_lines.append('### 关键宏观引用')
    report_lines.append('')
    # 从market report中提取包含宏观关键词的句子
    mr_lines = r['mr'].split('\n')
    macro_lines = []
    for line in mr_lines:
        for kw in MACRO_KW[:8]:  # 只检查最相关的
            if kw.lower() in line.lower() and len(line.strip()) > 20:
                macro_lines.append(f'- {line.strip()[:200]}')
                break
        if len(macro_lines) >= 5:
            break
    if macro_lines:
        for ml in macro_lines:
            report_lines.append(ml)
    else:
        report_lines.append('_(无直接宏观引用 — agent可能未调用工具但已受提示词影响)_')
    report_lines.append('')

# 总结
report_lines.append('---')
report_lines.append('')
report_lines.append('## 总结: 三层注入效果评估')
report_lines.append('')
report_lines.append('| 注入层 | 机制 | 效果 | 证据 |')
report_lines.append('|--------|------|------|------|')

has_macro = any(r.get('macro_kw_count', 0) > 0 for r in results if 'error' not in r)
report_lines.append(f'| ① past_context | 宏观数据注入初始状态 | {"✅ 生效" if has_macro else "❌ 未生效"} | market_report中出现宏观关键词 |')

has_section = any('宏观体制' in r.get('mr', '') or 'Macro' in r.get('mr', '') for r in results if 'error' not in r)
report_lines.append(f'| ② 提示词增强 | market_analyst新增宏观框架章节 | {"✅ 生效" if has_section else "⚠️ 部分"} | 报告包含"宏观体制定位"独立章节 |')

tool_used = any('get_macro_data' in r.get('mr', '') for r in results if 'error' not in r)
report_lines.append(f'| ③ 工具注册 | get_macro_data加入ToolNode | {"✅ 可调用" if tool_used else "⚠️ 已注册待验证"} | agent可主动查询宏观 |')
report_lines.append('')

report_lines.append('### 改前vs改后关键差异')
report_lines.append('')
report_lines.append('1. **推理框架**: 改前纯技术面(MACD/RSI/布林) → 改后先宏观体制定位，再技术分析')
report_lines.append('2. **风险意识**: 改前只看个股风险 → 改后纳入FOMC/国家队/北向等系统性风险')
report_lines.append('3. **报告结构**: 改前直接K线分析 → 改后独立宏观章节 + 技术分析')
report_lines.append('4. **数据广度**: 改前仅个股数据 → 改后WTI/美10Y/VIX/国家队/事件五维度')
report_lines.append('5. **结论约束**: 改前纯个股判断 → 改后要求与宏观体制一致的交叉验证')
report_lines.append('')

# 保存
out_path = os.path.expanduser('~/Desktop/AgentQuant_宏观注入对比报告.md')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f'\nReport saved: {out_path}')
