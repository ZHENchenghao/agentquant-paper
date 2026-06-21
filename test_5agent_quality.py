# -*- coding: utf-8 -*-
"""
AgentQuant · TDD测试: 5 Agent推理质量
======================================
对比: 5 Agent (新) vs 16 Agent (旧)
验证: 穿透判断 / 数据引用 / 持续性标签 / 事实推演分离 / 铁律遵守
"""
import sys, os, json, time, io
sys.path.insert(0, 'D:/AgentQuant/our')
os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY_PLACEHOLDER')

from datetime import date
from langchain_core.messages import HumanMessage
from tradingagents.dataflows.config import set_config

# 测试配置
TEST_STOCKS = [
    ('601658', '邮储银行', '防守池标的, 应偏谨慎'),
    ('688256', '寒武纪', '高PE成长股, 应侧重排雷+穿透'),
    ('600600', '青岛啤酒', '纸交持仓, 应出方向判断'),
]

QUALITY_CHECKS = {
    'penetration': ['定价', '错杀', '预期差', '穿透'],   # 穿透判断
    'data_citation': ['29次', '样本', '次)','数据显示'],  # 数据引用
    'duration_label': ['1-3天','1-4周','持续性'],         # 持续性标签
    'fact_vs_guess': ['事实','推演','猜测','置信度'],     # 事实/推演分离
    'iron_law': ['纠错线','止损','反向操作'],             # 纠错线
    'validated_rules': ['WTI','北向','OCF','商誉','排雷'],# 引用回测规律
    'direction': ['进攻','谨慎试探','防御观望','清仓','减仓'], # 明确方向
    'no_hallucination': ['数据缺失','NULL','不可用'],      # 诚实标注(如果有缺失)
}

def check_quality(text, checks=QUALITY_CHECKS):
    """检查文本是否包含各质量维度关键词"""
    scores = {}
    for dim, keywords in checks.items():
        hits = [kw for kw in keywords if kw in text]
        scores[dim] = len(hits)
    return scores


def run_5agent_test():
    """跑5 Agent, 检验质量"""
    from agentquant_graph import create_agentquant_graph
    from agent_macro_context import build_macro_context

    set_config({'llm_provider': 'deepseek', 'deep_think_llm': 'deepseek-chat', 'quick_think_llm': 'deepseek-chat'})
    graph = create_agentquant_graph()
    macro_ctx = build_macro_context()
    td = str(date.today())

    results = []
    total_tokens = 0

    for i, (code, name, expected) in enumerate(TEST_STOCKS):
        print(f'\n[Test {i+1}/3] {code} {name} — {expected}')
        print('-' * 60)

        t0 = time.time()
        state = {
            "messages": [HumanMessage(content=code)],
            "company_of_interest": code,
            "trade_date": td,
            "past_context": macro_ctx,
            "macro_report": "", "news_report": "", "deep_report": "",
        }

        try:
            result = graph.invoke(state, {"recursion_limit": 50})
            elapsed = time.time() - t0
        except Exception as e:
            print(f'  FAIL: {e}')
            results.append({'code': code, 'status': 'ERR', 'error': str(e)})
            continue

        fd = result.get("final_trade_decision", "")
        macro_r = result.get("macro_report", "")
        deep_r = result.get("deep_report", "")
        all_text = fd + macro_r + deep_r

        # 质量评分
        scores = check_quality(all_text)
        total_score = sum(scores.values())

        print(f'  耗时: {elapsed:.0f}s  文本: {len(all_text)}字  质量分: {total_score}')
        for dim, hits in scores.items():
            icon = 'OK' if hits > 0 else '--'
            sys.stdout.write(f'    {icon} {dim}: {hits} hits\n')

        # 提取裁决
        verdict = ''
        for kw in ['进攻','谨慎试探','防御观望','清仓','减仓']:
            if kw in fd:
                verdict = kw
                break
        v_display = verdict or '未检测到'
        print(f'  裁决: {v_display}')

        results.append({
            'code': code, 'name': name, 'status': 'OK',
            'elapsed': round(elapsed, 0), 'text_len': len(all_text),
            'quality_score': total_score, 'quality_detail': scores,
            'verdict': verdict, 'verdict_text': fd[:300],
        })

    return results


def benchmark_16agent():
    """跑16 Agent (旧) 做对比基准"""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG
    from agent_macro_context import build_macro_context

    config = DEFAULT_CONFIG.copy()
    config.update({'llm_provider': 'deepseek', 'deep_think_llm': 'deepseek-chat',
                   'quick_think_llm': 'deepseek-chat', 'max_debate_rounds': 1, 'max_risk_discuss_rounds': 1})

    macro_ctx = build_macro_context()
    results = []

    for code, name, _ in TEST_STOCKS[:1]:  # 只跑1只做对比(省钱)
        ta = TradingAgentsGraph(debug=False, config=config)
        # 注入宏观
        orig = ta.memory_log.get_past_context
        def patched(t, ctx=macro_ctx, o=orig):
            old = o(t) if callable(o) else ''
            return ctx + '\n\n---\n\n' + str(old) if old else ctx
        ta.memory_log.get_past_context = patched

        t0 = time.time()
        try:
            fs, decision = ta.propagate(code, str(date.today()))
            elapsed = time.time() - t0
        except Exception as e:
            results.append({'code': code, 'error': str(e), 'elapsed': 0})
            continue

        fd = str(decision) if decision else ''
        all_text = fd + fs.get('market_report','') + fs.get('fundamentals_report','')
        scores = check_quality(all_text)

        results.append({
            'code': code, 'elapsed': round(elapsed, 0),
            'text_len': len(all_text), 'quality_score': sum(scores.values()),
            'scores': scores,
        })
        break

    return results


def print_report(results_5, results_16):
    """打印对比报告"""
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print('\n' + '=' * 60)
    print('  AgentQuant TDD 质量对比报告')
    print('=' * 60)

    # 汇总
    pass_count = sum(1 for r in results_5 if r['status'] == 'OK')
    avg_score = sum(r['quality_score'] for r in results_5 if r['status'] == 'OK') / max(pass_count, 1)
    avg_time = sum(r['elapsed'] for r in results_5 if r['status'] == 'OK') / max(pass_count, 1)
    avg_len = sum(r['text_len'] for r in results_5 if r['status'] == 'OK') / max(pass_count, 1)

    print(f'\n5 Agent 结果: {pass_count}/{len(results_5)}通过')
    print(f'  平均质量分: {avg_score:.0f}')
    print(f'  平均耗时: {avg_time:.0f}s')
    print(f'  平均文本: {avg_len:.0f}字')

    if results_16:
        r16 = results_16[0]
        print(f'\n16 Agent 基准:')
        print(f'  质量分: {r16["quality_score"]}')
        print(f'  耗时: {r16["elapsed"]}s')
        print(f'  文本: {r16["text_len"]}字')
        if avg_time > 0:
            print(f'  加速: {r16["elapsed"]/avg_time:.1f}x')

    # 质量维度雷达
    print('\n质量维度覆盖 (0=无, >0=有):')
    dims = list(QUALITY_CHECKS.keys())
    for r in results_5:
        if r['status'] != 'OK': continue
        print(f'\n  {r["code"]} {r["name"]}:')
        for dim in dims:
            score = r.get('quality_detail', {}).get(dim, 0)
            bar = '█' * score if score > 0 else '—'
            print(f'    {dim:<20} {bar}')

    # 判定
    print('\n' + '=' * 60)
    if pass_count == len(results_5) and avg_score >= 20:
        print('  ✅ 通过: 5 Agent质量不退化, 建议切换')
    elif pass_count == len(results_5):
        print('  ⚠️ 部分通过: 建议补强后再切换')
    else:
        print('  ❌ 未通过: 需要修复')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', action='store_true', help='跑16 Agent基准对比(费token)')
    args = p.parse_args()

    print('AgentQuant TDD — 5 Agent 质量检验')
    print(f'测试标的: {len(TEST_STOCKS)}只')

    results_5 = run_5agent_test()
    results_16 = benchmark_16agent() if args.benchmark else []

    print_report(results_5, results_16)
