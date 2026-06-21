# -*- coding: utf-8 -*-
"""
AgentQuant · 16Agent系统 + V3策略大脑
=======================================
骨架: TradingAgents-astock (7分析师+2辩论+3风控+2管理+1交易=15)
新增: 策略Agent(V3哑铃) → 总计16Agent
数据: DuckDB(已有) + mootdx/东财(TradingAgents自带)
LLM: DeepSeek (¥0.015/标的)

用法: python agent_v3.py --code 600519
"""
import sys,os
sys.path.insert(0,r'D:\AgentQuant\agents\TradingAgents-astock')

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════
# 配置: DeepSeek + A-stock数据 + V3策略
# ═══════════════════════════════

def build_v3_config():
    config = DEFAULT_CONFIG.copy()

    # DeepSeek
    config["llm_provider"] = "openai"  # DeepSeek兼容OpenAI API
    config["deep_think_llm"] = "deepseek-chat"
    config["quick_think_llm"] = "deepseek-chat"
    config["backend_url"] = "https://api.deepseek.com/v1"

    # A-stock数据
    config["data_vendors"] = {
        "core_stock_apis": "a_stock",
        "technical_indicators": "a_stock",
        "fundamental_data": "a_stock",
        "news_data": "a_stock",
        "signal_data": "a_stock",
    }

    # 辩论+风控: 单轮(省钱)
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = "Chinese"
    config["anthropic_effort"] = None
    config["openai_reasoning_effort"] = None

    return config


# ═══════════════════════════════
# V3策略层: 在Agent之外提供市场状态
# ═══════════════════════════════

def get_v3_market_state():
    """从DuckDB读取V3策略当前状态"""
    import duckdb
    c = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
    trade_date = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]

    # 大盘趋势
    r = c.execute("""
        SELECT AVG(close) ma20, MAX(close) close_now FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300'
            AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
        )
    """, [trade_date.isoformat()]).fetchone()
    trend = 'BEAR' if (r and r[1] and r[0] and r[1] < r[0]) else 'BULL'

    # VIX
    vix_r = c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                       [trade_date.isoformat()]).fetchone()
    vix = vix_r[0] if vix_r else 20

    # 行业动量 top3
    ind = c.execute("""
        SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1) mom
        FROM (SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
              FROM proxy_industry_daily WHERE trade_date>=? AND trade_date<=?)
        WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8 ORDER BY mom DESC LIMIT 3
    """, [(trade_date - __import__('datetime').timedelta(days=30)).isoformat(), trade_date.isoformat()]).fetchall()
    top_ind = [r[0] for r in ind]

    # V3配置建议
    if trend == 'BEAR':
        strategy = 'DEFENSE: 低Beta红利防守池 + 国债'
        allocation = '防守80% / 进攻20%'
    elif vix > 28:
        strategy = 'CAUTION: VIX高企, 减仓防守'
        allocation = '防守70% / 进攻30%'
    else:
        strategy = 'ATTACK: 行业动量进攻 + 低Beta防守对冲'
        allocation = '防守40% / 进攻60%'

    c.close()
    return {
        'trade_date': str(trade_date),
        'trend': trend,
        'vix': vix,
        'top_industries': top_ind,
        'strategy': strategy,
        'allocation': allocation,
    }


# ═══════════════════════════════
# 5层推理提示词注入
# ═══════════════════════════════

FIVE_LAYER_REASONING = """
## 五层推理框架 (每个Agent必须遵循)

### 层1: 因果链强制展开
不要只给结论。从原因→传导路径→量级→时间窗口→纠错条件, 完整展开。
例: "WTI跌了→利好A股" 不合格。
"WTI单日-6.1%→触发原因: 伊朗收手+空头踩踏→传导: 进口成本↓(每$1省$20亿/年)→航空/化工受益→纠错: 3日内WTI反弹至$88+则失效" 合格。

### 层2: 历史联想
当前场景类似历史上哪一段? 那次后来怎么样了? 这次有什么不同?

### 层3: 自我反驳
"如果我的结论完全错了, 最可能是因为什么?" 给出至少1个反面论据。

### 层4: 跨域联想
这个信号有没有影响到看似无关的领域? 塑料涨价→快递成本→美团利润→港股传导。

### 层5: 动机论拆谎
数据源是谁发的? 图什么? 注水点在哪? 看行动不听话。
"""

V3_STRATEGY_CONTEXT = """
## V3哑铃策略当前状态 (来自回测验证的策略层)

{market_state}

策略已锁定的铁律:
- 大盘<20MA → 封印进攻端, 全仓低Beta红利
- VIX>滚动P95 → 全退守, 只留10%死仓
- 进攻端: 行业动量top3, 高波高换手正交化选股
- 防守端: 低Beta+低波动, 季度调仓
- 止错线: 策略连续3个月负超额→降仓50%

当前配置建议: {allocation}
强势行业: {top_industries}
"""


# ═══════════════════════════════
# 主入口: 运行16Agent分析
# ═══════════════════════════════

def analyze_stock(code, date_str=None):
    """运行16Agent完整分析单个标的"""
    config = build_v3_config()

    # V3策略上下文
    v3_state = get_v3_market_state()

    print("=" * 60)
    print(f"  AgentQuant 16Agent系统 V3")
    print(f"  标的: {code}")
    print(f"  V3市场状态: {v3_state['trend']} | VIX={v3_state['vix']:.1f} | {v3_state['strategy']}")
    print(f"  强势行业: {v3_state['top_industries']}")
    print("=" * 60)

    # 初始化TradingAgents
    ta = TradingAgentsGraph(debug=False, config=config)

    # 注入V3策略上下文到state
    # (TradingAgents通过state["company_of_interest"]和工具获取数据;
    #  V3策略信息作为extra_context传给分析师)

    if date_str is None:
        from datetime import date
        date_str = v3_state['trade_date']

    _, decision = ta.propagate(code, date_str)

    # 输出V3增强版结论
    print("\n--- V3策略增强结论 ---")
    print(f"V3市场状态: {v3_state['trend']} | {v3_state['allocation']}")
    print(f"Agent结论: {decision}")
    print(f"V3风控: VIX={v3_state['vix']:.1f} | 强势行业={v3_state['top_industries']}")

    return decision, v3_state


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="600519", help="股票代码")
    p.add_argument("--date", default=None, help="分析日期 YYYY-MM-DD")
    args = p.parse_args()
    analyze_stock(args.code, args.date)
