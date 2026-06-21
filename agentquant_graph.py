# -*- coding: utf-8 -*-
"""AgentQuant 5-Agent Graph v3 — real prices, distinct roles, paper state"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.agents.utils.agent_utils import (
    get_stock_data, get_indicators, get_macro_data, get_event_calendar,
    get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
    get_news, get_global_news, get_insider_transactions,
    get_profit_forecast, get_hot_stocks, get_northbound_flow,
    get_concept_blocks, get_fund_flow, get_dragon_tiger_board,
    get_lockup_expiry, get_industry_comparison,
    get_agent_knowledge, get_anti_hallucination_rules, get_language_instruction,
    build_instrument_context,
)
from tradingagents.llm_clients import create_llm_client
from tradingagents.dataflows.config import set_config


def _call_llm(llm, system_msg, state, tools=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="messages"),
    ])
    ctx = build_instrument_context(state["company_of_interest"])
    prompt = prompt.partial(current_date=state["trade_date"],
                            instrument_context=ctx,
                            past_context=state.get("past_context", ""))
    chain = prompt | llm.bind_tools(tools) if tools else prompt | llm
    return chain.invoke(state["messages"])


def _get_price_data(ticker):
    """Real K-line + financial data from DuckDB"""
    try:
        import duckdb
        c = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
        ts = ticker + '.SH' if ticker.startswith('6') else ticker + '.SZ'
        rows = c.execute("SELECT trade_date,close,change_pct,vol,turnover_rate FROM kline_daily WHERE ts_code=? ORDER BY trade_date DESC LIMIT 5", [ts]).fetchall()
        fin = c.execute("SELECT net_profit,operating_cf,revenue,accounts_receivable,roe FROM financial_statements WHERE ts_code=? AND report_type='annual' ORDER BY report_date DESC LIMIT 1", [ts]).fetchone()
        v = c.execute("SELECT pe_ttm,pb FROM valuation_daily WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1", [ts]).fetchone()
        c.close()
        lines = ["## REAL K-LINE DATA (DO NOT FABRICATE PRICES)"]
        lines.append("Latest: %s close=%.2f chg=%.2f%% turnover=%.2f%%" % (rows[0][0], rows[0][1], rows[0][2] or 0, rows[0][4] or 0))
        for r in rows:
            lines.append("  %s close=%.2f vol=%.0f" % (r[0], r[1], (r[3] or 0) / 1e4))
        if fin:
            np_v = (fin[0] or 0) / 1e8
            ocf_v = (fin[1] or 0) / 1e8
            rev_v = (fin[2] or 0) / 1e8
            lines.append("Annual: NP=%.1f OCF=%.1f Rev=%.1f ROE=%.1f%%" % (np_v, ocf_v, rev_v, fin[4] or 0))
            if np_v > 0:
                lines.append("OCF/NP=%.2f (mine-sweep: <0.3=fake profit)" % (ocf_v / np_v))
        if v:
            lines.append("PE=%.1f PB=%.2f" % (v[0] or 0, v[1] or 0))
        return '\n'.join(lines)
    except Exception as e:
        return "[Price data fetch failed: %s]" % str(e)


def _get_paper_state():
    """Paper trading portfolio state"""
    try:
        import duckdb
        with open('D:/AgentQuant/our/paper_portfolio.json', 'r', encoding='utf-8') as f:
            pf = json.load(f)
        c = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
        td = c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
        lines = ["## PAPER PORTFOLIO (REAL!)", "Cash: %.0f" % pf['cash']]
        total_pnl = 0
        total_cost = 0
        for code, pos in pf.get('positions', {}).items():
            r = c.execute("SELECT close FROM kline_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                          [code, td.isoformat()]).fetchone()
            px = r[0] if r else pos['buy_price']
            pnl = (px - pos['buy_price']) * pos['shares']
            total_pnl += pnl
            total_cost += pos['buy_price'] * pos['shares']
            lines.append("  %s %d shares cost=%.2f now=%.2f PnL=%+.0f" % (code, pos['shares'], pos['buy_price'], px, pnl))
        if total_cost > 0:
            lines.append("Total cost=%.0f PnL=%+.0f (%.1f%%)" % (total_cost, total_pnl, total_pnl / total_cost * 100))
        c.close()
        return '\n'.join(lines)
    except:
        return "[Paper portfolio: none]"


def _text_agent(llm, system, state, report_field):
    """Text agent with real prices + paper state injected"""
    ctx = state.get("past_context", "")
    ticker = state["company_of_interest"]
    price = _get_price_data(ticker)
    paper = _get_paper_state() if report_field == 'final_trade_decision' else ''
    input_text = "Ticker: %s\n\n%s\n\n%s\n\nMacro Context:\n%s" % (ticker, price, paper, ctx[:2000])
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=input_text)])
    txt = result.content if hasattr(result, 'content') else str(result)
    return {"messages": [result], report_field: txt if txt else "[%s: empty]" % report_field}


def create_agentquant_graph(llm_config=None):
    if llm_config is None:
        llm_config = {'provider': 'deepseek', 'model': 'deepseek-chat'}

    qc = create_llm_client(llm_config['provider'], llm_config['model'])
    ql = qc.get_llm()
    dc = create_llm_client(llm_config['provider'], llm_config['model'])
    dl = dc.get_llm()

    # Shared output fields for macro + verdict
    FOUR_FIELDS = """
## MANDATORY OUTPUT FIELDS (all 5 required)

### [1] PENETRATION: How much is priced in?
Format: [Signal] | pre-N-day move X% | priced in Y% | remaining Z% | duration:1-3d/1-4w

### [2] DURATION LABEL: 1-3 days (no position change) or 1-4 weeks (can adjust)

### [3] VIX FILTER
Check: current VIX > 20?
YES -> ALL signals weight * 0.5, mark "VIX downgrade applied"
NO -> mark "VIX=X.X < 20, full signal weight"

### [4] REVERSE TEST
Market consensus: "XXX" -> historical N samples: actual result YYY -> [PASS/FAIL]

### [5] ALTERNATIVE (if consensus FAILS)
Don't do [wrong thing], do [correct thing] instead -- cite validated_rules
"""

    # Agent 1: Macro Regime
    mt = [get_stock_data, get_indicators, get_macro_data, get_event_calendar]
    ms = (
        'You are the Macro Regime Analyst. Your ONLY job: judge "what game are we in".\n\n'
        'Do NOT do technical analysis. Do NOT evaluate individual stocks. Answer only:\n'
        '1. Master Regime: Who dominates? (Fed/WhiteHouse/WallStreet/NationalTeam) Who wins the power struggle?\n'
        '2. Position Ceiling: Max N% (cite VIX + FOMC + breadth + volume + northbound)\n'
        '3. Penetration: For each macro signal, how much is already priced in?\n'
        '4. VIX Filter: Current VIX? Trigger downgrade?\n\n'
        + FOUR_FIELDS + '\n'
        + get_agent_knowledge('market') + '\n'
        + get_anti_hallucination_rules()
    )

    # Agent 2: News Skeptic
    nt = [get_news, get_global_news, get_macro_data]
    ns = (
        'You are the News Skeptic. Your ONLY job: expose false narratives.\n\n'
        'Do NOT do penetration analysis. Do NOT do VIX filtering. Do NOT do technical analysis. Only:\n'
        '- For each narrative: Who published? Angle? Padding? Numbers check out? Already priced? Historical data?\n'
        '- If a narrative is PROVEN FALSE by data (validated_rules): MUST provide ALTERNATIVE action.\n\n'
        + get_agent_knowledge('news') + '\n'
        + get_anti_hallucination_rules()
    )

    # Agent 3: Deep Analyst
    dt = [get_stock_data, get_indicators, get_fundamentals, get_balance_sheet,
          get_cashflow, get_income_statement, get_profit_forecast, get_industry_comparison,
          get_hot_stocks, get_northbound_flow, get_fund_flow, get_dragon_tiger_board]
    ds = (
        'You are the Deep Analyst. Your ONLY job: stock-level deep dive.\n\n'
        'Use the REAL K-line data provided above. Do NOT make up prices.\n'
        'Do NOT do macro regime (already done). Do NOT do VIX filter (already done).\n\n'
        'Output:\n'
        '1. Technical: position + volume + supports/resistances (use REAL prices!)\n'
        '2. Mine-sweep: OCF quality / AR buildup / goodwill (cite validated_rules)\n'
        '3. Capital flow: northbound / hot money / institutional\n'
        '4. Bull vs Bear debate (argue BOTH sides yourself)\n'
        '5. Conclusion: direction + confidence + duration label (1-3d or 1-4w)\n\n'
        + get_agent_knowledge('fundamentals') + '\n'
        + get_agent_knowledge('bull') + '\n'
        + get_anti_hallucination_rules()
    )

    # Agent 4: Verdict
    vs = (
        'You are the Portfolio Manager. 20 years A-share experience. 4 bull-bear cycles.\n\n'
        'Read the 3 reports above. Check the paper portfolio state. Deliver final decision.\n\n'
        'Output:\n'
        '1. Verdict (pick one: ATTACK / CAUTIOUS / DEFEND / REDUCE X% / LIQUIDATE)\n'
        '2. Position % (based on REAL paper portfolio holdings!)\n'
        '3. Entry/Stop prices (based on REAL K-line prices, not guesses!)\n'
        '4. Probability: win X% vs lose Y% vs worst Z%\n'
        '5. Correction line: "If X within Y days -> reverse, execute Z"\n'
        '6. Most likely loss scenario (specific, not vague)\n'
        '7. Alternative: if market consensus was disproven, what should we do instead?\n\n'
        + get_agent_knowledge('pm') + '\n'
        + get_agent_knowledge('trader') + '\n'
        + get_anti_hallucination_rules()
    )

    def verdict_node(state):
        market = state.get("macro_report", "")[:2000]
        news_r = state.get("news_report", "")[:2000]
        deep = state.get("deep_report", "")[:2000]
        paper = _get_paper_state()
        ctx = "Analysis:\n=== MACRO ===\n%s\n=== NEWS ===\n%s\n=== DEEP ===\n%s\n\n%s\n\nBased on above, deliver final verdict." % (market, news_r, deep, paper)
        result = dl.invoke([SystemMessage(content=vs), HumanMessage(content=ctx)])
        txt = result.content if hasattr(result, 'content') else str(result)
        return {"messages": [result], "final_trade_decision": txt}

    # Build graph: 4 nodes in chain
    wf = StateGraph(AgentState)
    wf.add_node("macro_agent", lambda s: _text_agent(ql, ms, s, 'macro_report'))
    wf.add_node("news_agent", lambda s: _text_agent(ql, ns, s, 'news_report'))
    wf.add_node("deep_agent", lambda s: _text_agent(dl, ds, s, 'deep_report'))
    wf.add_node("Verdict", verdict_node)

    wf.add_edge(START, "macro_agent")
    wf.add_edge("macro_agent", "news_agent")
    wf.add_edge("news_agent", "deep_agent")
    wf.add_edge("deep_agent", "Verdict")
    wf.add_edge("Verdict", END)

    return wf.compile()


if __name__ == '__main__':
    os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', 'DEEPSEEK_API_KEY_PLACEHOLDER')
    set_config({'llm_provider': 'deepseek', 'deep_think_llm': 'deepseek-chat', 'quick_think_llm': 'deepseek-chat'})
    print("AgentQuant 5-Agent Graph v3 compiled OK")
