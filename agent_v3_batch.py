# -*- coding: utf-8 -*-
"""
AgentQuant · 16Agent批量分析 + V3策略 + DuckDB数据
=====================================================
流程: V3策略选池 → DuckDB预加载数据 → 16Agent逐一验 → 排名输出

用法: python agent_v3_batch.py
"""
import sys,os,io,json,time
sys.path.insert(0,r'D:\AgentQuant\agents\TradingAgents-astock')
sys.path.insert(0,r'D:\AgentQuant\our')

import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
from factor_pipeline import (
    get_clean_universe, calc_defense_score, calc_offense_score
)

DB='D:/FreeFinanceData/data/duckdb/finance.db'

# ═══════════════════════════════
# 1. DuckDB数据预加载
# ═══════════════════════════════

def load_stock_context(code):
    """从DuckDB加载个股核心数据, 替代TradingAgents的数据工具"""
    c=duckdb.connect(DB,read_only=True)
    ts=f'{code}.SH' if code.startswith('6') else f'{code}.SZ'
    ctx={}

    # 最新行情
    r=c.execute("SELECT trade_date,close,change_pct,vol,amount,turnover_rate FROM kline_daily WHERE ts_code=? ORDER BY trade_date DESC LIMIT 5",[ts]).fetchall()
    ctx['kline']=[{'date':str(x[0]),'close':x[1],'change_pct':x[2],'vol':x[3],'amount':x[4]} for x in r]

    # PE/PB
    r=c.execute("SELECT pe_ttm,pb,total_mv FROM valuation_daily WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1",[ts]).fetchone()
    ctx['valuation']={'pe':r[0] if r else None,'pb':r[1] if r else None,'mv':r[2] if r else None}

    # 最新财报
    r=c.execute("""
        SELECT report_date,net_profit,revenue,roe,gross_margin,eps
        FROM financial_statements WHERE ts_code=? AND report_type='annual'
        ORDER BY report_date DESC LIMIT 2
    """,[ts]).fetchall()
    ctx['financials']=[{'date':str(x[0]),'net_profit':x[1],'revenue':x[2],'roe':x[3],'gross_margin':x[4],'eps':x[5]} for x in r]

    # 北向资金
    r=c.execute("SELECT trade_date,net_flow FROM north_bound_flow WHERE ts_code=? ORDER BY trade_date DESC LIMIT 5",[ts]).fetchall()
    ctx['north_flow']=[{'date':str(x[0]),'net_flow':x[1]} for x in r] if r else []

    # 最近新闻
    r=c.execute("SELECT title,source,publish_date FROM news_articles ORDER BY id DESC LIMIT 5").fetchall()
    ctx['news']=[{'title':x[0][:80],'source':x[1],'date':str(x[2])} for x in r]

    c.close()
    return ctx


# ═══════════════════════════════
# 2. V3策略上下文
# ═══════════════════════════════

def get_v3_strategy():
    c=duckdb.connect(DB,read_only=True)
    td=c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]

    # 大盘趋势
    r=c.execute("""
        SELECT AVG(close),MAX(close) FROM (SELECT close FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20)
    """,[td.isoformat()]).fetchone()
    trend='BEAR' if r[1] and r[0] and r[1]<r[0] else 'BULL'

    # VIX
    vix_r=c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",[td.isoformat()]).fetchone()
    vix=vix_r[0] if vix_r else 20

    # 行业top3
    ind=c.execute(f"""
        SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1) mom
        FROM (SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
              FROM proxy_industry_daily WHERE trade_date>='{(td-timedelta(days=30)).isoformat()}') t
        WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8 ORDER BY mom DESC LIMIT 3
    """).fetchall()
    top_ind=[r[0] for r in ind]

    # 沪深300 20日涨跌
    r=c.execute("SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20",[td.isoformat()]).fetchall()
    idx_chg=(r[0][0]/r[-1][0]-1)*100 if len(r)>=2 else 0

    c.close()

    if trend=='BEAR':
        strategy='DEFENSE: 低Beta红利池 + 国债'
        allocation='防守80%进攻20%'
    elif vix>28:
        strategy='CAUTION: VIX高企'
        allocation='防守70%进攻30%'
    else:
        strategy='ATTACK: 行业动量进攻+低Beta对冲'
        allocation='防守40%进攻60%'

    return {'trade_date':str(td),'trend':trend,'vix':vix,'top_industries':top_ind,
            'idx_20d_chg':round(idx_chg,1),'strategy':strategy,'allocation':allocation}


# ═══════════════════════════════
# 3. V3选池
# ═══════════════════════════════

def get_v3_pools(v3_state):
    """V3策略: 防守池+进攻池"""
    c=duckdb.connect(DB,read_only=True)
    trade_date_str=v3_state['trade_date']
    trade_date=date.fromisoformat(trade_date_str)
    universe=get_clean_universe(c,trade_date)

    defense=calc_defense_score(c,trade_date,universe)
    offense=calc_offense_score(c,trade_date,universe)

    def_top=defense.head(15)['ts_code'].tolist() if not defense.empty else []
    off_top=offense.head(15)['ts_code'].tolist() if not offense.empty else []

    c.close()
    return {'defense':def_top,'offense':off_top}


# ═══════════════════════════════
# 4. Agent分析 (逐只)
# ═══════════════════════════════

# 全局宏观上下文(只构建一次)
_macro_context = None

def get_macro_context():
    global _macro_context
    if _macro_context is None:
        try:
            from agent_macro_context import build_macro_context
            _macro_context = build_macro_context()
        except Exception as e:
            _macro_context = f"[宏观数据获取失败: {e}]"
    return _macro_context


def agent_analyze_batch(pool_codes, v3_state, label, max_stocks=10):
    """批量Agent分析, 捕获完整推理链(含宏观上下文注入)"""
    os.environ['DEEPSEEK_API_KEY']=os.environ.get('DEEPSEEK_API_KEY','DEEPSEEK_API_KEY_PLACEHOLDER')
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    config=DEFAULT_CONFIG.copy()
    config['llm_provider']='deepseek'
    config['deep_think_llm']='deepseek-chat'
    config['quick_think_llm']='deepseek-chat'
    config['max_debate_rounds']=1
    config['max_risk_discuss_rounds']=1

    ta=TradingAgentsGraph(debug=False,config=config)

    # ── 注入宏观上下文 ──
    macro_ctx = get_macro_context()
    _orig_get_past = ta.memory_log.get_past_context
    def _patched_get_past(ticker):
        orig = _orig_get_past(ticker)
        return f"{macro_ctx}\n\n---\n\n{orig}" if orig else macro_ctx
    ta.memory_log.get_past_context = _patched_get_past
    # ──────────────────

    results=[]
    codes=pool_codes[:max_stocks]
    n=len(codes)

    # 并行模式: 多只股票同时跑 (每只独立TradingAgentsGraph)
    import concurrent.futures
    import threading

    def _analyze_one(code, idx):
        """单只股票分析 (线程安全)"""
        # 每线程独立的TradingAgentsGraph实例
        ta_local = TradingAgentsGraph(debug=False, config=config)
        ctx_local = get_macro_context()
        _orig = ta_local.memory_log.get_past_context
        def patched(ticker, ctx=ctx_local, orig_fn=_orig):
            old = orig_fn(ticker) if callable(orig_fn) else ''
            return ctx + '\n\n---\n\n' + str(old) if old else ctx
        ta_local.memory_log.get_past_context = patched

        for attempt in range(3):
            try:
                final_state, decision = ta_local.propagate(code, v3_state['trade_date'])
                if decision and final_state:
                    d_str = str(decision)
                    signal = 'HOLD'
                    for line in d_str.split('\n'):
                        ls = line.strip()
                        if ls.startswith('**Rating**'):
                            r = ls.split('**Rating**')[1].strip().strip(':').strip().lower()
                            if 'buy' in r: signal = 'BUY'
                            elif 'sell' in r: signal = 'SELL'
                            elif 'overweight' in r: signal = 'BUY'
                            elif 'underweight' in r: signal = 'SELL'
                            elif 'hold' in r: signal = 'HOLD'
                            break
                    print(f'[{idx+1}/{n}] {code}...{signal}')
                    return {
                        'code': code, 'signal': signal, 'final_decision': d_str,
                        'market_report': final_state.get('market_report', ''),
                        'sentiment_report': final_state.get('sentiment_report', ''),
                        'news_report': final_state.get('news_report', ''),
                        'fundamentals_report': final_state.get('fundamentals_report', ''),
                        'investment_plan': final_state.get('investment_plan', ''),
                        'trader_decision': final_state.get('trader_investment_plan', ''),
                    }
            except Exception as e:
                if attempt < 2:
                    print(f'[{idx+1}/{n}] {code}...R{attempt+1}')
                    time.sleep(3)
                else:
                    print(f'[{idx+1}/{n}] {code}...ERR: {e}')
        return {'code': code, 'signal': 'ERR', 'final_decision': f'Failed after 3 retries'}

    # 用线程池并行 (I/O密集型, DeepSeek API)
    workers = min(4, n)  # 最多4并发, 避免API限流
    print(f'\nParallel mode: {workers} workers for {n} stocks\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_analyze_one, code, i): code for i, code in enumerate(codes)}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # 恢复原始顺序
    results.sort(key=lambda r: codes.index(r['code']))
    return results


def generate_full_report(v3_state, results, output_path):
    """生成完整推理报告 (Markdown)"""
    buys=[r for r in results if r['signal']=='BUY']
    holds=[r for r in results if r['signal']=='HOLD']
    sells=[r for r in results if r['signal']=='SELL']
    errs=[r for r in results if r['signal']=='ERR']

    lines=[]
    lines.append('# AgentQuant 16Agent 完整推理报告')
    lines.append(f'**日期**: {v3_state["trade_date"]}')
    lines.append(f'**市场状态**: {v3_state["trend"]} | VIX={v3_state["vix"]:.1f}')
    lines.append(f'**策略**: {v3_state["strategy"]}')
    lines.append(f'**配置**: {v3_state["allocation"]}')
    lines.append(f'**沪深300 20日**: {v3_state["idx_20d_chg"]:+.1f}%')
    lines.append(f'**强势行业**: {", ".join(v3_state["top_industries"])}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 汇总')
    lines.append(f'| 信号 | 数量 |')
    lines.append(f'|------|------|')
    lines.append(f'| ✅ BUY  | {len(buys)} |')
    lines.append(f'| ⏸️ HOLD | {len(holds)} |')
    lines.append(f'| ❌ SELL | {len(sells)} |')
    if errs: lines.append(f'| ⚠️ ERR  | {len(errs)} |')
    lines.append('')

    # 逐票详细推理
    for r in results:
        code=r['code']
        signal=r['signal']
        sig_emoji={'BUY':'✅','HOLD':'⏸️','SELL':'❌','ERR':'⚠️'}.get(signal,'❓')
        lines.append('---')
        lines.append(f'## {sig_emoji} {code} → {signal}')
        lines.append('')

        # 最终决策
        fd=r.get('final_decision','')
        if fd:
            lines.append('### 最终裁决')
            lines.append(f'```')
            lines.append(fd[:2000])
            lines.append(f'```')
            lines.append('')

        # 投资计划
        plan=r.get('investment_plan','')
        if plan:
            lines.append('### 投资计划')
            lines.append(f'```')
            lines.append(plan[:1500])
            lines.append(f'```')
            lines.append('')

        # 辩论裁决
        judge=r.get('judge_decision','')
        if judge:
            lines.append('### 多空辩论裁决')
            lines.append(f'```')
            lines.append(judge[:1500])
            lines.append(f'```')
            lines.append('')

        # 风控裁决
        rj=r.get('risk_judge','')
        if rj:
            lines.append('### 风控辩论裁决')
            lines.append(f'```')
            lines.append(rj[:1500])
            lines.append(f'```')
            lines.append('')

        # 市场报告 (摘要)
        mr=r.get('market_report','')
        if mr:
            lines.append('### 市场环境')
            lines.append(f'```')
            lines.append(mr[:1000])
            lines.append(f'```')
            lines.append('')

        # 基本面报告 (摘要)
        fr=r.get('fundamentals_report','')
        if fr:
            lines.append('### 基本面分析')
            lines.append(f'```')
            lines.append(fr[:1000])
            lines.append(f'```')
            lines.append('')

        # 情绪报告
        sr=r.get('sentiment_report','')
        if sr:
            lines.append('### 市场情绪')
            lines.append(f'```')
            lines.append(sr[:800])
            lines.append(f'```')
            lines.append('')

        # 交易员计划
        td=r.get('trader_decision','')
        if td:
            lines.append('### 交易员执行计划')
            lines.append(f'```')
            lines.append(td[:1000])
            lines.append(f'```')
            lines.append('')

    # 尾部汇总
    lines.append('---')
    lines.append('')
    lines.append('## 最终建议')
    if buys:
        lines.append(f'**买入候选 ({len(buys)}只)**: {", ".join([r["code"] for r in buys])}')
    else:
        lines.append('**买入候选: 无** — Agent全灭进攻池')
    lines.append('')

    # 写入文件
    report='\n'.join(lines)
    with open(output_path,'w',encoding='utf-8') as f:
        f.write(report)
    print(f'\n  完整推理报告: {output_path}')
    return report


# ═══════════════════════════════
# 5. 主入口
# ═══════════════════════════════

def main(max_per_pool=10):
    print('='*60)
    print('  AgentQuant 16Agent批量分析 V3')
    print('='*60)

    # V3策略
    v3=get_v3_strategy()
    print(f'\n  V3策略: {v3["trend"]} | VIX={v3["vix"]:.1f} | {v3["strategy"]}')
    print(f'  沪深300 20日: {v3["idx_20d_chg"]:+.1f}%')
    print(f'  强势行业: {v3["top_industries"]}')

    # 选池
    pools=get_v3_pools(v3)
    print(f'\n  防守池: {len(pools["defense"])}只')
    print(f'  进攻池: {len(pools["offense"])}只')

    # 根据市场状态决定先分析哪个池, 为空则fallback
    off_n=len(pools['offense']); def_n=len(pools['defense'])
    if v3['trend']=='BEAR' or off_n==0:
        primary=pools['defense']; primary_label='防守'
        print(f'\n  >>> 防守池 (进攻池={off_n}只) <<<')
    else:
        primary=pools['offense']; primary_label='进攻'
        print(f'\n  >>> 进攻池 <<<')

    # Agent批量
    print(f'\n--- {primary_label}池Agent分析 ---')
    results=agent_analyze_batch(primary,v3,primary_label,max_per_pool)

    # 汇总
    buys=[r for r in results if r['signal']=='BUY']
    holds=[r for r in results if r['signal']=='HOLD']
    sells=[r for r in results if r['signal']=='SELL']
    errs=[r for r in results if r['signal']=='ERR']

    print(f'\n{"="*60}')
    print(f'  Agent裁决汇总')
    print(f'{"="*60}')
    print(f'  BUY:  {len(buys)}只')
    print(f'  HOLD: {len(holds)}只')
    print(f'  SELL: {len(sells)}只')
    if errs: print(f'  ERR:  {len(errs)}只')

    # 保存JSON
    with open('D:/AgentQuant/our/agent_verdicts.json','w',encoding='utf-8') as f:
        out={'v3_strategy':v3,'results':results,'date':str(date.today())}
        json.dump(out,f,ensure_ascii=False,indent=2,default=str)
    print(f'\n  裁决JSON: D:/AgentQuant/our/agent_verdicts.json')

    # 生成完整推理报告
    report_path=f'D:/AgentQuant/our/agent_full_report_{date.today().strftime("%Y%m%d")}.md'
    generate_full_report(v3,results,report_path)

    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--max',type=int,default=15,help='每池最多分析数')
    args=p.parse_args()
    main(args.max)
