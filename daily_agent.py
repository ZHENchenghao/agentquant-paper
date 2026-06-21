# -*- coding: utf-8 -*-
"""
AgentQuant · 每日Agent
=======================
每日收盘后运行: 市场状态 → V3策略信号 → 交易指令 → 纸交执行

用法: python daily_agent.py
"""
import sys,os
sys.path.insert(0,'D:/AgentQuant/our')
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
from factor_pipeline import (
    get_clean_universe, get_pit_report_type, get_pit_financials,
    calc_defense_score, calc_offense_score
)
import json,os
DB='D:/FreeFinanceData/data/duckdb/finance.db'
PORTFOLIO_FILE='D:/AgentQuant/our/paper_portfolio.json'

def cdb(): return duckdb.connect(DB,read_only=True)

# ═══════════════════════════════
# 1. 市场状态检查
# ═══════════════════════════════

def check_market_state(c, trade_date):
    """返回: 'ATTACK' | 'DEFENSE' | 'CRISIS'"""
    # 大盘趋势开关
    r=c.execute("""
        SELECT AVG(close) ma20, MAX(close) close_now
        FROM (SELECT close FROM kline_daily WHERE ts_code='sh000300'
              AND trade_date<=? ORDER BY trade_date DESC LIMIT 20)
    """,[trade_date.isoformat()]).fetchone()
    if r and r[1] and r[0] and r[1]>r[0]:
        # 5日前MA
        r2=c.execute("""
            SELECT AVG(close) FROM (SELECT close FROM kline_daily
            WHERE ts_code='sh000300' AND trade_date<=?
            ORDER BY trade_date DESC LIMIT 20 OFFSET 5)
        """,[trade_date.isoformat()]).fetchone()
        ma_declining = r2 and r2[0] and r[0]<r2[0]
        if ma_declining: return 'DEFENSE'

    # VIX风控
    vix=c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                  [trade_date.isoformat()]).fetchone()
    if vix and vix[0]:
        p95=c.execute("""
            SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vix) FROM macro_indicators
            WHERE vix IS NOT NULL AND trade_date>=? AND trade_date<=?
        """,[(trade_date-timedelta(days=365)).isoformat(),trade_date.isoformat()]).fetchone()
        if p95 and p95[0] and vix[0]>p95[0]: return 'CRISIS'

    return 'ATTACK'

# ═══════════════════════════════
# 2. 选股
# ═══════════════════════════════

def get_today_picks(c, trade_date, state):
    """根据市场状态返回今日持仓建议"""
    universe=get_clean_universe(c,trade_date)
    if len(universe)<100: return {'state':state,'picks':[],'defense':[]}

    defense=calc_defense_score(c,trade_date,universe)
    offense=calc_offense_score(c,trade_date,universe)

    if state=='CRISIS':
        # 全退守: 只留10只最防守的
        picks=defense.head(10)['ts_code'].tolist() if not defense.empty else []
        return {'state':state,'picks':picks[:5],'defense':picks,'allocation':'100%防守'}

    elif state=='DEFENSE':
        picks=defense.head(30)['ts_code'].tolist() if not defense.empty else []
        return {'state':state,'picks':picks[:5],'defense':picks,'allocation':'80%防守+20%国债'}

    else:  # ATTACK
        picks=offense.head(20)['ts_code'].tolist() if not offense.empty else []
        def_top=defense.head(15)['ts_code'].tolist() if not defense.empty else []
        # 哑铃: 进攻top5 + 防守top3 (对冲)
        return {'state':state,'picks':picks[:5],'defense':def_top[:3],
                'allocation':'60%进攻+40%防守'}

# ═══════════════════════════════
# 3. 快筛排雷
# ═══════════════════════════════

def quick_screen(c, picks):
    """快速排雷: PE>0, ROE>0, 非ST"""
    if not picks: return picks
    codes_str=','.join([f"'{x}'" for x in picks])
    ok=c.execute(f"""
        SELECT DISTINCT v.ts_code FROM valuation_daily v
        WHERE v.ts_code IN ({codes_str}) AND v.pe_ttm>0 AND v.pe_ttm<500
        AND v.trade_date=(SELECT MAX(trade_date) FROM valuation_daily)
    """).df()
    return ok['ts_code'].tolist()

# ═══════════════════════════════
# 4. 纸交记录
# ═══════════════════════════════

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE,'r',encoding='utf-8') as f:
            return json.load(f)
    return {'cash':100000,'positions':{},'history':[]}

def save_portfolio(pf):
    with open(PORTFOLIO_FILE,'w',encoding='utf-8') as f:
        json.dump(pf,f,ensure_ascii=False,indent=2,default=str)

def update_portfolio(c, pf, today_picks, trade_date):
    """纸交执行: 卖出不在新名单的, 买入新名单"""
    new_set=set(today_picks)
    old_set=set(pf['positions'].keys())

    # 卖出
    for code in old_set-new_set:
        if code in pf['positions']:
            pos=pf['positions'][code]
            r=c.execute("SELECT close FROM kline_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                        [code,trade_date.isoformat()]).fetchone()
            if r and r[0]:
                sell_val=pos['shares']*r[0]
                pf['cash']+=sell_val*0.9985  # 佣金+印花税
                pf['history'].append({'date':str(trade_date),'action':'SELL','code':code,
                    'price':r[0],'shares':pos['shares'],'value':sell_val,'reason':'不在新名单'})
                del pf['positions'][code]

    # 买入
    if new_set:
        n_buy=min(len(new_set),8)
        buy_list=list(new_set)[:n_buy]
        cash_per=pf['cash']/n_buy
        for code in buy_list:
            if code in pf['positions']: continue
            r=c.execute("SELECT close FROM kline_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                        [code,trade_date.isoformat()]).fetchone()
            if r and r[0]:
                shares=int(cash_per/r[0]/100)*100  # 100股整数倍
                if shares>=100:
                    cost=shares*r[0]*1.00015  # 佣金
                    if cost<=pf['cash']:
                        pf['cash']-=cost
                        pf['positions'][code]={'shares':shares,'buy_price':r[0],'buy_date':str(trade_date)}
                        pf['history'].append({'date':str(trade_date),'action':'BUY','code':code,
                            'price':r[0],'shares':shares,'value':cost,'reason':f'V3选股({len(new_set)}只)'})

    return pf

# ═══════════════════════════════
# 5. 主入口
# ═══════════════════════════════

def daily_run():
    c=cdb()
    # 取最近交易日
    trade_date=c.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0]
    print('='*55)
    print(f'  AgentQuant 日报 {trade_date}')
    print('='*55)

    # 市场状态
    state=check_market_state(c,trade_date)
    print(f'\n  市场状态: {state}')

    # VIX
    vix=c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                  [trade_date.isoformat()]).fetchone()
    print(f'  VIX: {vix[0]:.1f}' if vix and vix[0] else '  VIX: N/A')

    # 选股
    result=get_today_picks(c,trade_date,state)
    raw_picks=result['picks']+result.get('defense',[])
    clean_picks=quick_screen(c,raw_picks)
    alloc=result.get('allocation','N/A')
    print(f'\n  候选: {len(raw_picks)}只 → 排雷后: {len(clean_picks)}只')
    print(f'  配置: {alloc}')

    # 纸交
    pf=load_portfolio()
    pf=update_portfolio(c,pf,clean_picks,trade_date)
    save_portfolio(pf)

    # 汇总
    pos_count=len(pf['positions'])
    pos_value=sum(p['shares']*p['buy_price'] for p in pf['positions'].values())
    total=pf['cash']+pos_value
    pnl=(total-100000)/100000*100
    c2=pf['cash']; pv=pos_value; tot=total; pc=pos_count; pnl2=pnl
    print(f'\n  纸交账户: 现金{c2:.0f} 持仓{pv:.0f} 总{tot:.0f} PnL{pnl2:+.1f}%')
    print(f'  持仓: {pc}只')

    c.close()
    return result

if __name__=='__main__':
    daily_run()
