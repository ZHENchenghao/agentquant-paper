# -*- coding: utf-8 -*-
"""
AgentQuant · ETF版哑铃回测
===========================
进攻: 行业动量top1 → 买该行业指数
防守: 红利低波(中证红利sh000922) / 可用行业替代
调仓: 月换, 大盘<20MA全仓防守
资金: 无需个股, 几千就够
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sys.path.insert(0,'D:/AgentQuant/our')
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
DB='D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB,read_only=True)

def get_month_end(c, year, month):
    target=date(year,month,28)+timedelta(days=4)
    target=target.replace(day=1)-timedelta(days=1)
    r=c.execute("SELECT MAX(trade_date) FROM kline_daily WHERE trade_date<=?",[target.isoformat()]).fetchone()
    return r[0] if r and r[0] else None

def get_industry_returns(c, ind_code, start, end):
    """行业指数区间收益(首日→末日)"""
    r=c.execute("""
        SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/
                NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
        FROM (SELECT close,
              ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
              ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
              FROM proxy_industry_daily WHERE stock_code=? AND trade_date>=? AND trade_date<=?)
    """,[ind_code,start.isoformat(),end.isoformat()]).fetchone()
    return r[0] if r and r[0] else 0

def get_top_industry(c, trade_date):
    """20日动量最强行业"""
    lookback=(trade_date-timedelta(days=30)).isoformat()
    td=trade_date.isoformat()
    r=c.execute(f"""
        SELECT industry,stock_code,
               (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=20 THEN close END),0)-1) mom
        FROM (SELECT industry,stock_code,close,
              ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
              FROM proxy_industry_daily WHERE trade_date>='{lookback}' AND trade_date<='{td}')
        WHERE rn<=20 GROUP BY industry,stock_code HAVING COUNT(*)>=15 ORDER BY mom DESC LIMIT 1
    """).fetchone()
    return (r[0],r[1],r[2]) if r else (None,None,0)

def get_defense_index(c):
    """防守: 红利低波 (近似: 银行/公用事业中动量最强的)"""
    # 申万行业代码: 银行=801780, 公用事业=801160, 交通运输=801170
    defense_inds=['801780','801160','801170']
    best_mom=-99; best_code=None
    td=date.today().isoformat()
    lookback=(date.today()-timedelta(days=30)).isoformat()
    for code in defense_inds:
        r=c.execute(f"""
            SELECT (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=20 THEN close END),0)-1) mom
            FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                  FROM proxy_industry_daily WHERE stock_code=? AND trade_date>='{lookback}' AND trade_date<='{td}')
            WHERE rn<=20
        """,[code]).fetchone()
        if r and r[0] and r[0]>best_mom:
            best_mom=r[0]; best_code=code
    return best_code or '801780'

def market_trend(c, trade_date):
    """沪深300是否>20MA"""
    r=c.execute("""
        SELECT MAX(close), AVG(close) FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
        )
    """,[trade_date.isoformat()]).fetchone()
    return r and r[0] and r[1] and r[0]>r[1]

def run_etf_backtest():
    c=cdb()
    months=[]
    for y in range(2021,2027):
        for m in range(1,13):
            dt=get_month_end(c,y,m)
            if dt and dt>=date(2021,1,29) and dt<=date.today(): months.append(dt)

    nav=1.0; nav_buyhold=1.0
    results=[]
    def_code=get_defense_index(c)

    for i,rebal_date in enumerate(months):
        if i==len(months)-1: break
        next_date=months[i+1]
        bull=market_trend(c,rebal_date)

        if bull:
            ind_name,ind_code,mom=get_top_industry(c,rebal_date)
            r=get_industry_returns(c,ind_code or def_code,rebal_date,next_date)
        else:
            ind_name='防守'; ind_code=def_code
            r=get_industry_returns(c,def_code,rebal_date,next_date)

        nav*=(1+r)

        # 基准: 沪深300 buy&hold (首尾收益)
        r300=c.execute("""
            SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
            FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
                  ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
                  FROM kline_daily WHERE ts_code='sh000300' AND trade_date>=? AND trade_date<=?)
        """,[rebal_date.isoformat(),next_date.isoformat()]).fetchone()
        r300_val=r300[0] if r300 and r300[0] else 0
        nav_buyhold*=(1+r300_val)

        results.append({'date':rebal_date,'bull':bull,'industry':ind_name,'ret':r,'nav':nav})
        if (i+1)%12==0: print(f'[{i+1}] {rebal_date} nav={nav:.3f}')

    m=len(results)
    ann=(nav)**(12/m)-1
    ann_bh=(nav_buyhold)**(12/m)-1
    # MDD
    nvs=[r['nav'] for r in results]
    peak=1.0; mdd=0
    for nv in nvs:
        if nv>peak: peak=nv
        dd=(nv/peak-1)*100
        if dd<mdd: mdd=dd

    print(f'\nETF哑铃: 年化={ann*100:.1f}% MDD={mdd:.1f}%')
    print(f'沪深300持有: 年化={ann_bh*100:.1f}%')
    print(f'超额: {(ann-ann_bh)*100:.1f}%')
    print(f'本金需求: <1000元 (1手ETF)')

    # 最近信号
    last=results[-1]
    sig='进攻' if last['bull'] else '防守'
    ind=last.get('industry','?')
    print(f'\n当前信号: {sig} | 行业: {ind}')
    c.close()
    return results

if __name__=='__main__':
    run_etf_backtest()
