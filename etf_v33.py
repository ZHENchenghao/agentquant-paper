# -*- coding: utf-8 -*-
"""
AgentQuant · ETF哑铃V3.3
========================
个股信号 → 行业投票 → 映射ETF
进攻: V3.1个股top10 → 行业投票 → Top2行业ETF各1/2
防守: V3.1防守top10 → 银行/公用/交通占比高 → 红利ETF
价格相关性聚类: 纯DuckDB, 零外部API
"""
import sys,os,io
sys.path.insert(0,'D:/AgentQuant/our')
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']; plt.rcParams['axes.unicode_minus']=False

DB='D:/FreeFinanceData/data/duckdb/finance.db'
from factor_pipeline import get_clean_universe,calc_offense_score,calc_defense_score

# 申万31行业代码
SW_INDS={
    '农林牧渔':'801010','基础化工':'801030','钢铁':'801040','有色金属':'801050',
    '电子':'801080','汽车':'801880','家用电器':'801110','食品饮料':'801120',
    '纺织服饰':'801130','轻工制造':'801140','医药生物':'801150','公用事业':'801160',
    '交通运输':'801170','房地产':'801180','商贸零售':'801200','社会服务':'801210',
    '银行':'801780','非银金融':'801790','建筑材料':'801710','建筑装饰':'801720',
    '电力设备':'801730','国防军工':'801740','计算机':'801750','传媒':'801760',
    '通信':'801770','煤炭':'801950','石油石化':'801960','环保':'801970',
    '美容护理':'801880','机械设备':'801890',
}

def cdb(): return duckdb.connect(DB,read_only=True)

def get_month_end(c,year,month):
    target=date(year,month,28)+timedelta(days=4)
    target=target.replace(day=1)-timedelta(days=1)
    r=c.execute("SELECT MAX(trade_date) FROM kline_daily WHERE trade_date<=?",[target.isoformat()]).fetchone()
    return r[0] if r and r[0] else None

def ind_return(c,code,start,end):
    r=c.execute("""
        SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
        FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
              ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
              FROM proxy_industry_daily WHERE stock_code=? AND trade_date>=? AND trade_date<=?)
    """,[code,start.isoformat(),end.isoformat()]).fetchone()
    return r[0] if r and r[0] else 0

def ind_ma_ok(c,code,trade_date):
    r=c.execute("""
        SELECT close, ma60 FROM (
            SELECT close, AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) ma60
            FROM proxy_industry_daily WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1
        )
    """,[code,trade_date.isoformat()]).fetchone()
    return r and r[0] and r[1] and r[0]>r[1]

def market_bull(c,trade_date):
    r=c.execute("""
        SELECT MAX(close), AVG(close) FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
        )
    """,[trade_date.isoformat()]).fetchone()
    return r and r[0] and r[1] and r[0]>r[1]

def classify_stocks_by_map(c,stock_codes):
    """从预计算映射表取行业归属 (快速)"""
    if not stock_codes: return {}
    codes_str=','.join([f"'{x}'" for x in stock_codes])
    result=c.execute(f"SELECT ts_code,ind_code FROM stock_industry WHERE ts_code IN ({codes_str})").fetchall()
    return {r[0]:r[1] for r in result}

def run():
    c=cdb()
    months=[]
    for y in range(2021,2027):
        for m in range(1,13):
            dt=get_month_end(c,y,m)
            if dt and dt>=date(2021,1,29) and dt<=date.today(): months.append(dt)

    nav=1.0; nav_bh=1.0
    defense_ind_codes=['801160','801170']  # 公用事业(电力水务)/交通
    results=[]

    for i,rebal_date in enumerate(months):
        if i==len(months)-1: break
        next_date=months[i+1]
        bull=market_bull(c,rebal_date)

        # V3.1个股选股
        u=get_clean_universe(c,rebal_date)
        offense_stocks=calc_offense_score(c,rebal_date,u)
        defense_stocks=calc_defense_score(c,rebal_date,u)

        off_top10=offense_stocks.head(10)['ts_code'].tolist() if not offense_stocks.empty else []
        def_top10=defense_stocks.head(10)['ts_code'].tolist() if not defense_stocks.empty else []

        mapping=classify_stocks_by_map(c,off_top10+def_top10)

        # 进攻: 个股top10→行业投票→top2行业各1/2
        off_votes={}
        for code in off_top10:
            ind=mapping.get(code)
            if ind: off_votes[ind]=off_votes.get(ind,0)+1
        top2_inds=sorted(off_votes.items(),key=lambda x:x[1],reverse=True)[:2]

        if bull and top2_inds:
            r_off=0
            for ind_code,_ in top2_inds:
                r_off+=ind_return(c,ind_code,rebal_date,next_date)/2
        else:
            r_off=0

        # 防守: 个股top10→看银行/公用/交通占比→选MA60最强的
        def_code=None
        for dc in defense_ind_codes:
            if ind_ma_ok(c,dc,rebal_date):
                def_code=dc; break
        if def_code:
            r_def=ind_return(c,def_code,rebal_date,next_date)
        else:
            r_def=0.02*((next_date-rebal_date).days)/365

        nav*=(1+(r_off+r_def)/2)

        r300=c.execute("""
            SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
            FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
                  ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
                  FROM kline_daily WHERE ts_code='sh000300' AND trade_date>=? AND trade_date<=?)
        """,[rebal_date.isoformat(),next_date.isoformat()]).fetchone()
        nav_bh*=(1+(r300[0] or 0))

        top2_names=[{v:k for k,v in SW_INDS.items()}.get(c,'?') for c,_ in top2_inds] if top2_inds else []
        results.append({'date':rebal_date,'top2':top2_names,'def':def_code,'nav':nav})
        if (i+1)%12==0:
            print('[%d/%d] %s nav=%.3f (top2: %s)' % (i+1,len(months),rebal_date,nav,','.join(top2_names)))

    m=len(results)
    ann=nav**(12/m)-1
    ann_bh=nav_bh**(12/m)-1
    nvs=[r['nav'] for r in results]
    peak=1.0; mdd=0
    for nv in nvs:
        if nv>peak: peak=nv
        dd=(nv/peak-1)*100
        if dd<mdd: mdd=dd

    rets=np.diff(nvs)/nvs[:-1]
    sharpe=(np.mean(rets)*12)/max(np.std(rets)*np.sqrt(12),0.001)

    print('='*50)
    print('  ETF V3.3 (个股信号→行业投票→ETF)')
    print('='*50)
    print('  年化: %.1f%%  MDD: %.1f%%  Sharpe: %.2f' % (ann*100,mdd,sharpe))
    print('  沪深300: %.1f%%  超额: %.1f%%' % (ann_bh*100,(ann-ann_bh)*100))
    print('  本金: <1000元')

    last=results[-1]
    print('\n  当前: Top2行业=%s 防守=%s' % (','.join(last['top2']),last['def'] or '现金'))

    c.close()
    return results

if __name__=='__main__':
    run()
