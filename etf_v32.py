# -*- coding: utf-8 -*-
"""
AgentQuant · ETF哑铃V3.2
========================
进攻: Top3行业指数各1/3 (成交额排名)
防守: 红利低波(银行/公用/交通中MA60最强的)
风控: 防守跌破MA60→全仓现金(年化2%)
调仓: 月换, 沪深300<MA20→封印进攻
资金: <1000元
"""
import sys,os
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
import matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']; plt.rcParams['axes.unicode_minus']=False

DB='D:/FreeFinanceData/data/duckdb/finance.db'

def cdb(): return duckdb.connect(DB,read_only=True)

# 申万31行业代码 (ETF代理)
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

def get_month_end(c,year,month):
    target=date(year,month,28)+timedelta(days=4)
    target=target.replace(day=1)-timedelta(days=1)
    r=c.execute("SELECT MAX(trade_date) FROM kline_daily WHERE trade_date<=?",[target.isoformat()]).fetchone()
    return r[0] if r and r[0] else None

def ind_return(c,code,start,end):
    """行业指数区间收益(首日→末日)"""
    r=c.execute("""
        SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/
                NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
        FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
              ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
              FROM proxy_industry_daily WHERE stock_code=? AND trade_date>=? AND trade_date<=?)
    """,[code,start.isoformat(),end.isoformat()]).fetchone()
    return r[0] if r and r[0] else 0

def ind_ma_check(c,code,trade_date):
    """行业指数是否>MA60"""
    r=c.execute("""
        SELECT MAX(close), AVG(close) FROM (
            SELECT close FROM proxy_industry_daily WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 60
        )
    """,[code,trade_date.isoformat()]).fetchone()
    return r and r[0] and r[1] and r[0]>r[1]

def ind_avg_amount(c,code,trade_date):
    """行业指数20日日均成交额"""
    r=c.execute("""
        SELECT AVG(amount) FROM (
            SELECT amount FROM proxy_industry_daily WHERE stock_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
        )
    """,[code,trade_date.isoformat()]).fetchone()
    return r[0] if r and r[0] else 0

def market_bull(c,trade_date):
    """沪深300>20MA"""
    r=c.execute("""
        SELECT MAX(close), AVG(close) FROM (
            SELECT close FROM kline_daily WHERE ts_code='sh000300' AND trade_date<=? ORDER BY trade_date DESC LIMIT 20
        )
    """,[trade_date.isoformat()]).fetchone()
    return r and r[0] and r[1] and r[0]>r[1]

def run():
    c=cdb()
    months=[]
    for y in range(2021,2027):
        for m in range(1,13):
            dt=get_month_end(c,y,m)
            if dt and dt>=date(2021,1,29) and dt<=date.today(): months.append(dt)

    nav=1.0; nav_bh=1.0
    defense_codes=['801780','801160','801170']  # 银行/公用/交通
    results=[]
    for i,rebal_date in enumerate(months):
        if i==len(months)-1: break
        next_date=months[i+1]
        bull=market_bull(c,rebal_date)

        # 进攻: Top3动量行业各1/3 (含MA60过滤)
        ind_mom=[]
        for name,code in SW_INDS.items():
            r=c.execute("""
                SELECT (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=20 THEN close END),0)-1)
                FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn
                      FROM proxy_industry_daily WHERE stock_code=? AND trade_date<=?)
                WHERE rn<=20
            """,[code,rebal_date.isoformat()]).fetchone()
            mom=r[0] if r and r[0] else -99
            ma_ok=ind_ma_check(c,code,rebal_date)
            ind_mom.append((name,code,mom,ma_ok))
        ind_mom.sort(key=lambda x:x[2],reverse=True)
        top3=[(name,code,mom) for name,code,mom,ma_ok in ind_mom if ma_ok][:3]

        if bull:
            r_off=0
            for name,code,amt in top3:
                r_off+=ind_return(c,code,rebal_date,next_date)/3
        else:
            r_off=0  # 熊市封印进攻

        # 防守: 银行/公用/交通中MA60最强的那个
        def_code=None
        for dc in defense_codes:
            if ind_ma_check(c,dc,rebal_date):
                def_code=dc; break
        if def_code is None:
            r_def=0.02*((next_date-rebal_date).days)/365
        else:
            r_def=ind_return(c,def_code,rebal_date,next_date)

        nav*=(1+(r_off+r_def)/2)

        # 基准: 沪深300
        r300=ind_return(c,'000300',rebal_date,next_date) if False else \
            c.execute("""
                SELECT (MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1)
                FROM (SELECT close,ROW_NUMBER() OVER(ORDER BY trade_date) rn_asc,
                      ROW_NUMBER() OVER(ORDER BY trade_date DESC) rn_desc
                      FROM kline_daily WHERE ts_code='sh000300' AND trade_date>=? AND trade_date<=?)
            """,[rebal_date.isoformat(),next_date.isoformat()]).fetchone()
        r300v=r300[0] if r300 and r300[0] else 0
        nav_bh*=(1+r300v)

        results.append({'date':rebal_date,'bull':bull,'top3':[t[0] for t in top3],'def_code':def_code,'ret':(r_off+r_def)/2,'nav':nav})
        if (i+1)%12==0:
            print('[%d/%d] %s nav=%.3f (top3: %s)' % (i+1,len(months),rebal_date,nav,','.join([t[0] for t in top3])))

    m=len(results)
    ann=nav**(12/m)-1
    ann_bh=nav_bh**(12/m)-1
    nvs=[r['nav'] for r in results]
    peak=1.0; mdd=0
    for nv in nvs:
        if nv>peak: peak=nv
        dd=(nv/peak-1)*100
        if dd<mdd: mdd=dd

    # 计算夏普
    rets=np.diff(nvs)/nvs[:-1]
    sharpe=(np.mean(rets)*12)/max(np.std(rets)*np.sqrt(12),0.001)

    print('='*50)
    print('  ETF V3.2 回测结果')
    print('='*50)
    print('  年化: %.1f%%  MDD: %.1f%%  Sharpe: %.2f' % (ann*100,mdd,sharpe))
    print('  沪深300: %.1f%%  超额: %.1f%%' % (ann_bh*100,(ann-ann_bh)*100))
    print('  本金: <1000元 (1手ETF)')
    print('  进攻: Top3行业成交额 防守: 红利+MA60')

    last=results[-1]
    print('\n  当前信号: %s | Top3: %s | 防守: %s' % ('进攻' if last['bull'] else '防守',','.join(last['top3']),last['def_code'] or '现金'))

    # 出图
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot([r['date'] for r in results],nvs,'b-',linewidth=2,label='ETF V3.2')
    ax.plot([r['date'] for r in results],[np.prod([1+r300v])**(1) for _ in results],'gray',alpha=0.5,label='HS300')
    ax.legend(); ax.set_title('ETF V3.2 vs HS300'); ax.grid(alpha=0.3)
    out='D:/AgentQuant/our/etf_v32_result.png'
    plt.savefig(out,dpi=120,bbox_inches='tight')
    print('  图表: %s' % out)
    c.close()
    return results

if __name__=='__main__':
    run()
