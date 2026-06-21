# -*- coding: utf-8 -*-
"""
AgentQuant - 因子IC快速检验
8因子: PE/ROE/净利增速/动量/波动率/北向/VIX/行业动量
标准: |Rank IC| > 0.02 = 有效
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
from scipy import stats

DB='D:/FreeFinanceData/data/duckdb/finance.db'
S55='='*55

def conn():
    for i in range(20):
        try:
            c=duckdb.connect(DB,read_only=True);c.execute('SELECT 1');return c
        except:
            import time;time.sleep(min(2**i,30))
    return duckdb.connect(DB,read_only=True)

def rank_ic(fv,fr):
    m=fv.notna()&fr.notna()
    if m.sum()<30: return None,None
    ic,p=stats.spearmanr(fv[m],fr[m])
    return round(ic,4),round(p,4)

def layered(fv,fr,g=5):
    m=fv.notna()&fr.notna()
    if m.sum()<50: return None
    df=pd.DataFrame({'f':fv[m],'r':fr[m]})
    df['g']=pd.qcut(df['f'].rank(method='first'),g,labels=False,duplicates='drop')
    grp=df.groupby('g')['r'].mean()
    return {f'G{i}':round(grp.get(i,0)*100,2) for i in range(g)}

def main():
    c=conn()
    print(S55+'\n  AgentQuant 因子IC检验\n'+S55)
    today=date.today()
    T20=(today-timedelta(days=20)).isoformat()
    T10=(today-timedelta(days=10)).isoformat()
    T30=(today-timedelta(days=30)).isoformat()
    results=[]

    # 收益: 用close算, change_pct为NULL
    # 收益: 20日/10日 forward return (从close算)
    def get_fwd_ret(c, days):
        sd=(today-timedelta(days=days+5)).isoformat()
        return c.execute(f"""
            SELECT ts_code,
                   (MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn={days} THEN close END),0)-1) ret
            FROM (
                SELECT ts_code,close,ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC) rn
                FROM kline_daily WHERE trade_date>='{sd}'
            ) t WHERE rn<={days} GROUP BY ts_code HAVING COUNT(*)>={days-3}
        """).df()

    rets_20=get_fwd_ret(c,20)
    rets_10=get_fwd_ret(c,10)

    # ---------- 1 PE ----------
    print('[1/8] PE...',end=' ',flush=True)
    try:
        df=c.execute("""
            SELECT v.ts_code,v.pe_ttm FROM valuation_daily v
            WHERE v.pe_ttm>0 AND v.pe_ttm<500 AND v.trade_date='2026-06-12'
        """).df()
        if len(df)>100:
            m=df.merge(rets_20,on='ts_code')
            m['factor']=-m['pe_ttm']
            ic,p=rank_ic(m['factor'],m['ret'])
            ly=layered(m['factor'],m['ret'])
            results.append(('PE(低估值)',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 2 ROE ----------
    print('[2/8] ROE...',end=' ',flush=True)
    try:
        df=c.execute("""
            SELECT f.ts_code,f.roe FROM financial_statements f
            WHERE f.roe>0 AND f.roe<100 AND f.report_type='annual'
            AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
        """).df()
        if len(df)>100:
            m=df.merge(rets_20,on='ts_code')
            ic,p=rank_ic(m['roe'],m['ret'])
            ly=layered(m['roe'],m['ret'])
            results.append(('ROE',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 3 净利增速 ----------
    print('[3/8] 净利增速...',end=' ',flush=True)
    try:
        df=c.execute("""
            WITH ranked AS (
                SELECT ts_code,net_profit,report_date,
                       ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY report_date DESC) rn
                FROM financial_statements WHERE net_profit>0 AND report_type='annual'
            )
            SELECT l.ts_code,(l.net_profit-p.net_profit)/NULLIF(p.net_profit,0) growth
            FROM (SELECT * FROM ranked WHERE rn=1) l
            JOIN (SELECT * FROM ranked WHERE rn=2) p ON l.ts_code=p.ts_code
            WHERE l.net_profit>0 AND p.net_profit>0
        """).df()
        if len(df)>100:
            m=df.merge(rets_20,on='ts_code')
            m['factor']=m['growth'].clip(-2,5)
            ic,p=rank_ic(m['factor'],m['ret'])
            ly=layered(m['factor'],m['ret'])
            results.append(('净利增速',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 4 20日动量 ----------
    print('[4/8] 动量...',end=' ',flush=True)
    try:
        df=c.execute(f"""
            SELECT ts_code,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=20 THEN close END),0)-1) momentum
            FROM (
                SELECT ts_code,close,ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC) rn
                FROM kline_daily WHERE trade_date>='{T30}'
            ) t WHERE rn<=20 GROUP BY ts_code HAVING COUNT(*)>=15
        """).df()
        if len(df)>100:
            m=df.merge(rets_10,on='ts_code')
            ic,p=rank_ic(m['momentum'],m['ret'])
            ly=layered(m['momentum'],m['ret'])
            results.append(('20日动量',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 5 波动率 ----------
    print('[5/8] 波动率...',end=' ',flush=True)
    try:
        df=c.execute(f"""
            SELECT ts_code,STDDEV(daily_ret) volatility FROM (
                SELECT ts_code,(close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) daily_ret
                FROM kline_daily WHERE trade_date>='{T30}'
            ) WHERE daily_ret IS NOT NULL
            GROUP BY ts_code HAVING COUNT(*)>=15
        """).df()
        if len(df)>100:
            m=df.merge(rets_10,on='ts_code')
            ic,p=rank_ic(m['volatility'],m['ret'])
            ly=layered(m['volatility'],m['ret'])
            results.append(('波动率',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 6 换手率 ----------
    print('[6/10] 换手率...',end=' ',flush=True)
    try:
        df=c.execute(f"""
            SELECT ts_code,AVG(turnover_rate) turnover
            FROM kline_daily WHERE trade_date>='{T30}' AND turnover_rate>0
            GROUP BY ts_code HAVING COUNT(*)>=15
        """).df()
        if len(df)>100:
            m=df.merge(rets_10,on='ts_code')
            ic,p=rank_ic(m['turnover'],m['ret'])
            ly=layered(m['turnover'],m['ret'])
            results.append(('换手率',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 7 毛利率 ----------
    print('[7/10] 毛利率...',end=' ',flush=True)
    try:
        df=c.execute("""
            SELECT f.ts_code,f.gross_margin FROM financial_statements f
            WHERE f.gross_margin>0 AND f.gross_margin<100 AND f.report_type='annual'
            AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
        """).df()
        if len(df)>100:
            m=df.merge(rets_20,on='ts_code')
            ic,p=rank_ic(m['gross_margin'],m['ret'])
            ly=layered(m['gross_margin'],m['ret'])
            results.append(('毛利率',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print(f'N={len(df)}<100')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 8 北向 ----------
    print('[8/10] 北向...',end=' ',flush=True)
    try:
        nf=c.execute("""
            SELECT AVG(total_flow) f FROM (
                SELECT trade_date,SUM(net_flow) total_flow
                FROM north_bound_flow WHERE net_flow IS NOT NULL AND net_flow!=0
                GROUP BY trade_date ORDER BY trade_date DESC LIMIT 5
            )
        """).fetchone()
        nf_val=nf[0] if nf and nf[0] else 0
        T5=(today-timedelta(days=5)).isoformat()
        rets=c.execute(f"SELECT AVG(change_pct) FROM kline_daily WHERE trade_date>='{T5}' AND ts_code='sh000300'").fetchone()
        idx_ret=rets[0] if rets and rets[0] else 0
        results.append(('北向净流入-' + ('看多' if nf_val>0 else '看空'), round(nf_val/1e10,4), None, None, 5))
        print(f'{nf_val/1e8:.1f}亿 指数{idx_ret:+.2f}%')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 9 VIX ----------
    print('[9/10] VIX...',end=' ',flush=True)
    try:
        df=c.execute("SELECT vix,trade_date FROM macro_indicators WHERE vix IS NOT NULL ORDER BY trade_date DESC LIMIT 10").df()
        if len(df)>=2:
            vix_now=df['vix'].iloc[0]
            vix_chg=vix_now-df['vix'].iloc[-1]
            score=round((20-vix_now)/20,3)
            results.append(('VIX恐慌',score,None,None,len(df)))
            print(f'VIX={vix_now:.1f} chg={vix_chg:+.1f} score={score:+.2f}')
        else: print('数据<2')
    except Exception as e: print(f'ERR:{e}')

    # ---------- 10 行业动量 ----------
    print('[10/10] 行业动量...',end=' ',flush=True)
    try:
        T30=(today-timedelta(days=30)).isoformat()
        df=c.execute(f"""
            WITH ranked AS (
                SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
                FROM proxy_industry_daily WHERE trade_date>='{T30}'
            )
            SELECT industry,
                   MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=11 THEN close END),0)-1 mom
            FROM ranked WHERE rn<=11 GROUP BY industry HAVING COUNT(*)>=10
        """).df()
        if len(df)>10:
            T15=(today-timedelta(days=15)).isoformat()
            rets=c.execute(f"""
                WITH ranked AS (
                    SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
                    FROM proxy_industry_daily WHERE trade_date>='{T15}'
                )
                SELECT industry,
                       MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=6 THEN close END),0)-1 fwd
                FROM ranked WHERE rn<=6 GROUP BY industry
            """).df()
            m=df.merge(rets,on='industry')
            ic,p=rank_ic(m['mom'],m['fwd'])
            ly=layered(m['mom'],m['fwd'])
            results.append(('行业动量',ic,p,ly,len(m)))
            print(f'IC={ic} N={len(m)}')
        else: print('数据<10')
    except Exception as e: print(f'ERR:{e}')

    c.close()

    # ---------- 汇总 ----------
    print('\n'+S55)
    print('  因子IC检验结果')
    print(S55)
    print('  {:<14s} {:>8s} {:>8s} {:>6s} {:>25s}'.format('因子','IC','p值','有效?','分层(Top...Bot)'))
    print('  '+'-'*55)
    valid_count=0
    for name,ic,p,ly,n in results:
        ok='OK' if ic and abs(ic)>0.02 else ('--' if ic and abs(ic)>0.01 else 'XX')
        if ok=='OK': valid_count+=1
        ly_str=''
        if ly:
            vs=[ly.get(f'G{i}',0) for i in range(5)]
            ly_str=f'{vs[0]:+.1f}% -> {vs[-1]:+.1f}%'
        print('  {:<14s} {:>8} {:>8} {:>6s} {:>25s}'.format(name,ic or 'N/A',p or 'N/A',ok,ly_str))
    print('  '+'-'*55)
    print(f'  有效因子: {valid_count}/{len(results)} (|IC|>0.02)')
    print(S55)

if __name__=='__main__':
    main()
