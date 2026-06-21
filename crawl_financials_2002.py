# -*- coding: utf-8 -*-
"""批量爬取A股历史财报数据 (2002-2015), 补充financial_statements表"""
import sys,io,os,time,duckdb
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')

DB='D:/FreeFinanceData/data/duckdb/finance.db'

def get_stock_list():
    c=duckdb.connect(DB,read_only=True)
    existing=c.execute("SELECT COUNT(DISTINCT ts_code),MIN(report_date),MAX(report_date) FROM financial_statements").fetchone()
    print('现有财报: %d条 %s~%s'%existing)

    # 取所有股票代码(从kline)
    stocks=c.execute("""
        SELECT DISTINCT
            CASE WHEN ts_code LIKE 'sh%' THEN REPLACE(ts_code,'sh','')||'.SH'
                 WHEN ts_code LIKE 'sz%' THEN REPLACE(ts_code,'sz','')||'.SZ'
                 WHEN ts_code LIKE 'bj%' THEN REPLACE(ts_code,'bj','')||'.BJ'
                 WHEN ts_code LIKE '%.%' THEN ts_code
            END AS code
        FROM kline_daily WHERE trade_date>='2002-01-01'
    """).fetchall()
    c.close()
    return [s[0] for s in stocks if s[0] and '.' in s[0]]

def crawl_one(code, retries=3):
    """爬单只股票的财务数据"""
    import akshare as ak
    raw=code.split('.')[0]
    for attempt in range(retries):
        try:
            df=ak.stock_financial_abstract_ths(symbol=raw,indicator='按报告期')
            if df is None or df.empty:return None
            # 只取2002-2015年的年报
            df=df[df['报告期'].str.contains('-12-31',na=False)]
            df=df[(df['报告期']>='2002-12-31')&(df['报告期']<'2016-01-01')]
            if df.empty:return None
            # 映射列
            result=[]
            for _,row in df.iterrows():
                np_val=row.get('净利润','')
                eps_val=row.get('基本每股收益','')
                rev_val=row.get('营业总收入','')
                gm_val=row.get('销售毛利率','')
                nm_val=row.get('销售净利率','')
                roe_val=row.get('净资产收益率','')
                bvps_val=row.get('每股净资产','')

                def parse_num(v):
                    if v is None or v=='' or v=='False' or v==False:return None
                    s=str(v).replace(',','').replace('亿','e8').replace('万','e4').replace('%','')
                    try:return float(s)
                    except:return None

                result.append({
                    'ts_code':code,'report_date':str(row['报告期'])[:10],'report_type':'annual',
                    'net_profit':parse_num(np_val),'eps':parse_num(eps_val),
                    'revenue':parse_num(rev_val),'gross_margin':parse_num(gm_val),
                    'net_margin':parse_num(nm_val),'roe':parse_num(roe_val),
                    'shareholders_equity':parse_num(bvps_val)*parse_num(eps_val) if parse_num(bvps_val) and parse_num(eps_val) else None,
                    'data_source':'akshare_ths_2002',
                })
            return result
        except Exception as e:
            if attempt==retries-1:return None
            time.sleep(2)
    return None

def batch_crawl(stock_list, batch_size=100):
    """批量爬取+入库"""
    c=duckdb.connect(DB)
    total=len(stock_list)
    inserted=0;failed=0;skipped=0

    # 先查哪些股票已经有2002-2015数据
    has_data=set()
    existing=c.execute("SELECT DISTINCT ts_code FROM financial_statements WHERE report_date<'2016-01-01'").fetchall()
    has_data=set(e[0] for e in existing)
    print('已有%d只股票的历史数据'%len(has_data))

    buffer=[]
    for i,code in enumerate(stock_list):
        if code in has_data:
            skipped+=1;continue
        if i>0 and i%50==0:
            print('  %d/%d 已插入%d 失败%d 跳过%d'%(i,total,inserted,failed,skipped))

        rows=crawl_one(code)
        if rows is None:
            failed+=1;continue
        buffer.extend(rows)

        if len(buffer)>=batch_size:
            try:
                import pandas as pd
                df=pd.DataFrame(buffer)
                c.execute("INSERT INTO financial_statements SELECT * FROM df")
                c.execute("COMMIT" if hasattr(c,'commit') else "SELECT 1")
                inserted+=len(buffer);buffer=[]
            except Exception as e:
                print('  DB写入失败: %s'%str(e)[:80])
                failed+=len(buffer);buffer=[]

        if i>0 and i%20==0:time.sleep(0.5)

    # 剩余
    if buffer:
        import pandas as pd
        df=pd.DataFrame(buffer)
        try:
            c.execute("INSERT INTO financial_statements SELECT * FROM df")
            inserted+=len(buffer)
        except:failed+=len(buffer)

    c.close()
    print('完成: 插入%d 失败%d 跳过%d'%(inserted,failed,skipped))
    return inserted,failed,skipped

if __name__=='__main__':
    print('获取股票列表...');stocks=get_stock_list()
    print('共%d只股票'%len(stocks))
    print('开始爬取(预计%d分钟)...'%max(1,len(stocks)//20))
    batch_crawl(stocks,batch_size=200)
