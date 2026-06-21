# -*- coding: utf-8 -*-
"""
Valuation History — 直接调东方财富K线API (含PE/PB字段)
逐日批量: 一次请求获取当日全市场估值数据
"""
import sys,io,os,time,json; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb,pandas as pd,numpy as np,requests,re
from datetime import date,timedelta; import warnings; warnings.filterwarnings('ignore')
import akshare as ak

DB='D:/FreeFinanceData/data/duckdb/finance.db'
SLEEP=4

HEADERS={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer':'https://quote.eastmoney.com/',
}

# 建表
con=duckdb.connect(DB)
con.execute('''CREATE TABLE IF NOT EXISTS valuation_daily(ts_code VARCHAR,trade_date DATE,pe_ttm DOUBLE,pb DOUBLE,total_mv DOUBLE,float_mv DOUBLE,PRIMARY KEY(ts_code,trade_date))''')

# 交易日
existing=con.execute("SELECT DISTINCT trade_date FROM valuation_daily WHERE pe_ttm IS NOT NULL").df()
existing_dates=set(str(d)[:10] for d in existing['trade_date'].values)
con.close()

cal=ak.tool_trade_date_hist_sina()
cal_dates=sorted(cal['trade_date'].astype(str).values)
cal_dates=[d for d in cal_dates if '2010'<=d[:4]<='2026' and d not in existing_dates]
print(f'已有:{len(existing_dates)}天 待采:{len(cal_dates)}天 预计{len(cal_dates)*SLEEP/3600:.1f}小时')

# 测试: 先试最近5天
test_dates=cal_dates[-5:]
print(f'先试: {test_dates}')

for i,td in enumerate(test_dates):
    try:
        # 东方财富日K线API + 估值字段
        # secid: 1.000001 (上交所) 0.000001 (深交所)
        # klt=101 日线, fqt=1 前复权
        # 额外字段 f9=PE f17=PB f20=总市值 f21=流通市值
        # 全市场不能一次请求, 需要分交易所/分页

        # 方法: 用沪深300列表, 取所有成分股当日的估值
        # 获取全市场股票列表
        stocks=ak.stock_zh_a_spot_em()
        codes=stocks['代码'].tolist()
        print(f'{td}: {len(codes)}只股票')

        # 分批次取: 每批50只
        batch_size=50
        records=[]
        for j in range(0,len(codes),batch_size):
            batch=codes[j:j+batch_size]
            # 构造secid列表
            secids=[]
            for c in batch:
                if c.startswith('6'): secids.append(f'1.{c}')
                else: secids.append(f'0.{c}')

            # API请求
            url='https://push2his.eastmoney.com/api/qt/stock/kline/get'
            params={
                'secid':','.join(secids),
                'fields1':'f1,f2,f3,f4,f5,f6',
                'fields2':'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f9,f17,f20,f21',
                'klt':'101','fqt':'1',
                'beg':td.replace('-',''),'end':td.replace('-',''),
                'lmt':'100',
            }
            r=requests.get(url,params=params,headers=HEADERS,timeout=30)
            if r.status_code!=200:
                time.sleep(2)
                continue

            data=r.json()
            if data and data.get('data'):
                d=data['data']
                if isinstance(d,dict) and 'klines' in d:
                    klines=d['klines']
                elif isinstance(d,list):
                    klines=d
                else:
                    klines=[]

                # 处理返回的k线数据
                # 格式: 日期,开,收,高,低,量,额,振幅,涨跌幅,涨跌额,换手率,PE,PB,总市值,流通市值
                for kline in klines:
                    if isinstance(kline,str):
                        parts=kline.split(',')
                    else:
                        continue

                    # 从返回的secid对应的股票代码获取
                    # 这里需要从数据结构中解析

            time.sleep(0.1)  # 批次间小间隔

        if records:
            con=duckdb.connect(DB)
            df_b=pd.DataFrame(records,columns=['ts_code','trade_date','pe_ttm','pb','total_mv','float_mv'])
            con.execute('INSERT OR REPLACE INTO valuation_daily SELECT * FROM df_b')
            con.close()

        print(f'  {td}: {len(records)}条')
    except Exception as e:
        print(f'  {td}: ERR {str(e)[:60]}')

    time.sleep(SLEEP)

print('Done.')
