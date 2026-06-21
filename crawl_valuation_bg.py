# -*- coding: utf-8 -*-
"""后台估值爬虫: PE/PB/total_mv历史 → DuckDB"""
import sys,io,os,time; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb,pandas as pd; import akshare as ak; import warnings; warnings.filterwarnings('ignore')

DB='D:/FreeFinanceData/data/duckdb/finance.db'
SLEEP=3; MAX_RETRY=3

con=duckdb.connect(DB)
con.execute('''CREATE TABLE IF NOT EXISTS valuation_daily(ts_code VARCHAR,trade_date DATE,pe_ttm DOUBLE,pb DOUBLE,total_mv DOUBLE,PRIMARY KEY(ts_code,trade_date))''')

# Get stock list
stocks=ak.stock_zh_a_spot_em()['代码'].tolist()
print(f'Stock count: {len(stocks)}')

# Check existing
existing=con.execute("SELECT DISTINCT ts_code FROM valuation_daily").df()
done_codes=set(existing['ts_code'].values)
pending=[c for c in stocks if (f'sh{c}' if c.startswith('6') else f'sz{c}') not in done_codes]
print(f'Done: {len(done_codes)}  Pending: {len(pending)}')
con.close()

if not pending:
    print('All done.')
    sys.exit(0)

succ=fail=0; t0=time.time()
for i,code in enumerate(pending):
    ts=f'sh{code}' if code.startswith('6') else f'sz{code}'
    for retry in range(MAX_RETRY):
        try:
            # Try Baidu valuation
            df=ak.stock_zh_valuation_baidu(symbol=code, indicator='总市值')
            if df is not None and len(df)>0:
                records=[]
                for _,row in df.iterrows():
                    # Columns: date, value
                    try:
                        d=str(row.iloc[0])[:10]
                        v=float(row.iloc[1]) if pd.notna(row.iloc[1]) else None
                        if v: records.append((ts,d,None,None,v))
                    except: pass
                if records:
                    con=duckdb.connect(DB)
                    con.executemany('INSERT OR IGNORE INTO valuation_daily VALUES(?,?,?,?,?)', records)
                    con.close()
            succ+=1
            break
        except Exception as e:
            if retry==MAX_RETRY-1: fail+=1
            time.sleep(5)

    if (i+1)%100==0:
        e=time.time()-t0; eta=e/(i+1)*(len(pending)-i-1)
        print(f'{i+1}/{len(pending)} succ={succ} fail={fail} ETA={eta/60:.0f}min')
    time.sleep(SLEEP)

print(f'Done. succ={succ} fail={fail}')
