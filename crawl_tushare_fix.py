# -*- coding: utf-8 -*-
"""补爬剩余6张表 — 修复pyarrow类型问题
解决方法: 保存前显式转换所有列为object→不, 转numeric列为float+fillna(0)
"""
import tushare as ts, pandas as pd, numpy as np, time, os, sys, warnings
import requests, urllib3
warnings.filterwarnings('ignore')
urllib3.disable_warnings()

# monkey-patch
_original_post = requests.post
def _patched_post(url, **kwargs):
    kwargs['verify'] = False
    kwargs['proxies'] = {'http': None, 'https': None}
    kwargs.setdefault('timeout', 120)
    return _original_post(url, **kwargs)
requests.post = _patched_post

pro = ts.pro_api('0c55aa67719eafc8b9001cac813ed40b29cee808e9af2700')
pro._DataApi__http_url = 'https://teajoin.com'
pro._DataApi__timeout = 120

OUT = 'D:/AgentQuant/our/cache/ts'
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

def safe_get(func, kwargs, max_retries=5):
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and len(df) > 0:
                # 统一日期列
                for col in ['trade_date','ann_date','end_date','notice_date','report_date',
                           'record_date','ex_date','imp_ann_date']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y%m%d')
                # 数值列: 转float, 空字符串→NaN→0
                for col in df.columns:
                    if col not in ['ts_code','symbol','name','trade_date','ann_date','end_date',
                                   'notice_date','report_date','record_date','ex_date','imp_ann_date',
                                   'type','suspend_type','limit_type','concept_name','industry']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
                return df
            time.sleep(1)
        except Exception as e:
            wait = min((attempt+1)*8, 60)
            if attempt < max_retries-1:
                time.sleep(wait)
    return pd.DataFrame()

def crawl_range(func, date_field, start, end, chunk_days, outfile, **extra_kwargs):
    """分段爬取+增量保存"""
    outpath = os.path.join(OUT, outfile)
    existing = None
    existing_dates = set()
    if os.path.exists(outpath):
        try:
            existing = pd.read_parquet(outpath)
            if date_field in existing.columns:
                existing_dates = set(existing[date_field].astype(str))
            print(f'[{outfile}] 已有 {len(existing)} 行, 续传...')
        except:
            print(f'[{outfile}] 文件损坏, 重新下载')
            os.remove(outpath)

    chunks = []; new_total = 0
    cur = pd.Timestamp(start); end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        ce = min(cur + pd.Timedelta(days=chunk_days-1), end_ts)
        sd = cur.strftime('%Y%m%d'); ed = ce.strftime('%Y%m%d')
        # 简单跳过: 如果该chunk的start date在已有数据里就跳
        if sd in existing_dates:
            cur = ce + pd.Timedelta(days=1)
            continue

        kwargs = {date_field: sd, 'end_date': ed, **extra_kwargs}
        df = safe_get(func, kwargs)
        if len(df) > 0:
            chunks.append(df); new_total += len(df)
        cur = ce + pd.Timedelta(days=1)
        time.sleep(0.2)

        # 每20个chunk或每个年保存
        if len(chunks) >= 20 or (cur.month == 1 and len(chunks) > 0):
            merged = pd.concat(chunks, ignore_index=True)
            if existing is not None and len(existing) > 0:
                merged = pd.concat([existing, merged], ignore_index=True)
            merged.to_parquet(outpath, index=False, engine='pyarrow')
            existing = merged
            if date_field in merged.columns:
                existing_dates = set(merged[date_field].astype(str))
            chunks = []
            print(f'  [save] {len(merged)} 行 ({new_total} new)')

    # final save
    if chunks:
        merged = pd.concat(chunks, ignore_index=True)
        if existing is not None and len(existing) > 0:
            merged = pd.concat([existing, merged], ignore_index=True)
        merged.to_parquet(outpath, index=False, engine='pyarrow')
        existing = merged
        new_total = 0  # all saved

    if existing is not None:
        print(f'  DONE: {len(existing)} 行, {existing[date_field].min()}~{existing[date_field].max()}')
    return existing

# ============================================================
# 按顺序爬取
# ============================================================
print('='*60)
print('补爬开始')
print('='*60)

# 1. margin_detail (续传, 已有9294行到2017)
print('\n[1] margin_detail 续传...')
crawl_range(pro.margin_detail, 'trade_date', '2015-01-01', '2026-06-21', 60,
            'margin_detail_2015_2026.parquet')

# 2. share_float
print('\n[2] share_float...')
crawl_range(pro.share_float, 'ann_date', '2015-01-01', '2026-12-31', 180,
            'share_float_2015_2026.parquet')

# 3. repurchase
print('\n[3] repurchase...')
crawl_range(pro.repurchase, 'ann_date', '2015-01-01', '2026-06-21', 180,
            'repurchase_2015_2026.parquet')

# 4. block_trade
print('\n[4] block_trade...')
crawl_range(pro.block_trade, 'trade_date', '2015-01-01', '2026-06-21', 60,
            'block_trade_2015_2026.parquet')

# 5. forecast (按股票)
print('\n[5] forecast...')
try:
    stocks_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
    stocks = stocks_df['ts_code'].tolist()
    forecasts = []
    for i in range(0, len(stocks), 100):
        batch = stocks[i:i+100]
        ts_str = ','.join(batch)
        df = safe_get(pro.forecast, {
            'ts_code': ts_str, 'ann_date': '20150101', 'end_date': '20260621',
            'fields': 'ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,notice_date,report_date'
        })
        if len(df) > 0: forecasts.append(df)
        if (i//100+1) % 30 == 0: print(f'  {min(i+100,len(stocks))}/{len(stocks)}')
        time.sleep(0.3)
    if forecasts:
        result = pd.concat(forecasts, ignore_index=True)
        result.to_parquet(f'{OUT}/forecast_2015_2026.parquet', index=False)
        print(f'  DONE: {len(result)} 行, {result.ts_code.nunique()}只')
except Exception as e:
    print(f'  SKIP: {e}')

# 6. dividend
print('\n[6] dividend...')
try:
    divs = []
    for i in range(0, len(stocks), 200):
        batch = stocks[i:i+200]
        ts_str = ','.join(batch)
        df = safe_get(pro.dividend, {
            'ts_code': ts_str, 'ann_date': '20150101', 'end_date': '20260621'
        })
        if len(df) > 0: divs.append(df)
        if (i//200+1) % 20 == 0: print(f'  {min(i+200,len(stocks))}/{len(stocks)}')
        time.sleep(0.3)
    if divs:
        result = pd.concat(divs, ignore_index=True)
        result.to_parquet(f'{OUT}/dividend_2015_2026.parquet', index=False)
        print(f'  DONE: {len(result)} 行, {result.ts_code.nunique()}只')
except Exception as e:
    print(f'  SKIP: {e}')

print(f'\n{"="*60}')
print(f'补爬完成! 耗时: {(time.time()-t0)/60:.1f}min')
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f'  {f}: {size_mb:.1f} MB')
