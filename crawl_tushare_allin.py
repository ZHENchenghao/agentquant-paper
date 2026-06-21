# -*- coding: utf-8 -*-
"""Tushare全量日频数据爬取 + 第2轮非日频补齐
外循环: 年份 (2005→2026)
内循环: 每年分4段(90天), 每个chunk重试5次
增量保存: 每段写完立刻落盘, 中断可续
"""
import tushare as ts, pandas as pd, numpy as np, time, os, sys, warnings
import requests, urllib3
warnings.filterwarnings('ignore')
urllib3.disable_warnings()
t0 = time.time()
print('=== Tushare全量爬取启动 ===', flush=True)
print(f'开始时间: {pd.Timestamp.now()}', flush=True)

OUT = 'D:/AgentQuant/our/cache/ts'
os.makedirs(OUT, exist_ok=True)

# Bypass system proxy (127.0.0.1:15011) + SSL verify off
# tushare uses requests.post() directly, so monkey-patch at module level
_original_post = requests.post
def _patched_post(url, **kwargs):
    kwargs['verify'] = False
    kwargs['proxies'] = {'http': None, 'https': None}  # bypass system proxy
    kwargs.setdefault('timeout', 120)
    return _original_post(url, **kwargs)
requests.post = _patched_post
urllib3.disable_warnings()

pro = ts.pro_api('0c55aa67719eafc8b9001cac813ed40b29cee808e9af2700')
pro._DataApi__http_url = 'https://teajoin.com'
pro._DataApi__timeout = 120

# ============================================================
# 配置: 每张表, 起始年, chunk天数
# ============================================================
TABLES = [
    # (表名, 函数, 参数模板, 起始年, chunk天数, 输出文件)
    ('daily_basic', pro.daily_basic,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed,
        'fields': 'ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'},
     2005, 60, 'daily_basic_2005_2026.parquet'),

    ('moneyflow', pro.moneyflow,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2010, 60, 'moneyflow_2010_2026.parquet'),

    ('adj_factor', pro.adj_factor,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2005, 90, 'adj_factor_2005_2026.parquet'),

    ('limit_list', pro.limit_list,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2005, 90, 'limit_list_2005_2026.parquet'),

    ('suspend_d', pro.suspend_d,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2005, 180, 'suspend_d_2005_2026.parquet'),

    # 第2轮: 非日频表
    ('margin_detail', pro.margin_detail,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2015, 60, 'margin_detail_2015_2026.parquet'),

    ('share_float', pro.share_float,
     lambda sd, ed: {'ann_date': sd, 'end_date': ed},
     2015, 180, 'share_float_2015_2026.parquet'),

    ('repurchase', pro.repurchase,
     lambda sd, ed: {'ann_date': sd, 'end_date': ed},
     2015, 180, 'repurchase_2015_2026.parquet'),

    ('block_trade', pro.block_trade,
     lambda sd, ed: {'trade_date': sd, 'end_date': ed},
     2015, 60, 'block_trade_2015_2026.parquet'),
]

# forecast 和 dividend 需要股票级别循环, 单独处理
STOCK_LEVEL_TABLES = [
    ('forecast', pro.forecast,
     lambda ts_codes: {'ts_code': ts_codes, 'ann_date': '20150101', 'end_date': '20260621',
        'fields': 'ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,notice_date,report_date'},
     100, 'forecast_2015_2026.parquet'),

    ('dividend', pro.dividend,
     lambda ts_codes: {'ts_code': ts_codes, 'ann_date': '20150101', 'end_date': '20260621'},
     200, 'dividend_2015_2026.parquet'),
]

def safe_get(func, kwargs, max_retries=5):
    """带指数退避的重试"""
    for attempt in range(max_retries):
        try:
            df = func(**kwargs)
            if df is not None and len(df) > 0:
                # Normalize date columns to string to avoid type conflicts
                for col in ['trade_date', 'ann_date', 'end_date', 'notice_date', 'report_date', 'record_date', 'ex_date', 'imp_ann_date']:
                    if col in df.columns:
                        df[col] = df[col].astype(str)
                return df
            time.sleep(1)
        except Exception as e:
            wait = min((attempt + 1) * 8, 60)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                err = str(e)[:80]
                print(f'    [FAIL after {max_retries}x] {err}')
    return pd.DataFrame()

# ============================================================
# Phase 1: 日期分段表
# ============================================================
now = pd.Timestamp.now()
end_date = now.strftime('%Y%m%d')

for tbl_name, func, make_kwargs, start_year, chunk_days, outfile in TABLES:
    outpath = os.path.join(OUT, outfile)

    # 检查已有数据, 断点续传
    existing_dates = set()
    existing = None
    date_col = 'trade_date'
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        # Normalize date columns to string for safe concat later
        for col in ['trade_date', 'ann_date', 'end_date', 'notice_date', 'report_date', 'record_date', 'ex_date', 'imp_ann_date']:
            if col in existing.columns:
                existing[col] = pd.to_datetime(existing[col]).dt.strftime('%Y%m%d')
        print(f'\n{"="*60}')
        print(f'[{tbl_name}] 已有 {len(existing)} 行, 跳过已有日期...')
        # 提取已有日期范围
        if 'trade_date' in existing.columns:
            date_col = 'trade_date'
            existing_dates = set(existing['trade_date'])
        elif 'ann_date' in existing.columns:
            date_col = 'ann_date'
            existing_dates = set(existing['ann_date'])
        print(f'  已有 {len(existing_dates)} 个日期')
    else:
        print(f'\n{"="*60}')
        print(f'[{tbl_name}] 全新下载 {start_year}→{now.year}')

    all_chunks = []
    new_rows = 0
    for year in range(start_year, now.year + 1):
        cur = pd.Timestamp(f'{year}-01-01')
        year_end = min(pd.Timestamp(f'{year}-12-31'), now)
        if year == now.year:
            year_end = now

        # 跳过全年的周末
        if cur > year_end:
            continue

        while cur <= year_end:
            chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), year_end)
            sd_str = cur.strftime('%Y%m%d')
            ed_str = chunk_end.strftime('%Y%m%d')

            # 检查这个chunk是否已有 (简单策略: 开始日期已覆盖就跳过)
            if sd_str in existing_dates:
                cur = chunk_end + pd.Timedelta(days=1)
                continue

            kwargs = make_kwargs(sd_str, ed_str)
            df = safe_get(func, kwargs)
            if len(df) > 0:
                all_chunks.append(df)
                new_rows += len(df)
                # 每10个chunk保存一次
                if len(all_chunks) % 10 == 0:
                    merged = pd.concat(all_chunks, ignore_index=True)
                    if os.path.exists(outpath):
                        merged = pd.concat([existing, merged], ignore_index=True)
                    merged.to_parquet(outpath, index=False)
                    print(f'  [save] {len(merged)} 行 (本段+{new_rows})')
                    all_chunks = []
                    existing = merged
                    # 刷新existing_dates (不改动existing列类型)
                    date_col = 'trade_date' if 'trade_date' in existing.columns else 'ann_date'
                    if date_col in existing.columns:
                        temp_dates = pd.to_datetime(existing[date_col])
                        existing_dates = set(temp_dates.dt.strftime('%Y%m%d'))

            cur = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.25)

        elapsed = time.time() - t0
        print(f'  {year}: {new_rows}行 | 累计 {elapsed/60:.1f}min')

    # 最终保存
    if all_chunks:
        merged = pd.concat(all_chunks, ignore_index=True)
        if os.path.exists(outpath):
            existing = pd.read_parquet(outpath)
            for col in ['trade_date', 'ann_date']:
                if col in existing.columns and col in merged.columns:
                    existing[col] = pd.to_datetime(existing[col]).dt.strftime('%Y%m%d')
            merged = pd.concat([existing, merged], ignore_index=True)
        merged.to_parquet(outpath, index=False)
        print(f'[FINAL] {outfile}: {len(merged)} 行')

    elapsed = time.time() - t0
    print(f'  耗时: {elapsed/60:.1f}min')

# ============================================================
# Phase 2: 股票级别表 (forecast, dividend)
# ============================================================
print(f'\n{"="*60}')
print('Phase 2: 股票级别循环表')
print('='*60)

try:
    stocks_df = pro.stock_basic(exchange='', list_status='L', fields='ts_code')
    all_stocks = stocks_df['ts_code'].tolist()
    print(f'股票池: {len(all_stocks)}只')
except:
    all_stocks = []
    print('获取股票列表失败')

for tbl_name, func, make_kwargs, chunk_size, outfile in STOCK_LEVEL_TABLES:
    outpath = os.path.join(OUT, outfile)
    if os.path.exists(outpath):
        existing = pd.read_parquet(outpath)
        print(f'[{tbl_name}] 已有 {len(existing)} 行, 跳过')
        continue

    print(f'\n[{tbl_name}] 按股票爬取 (每批{chunk_size}只)...')
    all_data = []
    for i in range(0, len(all_stocks), chunk_size):
        batch = all_stocks[i:i+chunk_size]
        ts_codes = ','.join(batch)
        kwargs = make_kwargs(ts_codes)
        df = safe_get(func, kwargs)
        if len(df) > 0:
            all_data.append(df)
        if (i // chunk_size + 1) % 30 == 0:
            print(f'  {min(i+chunk_size, len(all_stocks))}/{len(all_stocks)}')
        time.sleep(0.3)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result.to_parquet(outpath, index=False)
        print(f'[{tbl_name}] 已保存: {len(result)} 行')

# ============================================================
# 汇总
# ============================================================
elapsed = time.time() - t0
print(f'\n{"="*60}')
print(f'全部完成! 耗时: {elapsed/60:.1f}min')
print('='*60)
for f in sorted(os.listdir(OUT)):
    path = os.path.join(OUT, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f'  {f}: {size_mb:.1f} MB')
