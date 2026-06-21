# -*- coding: utf-8 -*-
"""历史新闻补爬器 · 多源降级
目标: 补爬2023-2026年A股新闻, 至少10000条
源1: 东方财富新闻API (akshare)
源2: 新浪财经新闻
源3: 聚宽/通联数据
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import re

DB_PATH = 'D:/FreeFinanceData/data/duckdb/finance.db'

print('=' * 80)
print('历史新闻补爬器 v1.0')
print('=' * 80)

# ============================================================
# 源1: akshare - 东方财富新闻 (stock_news_em)
# ============================================================
print('\n[源1] akshare stock_news_em...')

try:
    import akshare as ak
    HAS_AK = True
    print('  akshare已安装')
except ImportError:
    HAS_AK = False
    print('  akshare未安装, 尝试pip...')
    os.system('pip install akshare -q')
    try:
        import akshare as ak
        HAS_AK = True
        print('  安装成功')
    except:
        print('  安装失败, 跳过源1')

# ============================================================
# 源2: 东方财富API直连 (比akshare更快, 可翻页)
# ============================================================
print('\n[源2] 东方财富API直连...')

def fetch_eastmoney_news(page=1, pagesize=200):
    """东方财富新闻API, 支持翻页"""
    url = 'https://push2.eastmoney.com/api/qt/ulist.np/get'
    params = {
        'fltt': 2,
        'fields': 'f3,f12,f14,f17,f18',
        'secids': '1.000001,0.399001,0.399006',  # 上证/深证/创业板
        'pn': page,
        'pz': pagesize,
        'po': 1,
        'np': 1,
        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
        'wbp2u': '|0|0|0|web',
        'invt': 2,
    }
    try:
        r = requests.get(url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return r.json()
    except Exception as e:
        return None

# 东方财富新闻搜索API (更直接)
def fetch_em_news_list(keyword='A股', page=1, pagesize=100):
    """东方财富新闻搜索, 按关键词+日期"""
    url = 'https://search-api-web.eastmoney.com/search/jsonp'
    params = {
        'cb': 'jQuery',
        'param': json.dumps({
            'uid': '',
            'keyword': keyword,
            'type': ['8196'],  # 新闻
            'client': 'web',
            'clientType': 'web',
            'clientVersion': 'curr',
            'param': {
                'cateCode': '',
                'startTime': '2023-01-01',
                'endTime': '2026-06-18',
                'pageIndex': page,
                'pageSize': pagesize,
            }
        }),
        '_': int(time.time() * 1000),
    }
    try:
        r = requests.get(url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://so.eastmoney.com/',
        })
        text = r.text
        # 去掉JSONP包装
        match = re.search(r'jQuery\((\{.*\})\)', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except Exception as e:
        return None

# ============================================================
# 源3: 新浪财经新闻API
# ============================================================
print('\n[源3] 新浪财经API...')

def fetch_sina_news(page=1):
    """新浪财经新闻滚动"""
    url = f'https://feed.mix.sina.com.cn/api/roll/get'
    params = {
        'pageid': 153,
        'lid': 2516,  # 要闻
        'k': '',
        'num': 100,
        'page': page,
        'r': str(time.time() * 1000)[:13],
        'callback': 'feed',
        'encode': 'utf-8',
    }
    try:
        r = requests.get(url, params=params, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://finance.sina.com.cn/',
        })
        text = r.text
        match = re.search(r'feed\((.*)\)', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except:
        return None

# ============================================================
# 源4: cls (财联社) API
# ============================================================
print('\n[源4] 财联社电报API...')

def fetch_cls_telegraph(page=1):
    """财联社电报"""
    url = 'https://www.cls.cn/api/sw'
    params = {
        'app': 'CailianpressWeb',
        'os': 'web',
        'sv': '8.4.6',
    }
    data = {
        'type': 'telegraph',
        'page': page,
        'pageSize': 200,
        'rn': int(time.time() * 1000),
    }
    try:
        r = requests.post(url, params=params, json=data, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.cls.cn/telegraph',
        })
        return r.json()
    except Exception as e:
        return None

# ============================================================
# 主爬取逻辑
# ============================================================
print('\n' + '=' * 80)
print('开始爬取...')
print('=' * 80)

all_news = []
sources_ok = 0

# --- 源1: akshare ---
if HAS_AK:
    try:
        print('\n[源1] akshare stock_news_em (最近500条)...')
        df = ak.stock_news_em()
        if len(df) > 0:
            cols_map = {
                '新闻标题': 'title', '新闻内容': 'content', '发布时间': 'publish_time',
                '文章来源': 'source', '新闻链接': 'url'
            }
            df = df.rename(columns={k: v for k, v in cols_map.items() if k in df.columns})
            if 'title' in df.columns:
                df['source'] = df.get('source', '东方财富')
                df['publish_date'] = pd.to_datetime(df.get('publish_time', datetime.now())).dt.strftime('%Y-%m-%d')
                for _, row in df.iterrows():
                    all_news.append({
                        'title': str(row.get('title', '')),
                        'content': str(row.get('content', ''))[:2000],
                        'source': str(row.get('source', '东方财富')),
                        'publish_date': str(row.get('publish_date', '')),
                        'publish_time': str(row.get('publish_time', '')),
                    })
                print(f'  获取{len(df)}条')
                sources_ok += 1
    except Exception as e:
        print(f'  源1失败: {e}')

# --- 源2: 东方财富搜索 (分页爬) ---
print('\n[源2] 东方财富搜索API (爬取历史)...')
keywords = ['A股', '涨停', '跌停', '资金', '业绩', '政策', '利好', '利空', 'IPO', '回购',
            'ETF', '北向资金', '融资融券', '分红', '重组']
fetched_pages = 0
for kw in keywords[:8]:  # 先跑8个关键词
    for page in range(1, 6):  # 每个关键词5页
        try:
            result = fetch_em_news_list(keyword=kw, page=page, pagesize=50)
            if result and 'Data' in result:
                items = result['Data']
                if not items:
                    break
                for item in items:
                    all_news.append({
                        'title': str(item.get('Title', '')),
                        'content': str(item.get('Content', ''))[:2000],
                        'source': '东方财富搜索',
                        'publish_date': str(item.get('ShowDate', '')),
                        'publish_time': str(item.get('Date', '')),
                        'keyword': kw,
                    })
                fetched_pages += 1
                if fetched_pages % 5 == 0:
                    print(f'  已爬{fetched_pages}页, 累计{len(all_news)}条...')
            time.sleep(0.3)  # 礼貌性延迟
        except Exception as e:
            continue
    if fetched_pages >= 40:
        break
print(f'  源2完成: {fetched_pages}页')

if fetched_pages > 0:
    sources_ok += 1

# --- 源3: 新浪滚动 ---
print('\n[源3] 新浪财经滚动...')
sina_count = 0
for page in range(1, 21):  # 20页
    try:
        result = fetch_sina_news(page=page)
        if result and 'result' in result and 'data' in result['result']:
            items = result['result']['data']
            if not items:
                break
            for item in items:
                ctime = item.get('ctime', '')
                all_news.append({
                    'title': str(item.get('title', '')),
                    'content': str(item.get('intro', '')),
                    'source': '新浪财经',
                    'publish_date': ctime[:10] if ctime else '',
                    'publish_time': ctime,
                })
                sina_count += 1
        time.sleep(0.2)
    except:
        break
print(f'  获取{sina_count}条')
if sina_count > 0:
    sources_ok += 1

# --- 源4: 财联社 ---
print('\n[源4] 财联社电报...')
cls_count = 0
for page in range(1, 51):  # 最多50页
    try:
        result = fetch_cls_telegraph(page=page)
        if result and 'data' in result and 'roll_data' in result['data']:
            items = result['data']['roll_data']
            if not items:
                break
            for item in items:
                ctime = item.get('ctime', 0)
                if ctime:
                    dt = datetime.fromtimestamp(ctime)
                    pub_date = dt.strftime('%Y-%m-%d')
                    pub_time = dt.strftime('%H:%M:%S')
                else:
                    pub_date = pub_time = ''
                all_news.append({
                    'title': str(item.get('title', '')),
                    'content': str(item.get('content', ''))[:2000],
                    'source': '财联社',
                    'publish_date': pub_date,
                    'publish_time': pub_time,
                })
                cls_count += 1
        time.sleep(0.15)
    except:
        break
print(f'  获取{cls_count}条')
if cls_count > 0:
    sources_ok += 1

# ============================================================
# 去重 + 入库
# ============================================================
print(f'\n{"="*80}')
print(f'爬取完成: {len(all_news)}条, 来源数={sources_ok}')

if len(all_news) > 0:
    df_news = pd.DataFrame(all_news)
    # 去重 (按标题)
    before = len(df_news)
    df_news = df_news.drop_duplicates(subset=['title'])
    print(f'去重: {before} → {len(df_news)}条')

    # 按日期统计
    if 'publish_date' in df_news.columns:
        df_news['publish_date'] = pd.to_datetime(df_news['publish_date'], errors='coerce')
        date_counts = df_news.groupby(df_news['publish_date'].dt.to_period('M')).size()
        print(f'\n日期范围: {df_news.publish_date.min()} ~ {df_news.publish_date.max()}')
        print(f'按月分布:')
        for period, count in date_counts.items():
            print(f'  {period}: {count}条')

    # 写入DuckDB
    try:
        con_write = duckdb.connect(DB_PATH)
        # 检查已存在的记录
        existing_titles = set()
        try:
            existing = con_write.execute("SELECT title FROM news_articles").fetchall()
            existing_titles = {x[0] for x in existing}
        except:
            pass

        # 过滤已存在的
        new_news = df_news[~df_news['title'].isin(existing_titles)]
        print(f'\n新新闻: {len(new_news)}条 (已存在{len(df_news)-len(new_news)}条)')

        if len(new_news) > 0:
            # 确保列名匹配
            new_news_db = new_news.copy()
            new_news_db = new_news_db.rename(columns={
                'publish_date': 'publish_date',
                'publish_time': 'publish_time',
            })
            # 处理时间格式
            new_news_db['publish_date'] = new_news_db['publish_date'].astype(str)
            new_news_db['publish_time'] = new_news_db['publish_time'].fillna('').astype(str)

            # 插入
            con_write.execute("BEGIN")
            for _, row in new_news_db.iterrows():
                try:
                    con_write.execute("""
                        INSERT INTO news_articles (title, content, source, publish_date, publish_time, sector_tags)
                        VALUES (?, ?, ?, ?, ?, '')
                    """, [
                        str(row.get('title', ''))[:500],
                        str(row.get('content', ''))[:5000],
                        str(row.get('source', ''))[:100],
                        str(row.get('publish_date', ''))[:20],
                        str(row.get('publish_time', ''))[:20],
                    ])
                except:
                    pass
            con_write.execute("COMMIT")
            print(f'✅ 已写入{len(new_news_db)}条新闻到DuckDB')

            # 统计总条数
            total = con_write.execute("SELECT count(*) FROM news_articles").fetchone()[0]
            date_range = con_write.execute("SELECT min(publish_date), max(publish_date) FROM news_articles").fetchone()
            print(f'  DuckDB总计: {total}条, {date_range[0]} ~ {date_range[1]}')
        else:
            print('无新新闻需要写入')

        con_write.close()
    except Exception as e:
        print(f'入库失败: {e}')
        # 保存为parquet备用
        df_news.to_parquet('cache/crawled_news.parquet')
        print(f'已保存到 cache/crawled_news.parquet')

print(f'\n完成! 来源={sources_ok}/4, 总条数={len(all_news)}')
