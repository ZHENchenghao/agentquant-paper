# -*- coding: utf-8 -*-
"""大规模历史新闻爬虫 · 多源+翻页+去重
源1: 新浪财经滚动 (历史可翻页)
源2: 东方财富文章页直爬 (URL含日期)
源3: akshare stock_news_em
目标: 补到至少10000条
"""
import sys, io, os, re, json, hashlib, time, ssl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ssl._create_default_https_context = ssl._create_unverified_context

import requests
import duckdb
import pandas as pd
from datetime import datetime, timedelta

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://finance.sina.com.cn/',
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

all_news = []
total_pages = 0

# ============================================================
# 源1: 新浪财经滚动新闻 (可翻历史页)
# ============================================================
print('=' * 70)
print('[源1] 新浪财经滚动新闻...')

SINA_API = 'https://feed.mix.sina.com.cn/api/roll/get'
# lid=2516=要闻, 可翻到几百页
for page in range(1, 301):  # 最多300页 × 50条 = 15000条
    params = {
        'pageid': 153, 'lid': 2516, 'k': '', 'num': 50,
        'page': page, 'r': str(time.time() * 1000)[:13],
        'callback': 'feed_cb', 'encode': 'utf-8',
    }
    try:
        r = SESSION.get(SINA_API, params=params, timeout=15)
        text = r.text
        # 格式: try{feed_cb({...});}catch(e){};
        cb_pos = text.find('feed_cb(')
        if cb_pos == -1:
            cb_pos = text.find('(')
        else:
            cb_pos = cb_pos + len('feed_cb')
        json_end = text.rfind(');')
        if cb_pos == -1 or json_end == -1:
            continue
        json_text = text[cb_pos+1:json_end]
        data = json.loads(json_text)
        items = data.get('result', {}).get('data', [])
        if not items:
            print(f'  第{page}页: 无数据, 停止')
            break

        page_new = 0
        dates_in_page = set()
        for item in items:
            ctime = item.get('ctime', '')
            title = item.get('title', '').strip()
            intro = item.get('intro', '').strip() if item.get('intro') else ''
            url = item.get('url', '')
            if not title:
                continue
            # ctime是Unix时间戳字符串, 转为日期
            try:
                ts = int(ctime)
                dt = datetime.fromtimestamp(ts)
                pub_date = dt.strftime('%Y-%m-%d')
                pub_time = dt.strftime('%H:%M:%S')
            except:
                pub_date = ctime[:10] if len(ctime) > 10 else str(ctime)
                pub_time = ''
            dates_in_page.add(pub_date)
            all_news.append({
                'title': title[:300],
                'content': intro[:2000],
                'source': '新浪财经',
                'publish_date': pub_date,
                'publish_time': ctime[11:19] if len(ctime) > 11 else '',
                'url': url,
                'keyword': '',
            })
            page_new += 1

        total_pages += 1
        if page <= 3 or page % 50 == 0:
            min_d = min(dates_in_page) if dates_in_page else '?'
            max_d = max(dates_in_page) if dates_in_page else '?'
            print(f'  第{page}页: +{page_new}条, 日期{min_d}~{max_d}')
        time.sleep(0.15)

    except json.JSONDecodeError:
        print(f'  第{page}页: JSON解析失败, 尝试下一页')
        continue
    except Exception as e:
        print(f'  第{page}页: 网络错误 - {str(e)[:80]}')
        time.sleep(2)
        continue

sina_total = len(all_news)
print(f'源1完成: {total_pages}页, 累计{sina_total}条')

# ============================================================
# 源2: akshare stock_news_em (补充最近)
# ============================================================
print(f'\n{"="*70}')
print('[源2] akshare stock_news_em...')
try:
    import akshare as ak
    df = ak.stock_news_em()
    if len(df) > 0:
        added = 0
        for _, row in df.iterrows():
            title = str(row.iloc[0] if len(row) > 0 else '')
            if title and title != 'nan':
                all_news.append({
                    'title': title[:300], 'content': '',
                    'source': '东方财富', 'publish_date': str(datetime.now().date()),
                    'publish_time': '', 'url': '', 'keyword': '',
                })
                added += 1
        print(f'  +{added}条')
except Exception as e:
    print(f'  跳过: {e}')

# ============================================================
# 源3: 东方财富日期URL直爬 (利用URL含日期的特点)
# ============================================================
print(f'\n{"="*70}')
print('[源3] 东方财富日期URL直爬...')
# EastMoney article URLs: https://finance.eastmoney.com/a/YYYYMMDDXXXXXXXX.html
# 尝试爬取指定日期范围内的文章
em_count = 0
# 从2026-01-01回到2024-01-01, 每天尝试
start_date = datetime(2024, 1, 1)
end_date = datetime(2026, 6, 17)
current = end_date
days_tried = 0

# 东方财富每日要闻列表页: https://finance.eastmoney.com/a/cnews_YYYYMMDD.html
# 这个页面格式更规律
while current >= start_date and days_tried < 60:  # 先试60天
    date_str = current.strftime('%Y%m%d')
    url = f'https://finance.eastmoney.com/a/cnews_{date_str}.html'
    try:
        r = SESSION.get(url, headers={**HEADERS, 'Referer': 'https://finance.eastmoney.com/'}, timeout=10)
        if r.status_code == 200 and len(r.text) > 5000:
            # 解析所有文章链接
            found = re.findall(r'/a/(\d{8})\d{6,}\.html', r.text)
            titles = re.findall(r'title="([^"]+)"', r.text)
            if found:
                # 提取标题文本
                clean_titles = [t for t in titles if len(t) > 10 and not t.startswith('http')]
                for i, t in enumerate(clean_titles):
                    pub_d = found[min(i, len(found)-1)]
                    pub_date_f = f'{pub_d[:4]}-{pub_d[4:6]}-{pub_d[6:8]}'
                    all_news.append({
                        'title': t[:300], 'content': '',
                        'source': '东方财富', 'publish_date': pub_date_f,
                        'publish_time': '', 'url': '', 'keyword': '',
                    })
                    em_count += 1
                if days_tried <= 3:
                    print(f'  {current.date()}: +{len(clean_titles)}条 (URL={url})')
        time.sleep(0.2)
    except:
        pass
    current -= timedelta(days=1)
    days_tried += 1

print(f'源3完成: {days_tried}天, +{em_count}条')

# ============================================================
# 去重 + 按日期过滤
# ============================================================
print(f'\n{"="*70}')
print(f'总采集: {len(all_news)}条')

if len(all_news) == 0:
    print('未采集到任何新闻, 退出')
    sys.exit(0)

df_all = pd.DataFrame(all_news)

# 按标题去重
before = len(df_all)
df_all = df_all.drop_duplicates(subset=['title'])
print(f'标题去重: {before} → {len(df_all)}')

# 过滤有效日期
df_all['publish_date'] = pd.to_datetime(df_all['publish_date'], errors='coerce')
df_all = df_all[df_all['publish_date'].notna()]
print(f'有效日期: {len(df_all)}条')

if len(df_all) == 0:
    print('过滤后无数据, 退出')
    sys.exit(0)

# 日期分布
date_range = f'{df_all.publish_date.min().date()} ~ {df_all.publish_date.max().date()}'
print(f'日期范围: {date_range}')

# 按月统计
df_all['ym'] = df_all['publish_date'].dt.to_period('M')
monthly = df_all.groupby('ym').size()
print('按月分布:')
for period, count in monthly.items():
    print(f'  {period}: {count}条')

# ============================================================
# 入库DuckDB
# ============================================================
print(f'\n{"="*70}')
print('入库DuckDB...')

con = duckdb.connect(DB)

# 查已存在
existing = set()
try:
    rows = con.execute("SELECT title FROM news_articles").fetchall()
    existing = {x[0] for x in rows}
except:
    pass

df_new = df_all[~df_all['title'].isin(existing)]
print(f'已存在: {len(existing)}条, 新: {len(df_new)}条')

if len(df_new) > 0:
    batch = []
    for _, row in df_new.iterrows():
        batch.append((
            str(row['title'])[:500],
            str(row.get('content', ''))[:5000],
            str(row.get('source', ''))[:100],
            str(row['publish_date'].date()),
            str(row.get('publish_time', ''))[:20],
            str(row.get('keyword', ''))[:200],
        ))

    con.execute("BEGIN")
    for b in batch:
        try:
            con.execute("""
                INSERT INTO news_articles (title, content, source, publish_date, publish_time, sector_tags)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [b[0], b[1], b[2], b[3], b[4], b[5]])
        except:
            pass
    con.execute("COMMIT")

    # 统计
    total = con.execute("SELECT count(*) FROM news_articles").fetchone()[0]
    dr = con.execute("SELECT min(publish_date), max(publish_date) FROM news_articles").fetchone()
    print(f'✅ DuckDB总计: {total}条, {dr[0]} ~ {dr[1]}')
else:
    print('无新数据')

con.close()
print('\nDone.')
