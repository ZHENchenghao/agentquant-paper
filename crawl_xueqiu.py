# -*- coding: utf-8 -*-
import sys, io, re, json, time, ssl
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ssl._create_default_https_context = ssl._create_unverified_context
import requests, duckdb, pandas as pd
from datetime import datetime

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

class XueqiuCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        })
        self._get_cookie()

    def _get_cookie(self):
        try:
            r = self.session.get('https://xueqiu.com/', timeout=15)
            print('  cookie: status=%d, cookies=%d' % (r.status_code, len(self.session.cookies)))
        except Exception as e:
            print('  cookie fail: %s' % str(e)[:80])

    def get_stock_timeline(self, symbol, page=1, count=20):
        url = 'https://xueqiu.com/statuses/stock_timeline.json'
        params = {'symbol_id': symbol, 'page': page, 'count': count, 'source': 'web'}
        try:
            r = self.session.get(url, params=params, timeout=15,
                                headers={'Referer': 'https://xueqiu.com/S/%s' % symbol})
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def search_status(self, keyword, page=1, count=20):
        url = 'https://xueqiu.com/statuses/search.json'
        params = {'q': keyword, 'page': page, 'count': count, 'source': 'web'}
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None

    def get_hot_stocks(self):
        url = 'https://xueqiu.com/stock/hot_list.json'
        try:
            r = self.session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
        except:
            pass
        return None


print('=' * 70)
print('Xueqiu retail attention crawler')
print('=' * 70)

crawler = XueqiuCrawler()

# 1. Hot stocks
print('\n[1] Hot stocks...')
hot = crawler.get_hot_stocks()
if hot:
    print('  got hot_list keys: %s' % list(hot.keys())[:5])
    if 'data' in hot and hot['data']:
        for item in hot['data'][:5]:
            print('    %s %s' % (item.get('symbol', ''), item.get('name', '')))

# 2. Stock timeline
print('\n[2] Stock timeline test...')
test_stocks = ['SH600519', 'SZ000001', 'SH688981', 'SZ300750']
for sym in test_stocks:
    data = crawler.get_stock_timeline(sym, page=1, count=5)
    if data and 'list' in data:
        posts = data['list']
        dates = set()
        for post in posts:
            ts = post.get('created_at', 0)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000)
                dates.add(dt.strftime('%Y-%m-%d'))
        print('  %s: %d posts, dates=%s' % (sym, len(posts), sorted(dates)))
        if posts:
            p = posts[0]
            raw_title = p.get('title', '') or p.get('text', '') or ''
            print('    first: [%s] reply=%d retweet=%d' % (
                str(raw_title)[:60],
                p.get('reply_count', 0) or 0,
                p.get('retweet_count', 0) or 0))
    else:
        print('  %s: no data' % sym)
    time.sleep(0.3)

# 3. Search history
print('\n[3] Search history...')
for kw in ['A股', '行情', '牛市']:
    data = crawler.search_status(kw, page=1, count=5)
    if data and 'list' in data:
        posts = data['list']
        dates = set()
        for post in posts:
            ts = post.get('created_at', 0)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000)
                dates.add(dt.strftime('%Y-%m-%d'))
        print('  "%s": %d posts, dates=%s' % (kw, len(posts), sorted(dates)))
        if posts:
            p = posts[0]
            raw = str(p.get('title', '') or p.get('text', ''))
            print('    first: [%s]' % raw[:80])
    else:
        print('  "%s": no data' % kw)
    time.sleep(0.3)

# 4. Top 50 stock attention
print('\n[4] Top 50 A-stock attention...')
con = duckdb.connect(DB, read_only=True)
top50 = con.execute("""
    SELECT ts_code, close * total_share / 10000 AS mcap
    FROM kline_daily
    WHERE trade_date = (SELECT max(trade_date) FROM kline_daily)
    ORDER BY mcap DESC LIMIT 50
""").df()
con.close()

top50['xq_symbol'] = top50['ts_code'].apply(
    lambda x: ('SH' + x[2:]) if x.startswith('sh') else (('SZ' + x[2:]) if x.startswith('sz') else None)
)
top50 = top50[top50['xq_symbol'].notna()]
print('  Top50 by mcap: %d stocks' % len(top50))

stock_attention = []
for i, (_, row) in enumerate(top50.iterrows()):
    sym = row['xq_symbol']
    data = crawler.get_stock_timeline(sym, page=1, count=3)
    if data and 'list' in data:
        posts = data['list']
        total_reply = sum(p.get('reply_count', 0) or 0 for p in posts)
        total_retweet = sum(p.get('retweet_count', 0) or 0 for p in posts)
        stock_attention.append({
            'ts_code': row['ts_code'],
            'xq_symbol': sym,
            'recent_posts': len(posts),
            'total_reply': total_reply,
            'total_retweet': total_retweet,
            'attention_score': total_reply + total_retweet * 2,
        })
    if (i + 1) % 10 == 0:
        print('  %d/%d...' % (i+1, len(top50)))
    time.sleep(0.2)

df_att = pd.DataFrame(stock_attention)
if len(df_att) > 0:
    df_att = df_att.sort_values('attention_score', ascending=False)
    print('\n  Retail attention Top10:')
    for _, row in df_att.head(10).iterrows():
        print('    %-12s posts=%d reply=%d retweet=%d score=%d' % (
            row['ts_code'], row['recent_posts'], row['total_reply'],
            row['total_retweet'], row['attention_score']))
    df_att.to_parquet('cache/xueqiu_attention.parquet')
    print('\n  Saved cache/xueqiu_attention.parquet (%d stocks)' % len(df_att))

print('\nDone.')
