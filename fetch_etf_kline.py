# -*- coding: utf-8 -*-
"""从Sina API批量下载ETF日线→写入DuckDB etf_daily表"""
import requests, json, time, duckdb, sys
from datetime import datetime

# ETF清单: 宽基 + 行业 + 主题
ETF_LIST = {
    # === 宽基指数ETF(国家队护盘主力) ===
    '510050': ('上证50', 'sh'),
    '510300': ('沪深300', 'sh'),
    '510500': ('中证500', 'sh'),
    '510880': ('红利ETF', 'sh'),
    '510210': ('上证综指', 'sh'),
    '510180': ('上证180', 'sh'),
    '159915': ('创业板', 'sz'),
    '159949': ('创业板50', 'sz'),
    '159922': ('中证500', 'sz'),
    '588000': ('科创50', 'sh'),
    '512100': ('中证1000', 'sh'),
    '512880': ('证券ETF', 'sh'),
    '512800': ('银行ETF', 'sh'),
    '512760': ('芯片ETF', 'sh'),
    '512690': ('酒ETF', 'sh'),
    '512660': ('军工ETF', 'sh'),
    '512480': ('半导体ETF', 'sh'),
    '512170': ('医疗ETF', 'sh'),
    '512010': ('医药ETF', 'sh'),
    '510900': ('H股ETF', 'sh'),
    '515050': ('5GETF', 'sh'),
    '515790': ('光伏ETF', 'sh'),
    '515700': ('新能源汽车ETF', 'sh'),
    '516160': ('新能源ETF', 'sh'),
    '159995': ('芯片ETF', 'sz'),
    '159996': ('家电ETF', 'sz'),
    '159865': ('养殖ETF', 'sz'),
    '159766': ('消费电子ETF', 'sz'),
    '159781': ('科创创业ETF', 'sz'),
    '159611': ('电力ETF', 'sz'),
    '159619': ('基建ETF', 'sz'),
    '159666': ('石化ETF', 'sz'),
    '159632': ('煤炭ETF', 'sz'),
    '159638': ('稀土ETF', 'sz'),
    '159625': ('互联网ETF', 'sz'),
    '159650': ('国企共赢ETF', 'sz'),
    '159682': ('创业板50ETF', 'sz'),
    '510050': ('上证50', 'sh'),
}

# 去重
seen = set()
unique_etfs = {}
for code, (name, market) in ETF_LIST.items():
    if code not in seen:
        seen.add(code)
        unique_etfs[code] = (name, market)

print("=" * 60)
print(f"ETF日线数据下载 · {len(unique_etfs)}只")
print("=" * 60)

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://finance.sina.com.cn/',
})

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
con = duckdb.connect(DB)

# 建表
con.execute("""
    CREATE TABLE IF NOT EXISTS etf_daily (
        ts_code VARCHAR,
        trade_date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume DOUBLE,
        name VARCHAR,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (ts_code, trade_date)
    )
""")

total_rows = 0
for i, (code, (name, market)) in enumerate(sorted(unique_etfs.items())):
    prefix = 'sh' if market == 'sh' else 'sz'
    sina_sym = f'{prefix}{code}'
    ts_code = f'{code}.{"SH" if market=="sh" else "SZ"}'

    try:
        url = f'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={sina_sym}&scale=240&ma=no&datalen=5000'
        r = session.get(url, timeout=30)
        data = json.loads(r.text)

        if not data or not isinstance(data, list):
            print(f"  [{i+1}/{len(unique_etfs)}] {ts_code} {name}: 无数据 (resp={str(r.text)[:100]})")
            continue

        # 先删旧数据再插入
        con.execute("DELETE FROM etf_daily WHERE ts_code=?", [ts_code])

        rows = []
        for d in data:
            try:
                rows.append((
                    ts_code,
                    d['day'],
                    float(d['open']),
                    float(d['high']),
                    float(d['low']),
                    float(d['close']),
                    float(d['volume']),
                    name
                ))
            except (KeyError, ValueError):
                continue

        if rows:
            con.executemany(
                "INSERT INTO etf_daily (ts_code, trade_date, open, high, low, close, volume, name) VALUES (?,?,?,?,?,?,?,?)",
                rows
            )
            total_rows += len(rows)
            print(f"  [{i+1}/{len(unique_etfs)}] {ts_code} {name}: {len(rows)}行 ({data[0]['day']}~{data[-1]['day']})")
        else:
            print(f"  [{i+1}/{len(unique_etfs)}] {ts_code} {name}: 解析0行")

    except Exception as e:
        print(f"  [{i+1}/{len(unique_etfs)}] {ts_code} {name}: FAIL {str(e)[:80]}")

    time.sleep(0.3)  # 礼貌爬取

con.commit()

# 验证
cnt = con.execute("SELECT COUNT(*) FROM etf_daily").fetchone()[0]
etf_cnt = con.execute("SELECT COUNT(DISTINCT ts_code) FROM etf_daily").fetchone()[0]
latest = con.execute("SELECT MAX(trade_date) FROM etf_daily").fetchone()[0]
print(f"\n总计: {cnt}行 / {etf_cnt}只ETF / 最新日期: {latest}")
con.close()
print("Done.")
