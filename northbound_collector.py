# -*- coding: utf-8 -*-
"""
AgentQuant · 北向资金修复
=========================
同花顺 hexin.cn 源 (东财全系2024-08后NaN, 已废弃)
本地CSV自缓存 → DuckDB north_bound_flow
"""
import requests, duckdb, pandas as pd
from pathlib import Path
from datetime import date, datetime

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE_DIR = Path.home() / '.tradingagents' / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HSGT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0 Safari/537.36",
    "Host": "data.hexin.cn",
    "Referer": "https://data.hexin.cn/",
}


def fetch_hsgt_realtime():
    """沪深股通当日实时分钟流向 (同花顺 hexin.cn)"""
    url = "https://data.hexin.cn/market/hsgtApi/method/dayChart/"
    r = requests.get(url, headers=HSGT_HEADERS, timeout=10)
    d = r.json()
    times = d.get("time", [])
    hgt = d.get("hgt", [])
    sgt = d.get("sgt", [])
    n = len(times)
    return pd.DataFrame({
        "time": times,
        "hgt_yi": hgt[:n] if len(hgt) == n else hgt + [None] * (n - len(hgt)),
        "sgt_yi": sgt[:n] if len(sgt) == n else sgt + [None] * (n - len(sgt)),
    })


def get_daily_snapshot():
    """从实时数据提取当日终值 (收盘时刻的累计净买入)"""
    df = fetch_hsgt_realtime()
    if df.empty:
        return None

    # 取最后一个有效数据点 (收盘净买入)
    valid = df.dropna(subset=['hgt_yi', 'sgt_yi'])
    if valid.empty:
        return None

    last = valid.iloc[-1]
    return {
        'date': date.today().isoformat(),
        'hgt_yi': float(last['hgt_yi']),  # 沪股通累计净买入(亿)
        'sgt_yi': float(last['sgt_yi']),  # 深股通累计净买入(亿)
    }


def save_to_cache(snapshot: dict):
    """写入本地CSV缓存"""
    path = CACHE_DIR / 'northbound_daily.csv'
    rows = {}
    if path.exists():
        for line in path.read_text().strip().split('\n')[1:]:
            parts = line.split(',')
            if len(parts) == 3:
                rows[parts[0]] = line
    rows[snapshot['date']] = f"{snapshot['date']},{snapshot['hgt_yi']},{snapshot['sgt_yi']}"
    with open(path, 'w') as f:
        f.write('date,hgt_yi,sgt_yi\n')
        for d in sorted(rows.keys()):
            f.write(rows[d] + '\n')
    return path


def load_history(n: int = 90):
    """读取缓存历史"""
    path = CACHE_DIR / 'northbound_daily.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).tail(n)


def sync_to_duckdb():
    """将CSV缓存同步到DuckDB north_bound_flow表
    注意: 同花顺API返回累计净买入, 需转换为每日净流入
    """
    hist = load_history(365 * 3)
    if hist.empty:
        print("No cached northbound data to sync")
        return 0

    c = duckdb.connect(DB)
    new = 0

    # 计算每日净流入 (累计值差分)
    hist = hist.sort_values('date').reset_index(drop=True)
    for i, row in hist.iterrows():
        d = str(row['date'])[:10]
        cum = float(row['hgt_yi']) + float(row['sgt_yi'])

        if i == 0:
            # 第一天: 累计值本身就是净流入 (假设从0开始)
            net = cum
        else:
            prev_cum = float(hist.iloc[i-1]['hgt_yi']) + float(hist.iloc[i-1]['sgt_yi'])
            net = cum - prev_cum  # 差分: 今日净流入

        # 检查是否已存在非零数据
        r = c.execute("SELECT net_flow FROM north_bound_flow WHERE trade_date=? AND net_flow != 0", [d]).fetchone()
        if r:
            continue  # 已有真实数据, 不覆盖

        # 删除旧的零值行, 写入真实数据
        c.execute("DELETE FROM north_bound_flow WHERE ts_code='NORTH' AND trade_date=?", [d])
        c.execute("INSERT INTO north_bound_flow (ts_code, trade_date, net_flow) VALUES ('NORTH', ?, ?)", [d, round(net, 2)])
        new += 1

    c.commit()
    c.close()
    return new


def run():
    """采集一次北向快照"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 北向采集开始 (同花顺hexin.cn)")
    snap = get_daily_snapshot()
    if snap is None:
        print("  无有效数据 (可能非交易时间)")
        return

    path = save_to_cache(snap)
    print(f"  缓存: {path}")

    # 显示今日
    hist = load_history(5)
    print(f"  最近5日:")
    for _, r in hist.iterrows():
        print(f"    {r['date']}: 沪+{r['hgt_yi']:.1f}亿  深+{r['sgt_yi']:.1f}亿  合计+{r['hgt_yi']+r['sgt_yi']:.1f}亿")

    # 同步到DuckDB
    n = sync_to_duckdb()
    print(f"  同步DuckDB: +{n}条")

    # 验证
    c = duckdb.connect(DB, read_only=True)
    r = c.execute("""
        SELECT trade_date, net_flow FROM north_bound_flow
        WHERE net_flow != 0 ORDER BY trade_date DESC LIMIT 5
    """).fetchall()
    c.close()
    if r:
        print(f"  DuckDB最新非零北向: {r[0][0]} {r[0][1]:+.1f}亿")
    else:
        print(f"  DuckDB: 全零 (历史数据需要积累)")


if __name__ == '__main__':
    run()
