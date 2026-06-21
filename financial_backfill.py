# -*- coding: utf-8 -*-
"""AgentQuant · 财务数据回填 (新浪源) → DuckDB"""
import requests, re, duckdb, time

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
HDR = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/117.0.0.0'}


def parse_sina(html):
    """正则解析新浪财务表 → {字段名: {date: value}}"""
    m = re.search(r'报表日期.*?</td>(.*?)</tr>', html, re.DOTALL)
    if not m:
        return {}
    dates = re.findall(r'<td[^>]*>\s*(\d{4}-\d{2}-\d{2})\s*</td>', m.group(1))
    if not dates:
        return {}

    rows = re.findall(r'<tr>\s*<td[^>]*>(.*?)</td>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
    skip = {'流动资产','非流动资产','流动负债','非流动负债','资产总计','负债和所有者权益',
            '负债及股东权益总计','所有者权益合计','经营活动产生的现金流量',
            '投资活动产生的现金流量','筹资活动产生的现金流量'}

    result = {}
    for raw_name, raw_vals in rows:
        name = re.sub(r'<[^>]+>', '', raw_name).strip()
        if not name or name == '报表日期' or name in skip:
            continue
        vals = re.findall(r'<td[^>]*>\s*([\d.,-]+)\s*</td>', raw_vals)
        if not vals or all(v == '--' for v in vals):
            continue
        fv = {}
        for i, v in enumerate(vals):
            if i >= len(dates): break
            if v != '--':
                fv[dates[i]] = float(v.replace(',', ''))
        if fv:
            result[name] = fv
    return result


def annual_date(fv):
    """取最近年报日期"""
    ad = sorted([d for d in fv if d and '12-31' in d], reverse=True)
    return ad[0] if ad else list(fv.keys())[0] if fv else None


def fetch(code, rtype):
    """拉新浪报表 rtype: balance/cashflow/income"""
    urls = {
        'balance': f'https://money.finance.sina.com.cn/corp/go.php/vFD_BalanceSheet/stockid/{code}/ctrl/part/displaytype/4.phtml',
        'cashflow': f'https://money.finance.sina.com.cn/corp/go.php/vFD_CashFlow/stockid/{code}/ctrl/part/displaytype/4.phtml',
        'income': f'https://money.finance.sina.com.cn/corp/go.php/vFD_ProfitStatement/stockid/{code}/ctrl/part/displaytype/4.phtml',
    }
    r = requests.get(urls[rtype], headers=HDR, timeout=15)
    return parse_sina(r.text)


def backfill(code):
    """回填一只"""
    ts = f'{code}.SH' if code.startswith('6') else f'{code}.SZ'

    try:
        bs = fetch(code, 'balance')
        time.sleep(0.3)
        cf = fetch(code, 'cashflow')
    except Exception as e:
        return f'{code} ERR:{e}'

    if not bs:
        return f'{code} 无数据'

    # 应收账款
    ar, ar_date = None, None
    for k in bs:
        if k.strip() in ('应收账款', '应收票据及应收账款'):
            d = annual_date(bs[k])
            ar, ar_date = bs[k].get(d), d
            break

    # 商誉
    gw = None
    for k in bs:
        if k.strip() == '商誉':
            d = annual_date(bs[k])
            gw = bs[k].get(d)
            break

    # 经营现金流净额
    ocf, ocf_date = None, None
    if cf:
        for k in cf:
            if '经营' in k and '现金流' in k and '净额' in k:
                d = annual_date(cf[k])
                ocf, ocf_date = cf[k].get(d), d
                break

    # 写DuckDB
    c = duckdb.connect(DB)
    n = 0
    rpt = ar_date or ocf_date

    if rpt and (ar is not None or ocf is not None):
        sets = []
        params = []
        if ar is not None: sets.append('accounts_receivable=?'); params.append(ar * 10000)  # 万元→元
        if ocf is not None: sets.append('operating_cf=?'); params.append(ocf * 10000)  # 万元→元
        if sets:
            params.extend([ts, rpt])
            sql = f"UPDATE financial_statements SET {', '.join(sets)} WHERE ts_code=? AND report_date=?"
            c.execute(sql, params)
            # Check if any row was updated; if not, try INSERT
            check = c.execute("SELECT COUNT(*) FROM financial_statements WHERE ts_code=? AND report_date=?",
                              [ts, rpt]).fetchone()
            if check and check[0] == 0:
                cols = ', '.join(['ts_code', 'report_date'] + [s.replace('=?', '') for s in sets])
                phs = ', '.join('?' * (2 + len(sets)))
                c.execute(f"INSERT INTO financial_statements ({cols}) VALUES ({phs})",
                          [ts, rpt] + params[:len(sets)])
            n += 1

    if gw is not None:
        try:
            # 计算商誉占净资产百分比 (需要净资产数据)
            # goodwill_pct = 商誉绝对值 / 净资产 * 100
            # 净资产从已解析的资产负债表中取
            equity = None
            for k in bs:
                if k.strip() in ('归属于母公司股东权益合计', '股东权益合计', '所有者权益合计'):
                    d = annual_date(bs[k])
                    equity = bs[k].get(d)
                    break
            if equity and equity > 0:
                gw_pct = round(gw / equity * 100, 2)
            else:
                gw_pct = None

            c.execute("INSERT INTO goodwill_detail (ts_code, report_date, goodwill_pct) VALUES (?,'2025-12-31',?)",
                      [ts, gw_pct])
            n += 1
        except: pass

    c.commit(); c.close()
    return f'{code}: AR={ar} 商誉={gw} OCF={ocf} ({n} fields)'


def batch(codes, delay=0.5):
    total = 0
    for i, code in enumerate(codes):
        r = backfill(code)
        print(f'  [{i+1}/{len(codes)}] {r}')
        total += 1 if 'ERR' not in r and '无数据' not in r else 0
        if i < len(codes) - 1: time.sleep(delay)
    return total


if __name__ == '__main__':
    print(backfill('300308'))
    print(backfill('688256'))
    print(backfill('601658'))
    # Verify
    c = duckdb.connect(DB, read_only=True)
    for ts in ['300308.SZ','688256.SH','601658.SH']:
        r = c.execute("SELECT accounts_receivable, operating_cf FROM financial_statements WHERE ts_code=? AND report_type='annual' ORDER BY report_date DESC LIMIT 1", [ts]).fetchone()
        print(f'{ts}: AR={r[0]} OCF={r[1]}' if r else f'{ts}: no data')
    c.close()
