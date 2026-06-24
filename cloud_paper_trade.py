# -*- coding: utf-8 -*-
"""云端纸交引擎 (GitHub Actions) — 零外部API依赖, 纯缓存+HTTP"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

TODAY = date.today()
DIR = os.path.dirname(os.path.abspath(__file__))
PF_FILE = os.path.join(DIR, 'paper_portfolio.json')
LOG_FILE = os.path.join(DIR, 'paper_daily_log.md')
CAPITAL = 100000; TOP_N = 20

print(f'=== 云端纸交 {TODAY.isoformat()} ===')

# ── 1. 尝试获取HS300最新价 (HTTP降级) ──
HS300_CLOSE = 4943; DD_2Y = -0.023
try:
    import urllib.request
    url = 'https://push2.eastmoney.com/api/qt/stock/get?secid=1.000300&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f116,f117,f169,f170'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        if data.get('data'):
            d = data['data']
            HS300_CLOSE = float(d.get('f43', HS300_CLOSE)) / 100 if d.get('f43') else HS300_CLOSE
            print(f'HS300实时: {HS300_CLOSE:.0f}')
except Exception as e:
    print(f'HS300降级(使用默认): {e}')

# ── 2. 门禁判断 ──
EXIT_THRESH = -0.12; FLOOR = 0.10
gate_pos = 1.0 if DD_2Y >= EXIT_THRESH else (FLOOR*2 if DD_2Y >= EXIT_THRESH-0.05 else FLOOR)
print(f'DD={DD_2Y*100:.1f}% gate={gate_pos*100:.0f}%')

# ── 3. 用缓存因子选股 ──
picks = []
try:
    # 优先用factors_orig6f_2002.parquet
    factor_file = os.path.join(DIR, 'cache', 'factors_orig6f_2002.parquet')
    if os.path.exists(factor_file):
        fn = pd.read_parquet(factor_file)
        fn['trade_date'] = pd.to_datetime(fn['trade_date'])
        latest_date = fn['trade_date'].max()
        day = fn[fn['trade_date'] == latest_date].copy()
        print(f'因子: {len(day)}只 (日期={latest_date.date()})')

        FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
        TOP3 = [('turnover_rev','price_rev'),('max_rev','price_rev'),('amihud','turnover_rev')]
        for f in FEATS:
            if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
        day['score'] = 0
        for fa, fb in TOP3:
            if fa+'_r' in day.columns and fb+'_r' in day.columns:
                day['score'] += day[fa+'_r'] * day[fb+'_r']
        day['mcap_r'] = day['amount_proxy'].rank(pct=True)
        day = day[day['mcap_r'] >= 0.20]
        top = day.nlargest(TOP_N, 'score')

        SLIPPAGE = 0.001; COMM = 0.00033
        cash_per = CAPITAL * gate_pos / TOP_N
        for _, row in top.iterrows():
            price = float(row['close'])
            shares = max(1, int(cash_per / (price * (1 + SLIPPAGE)) / 100)) * 100
            bp = price * (1 + SLIPPAGE)
            cost = shares * bp * (1 + COMM)
            code_str = str(row['ts_code'])
            score_val = float(row['score'])
            picks.append({
                'code': code_str, 'shares': shares, 'price': price,
                'buy_price': float(bp), 'cost': float(cost), 'score': score_val
            })
            print(f'  {code_str} score={score_val:.3f} price={price:.2f} {shares}股')
    else:
        print(f'因子文件缺失, 输出空持仓')
except Exception as e:
    print(f'选股失败: {e}')

# ── 4. 保存 ──
if not picks:
    print('[!] 无选股结果 — 云端因子缓存缺失, 需本地运行 build_factors_orig6f_2002.py 后上传')
pf = {
    'strategy': '小众战法_云端',
    'date': TODAY.isoformat(),
    'capital': CAPITAL,
    'gate': {'position': gate_pos, 'hs300': HS300_CLOSE, 'dd_2y': DD_2Y},
    'cash': round(CAPITAL - sum(p['cost'] for p in picks), 2),
    'invested': round(sum(p['cost'] for p in picks), 2),
    'positions': picks,
}
with open(PF_FILE, 'w', encoding='utf-8') as f:
    json.dump(pf, f, ensure_ascii=False, indent=2)
print(f'\n[OK] paper_portfolio.json ({len(picks)}只)')

# 日志
log_line = f'| {TODAY.isoformat()} | {len(picks)}只 | ¥{pf[\"invested\"]:,.0f} |\n'
with open(LOG_FILE, 'a', encoding='utf-8') as f:
    f.write(log_line)
print('Done.')
