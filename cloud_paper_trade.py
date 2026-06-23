# -*- coding: utf-8 -*-
"""云端纸交引擎 (GitHub Actions) — 零DB依赖, AKShare实时数据"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False

SLEEP = 3
TODAY = date.today()
DIR = os.path.dirname(os.path.abspath(__file__))
PF_FILE = os.path.join(DIR, 'paper_portfolio.json')
LOG_FILE = os.path.join(DIR, 'paper_daily_log.md')

print(f'=== 云端纸交 {TODAY.isoformat()} ===')

# ── 0. 交易日 ──
LTD = TODAY.isoformat()
if HAS_AK:
    try:
        time.sleep(SLEEP)
        cal = ak.tool_trade_date_hist_sina()
        cal_dates = set(cal['trade_date'].astype(str).values)
        if TODAY.isoformat() not in cal_dates:
            for td in sorted(cal_dates, reverse=True):
                if td <= TODAY.isoformat():
                    LTD = td; break
            print(f'非交易日 → {LTD}')
        else:
            print(f'交易日: {LTD}')
    except Exception as e:
        print(f'日历获取失败: {e}, 假定交易日')

# ── 1. 大盘门禁 ──
HS300_CLOSE = 4900; DD_2Y = -2.8
gate_pos = 1.0
if HAS_AK:
    try:
        time.sleep(SLEEP)
        hs300 = ak.stock_zh_index_daily(symbol='sh000300')
        hs300['date'] = pd.to_datetime(hs300['date'])
        hs300 = hs300.sort_values('date')
        close_series = hs300.set_index('date')['close']
        latest_close = close_series.iloc[-1]
        high_2y = close_series.rolling(500).max().iloc[-1]
        dd_2y_val = float(latest_close / high_2y - 1)
        EXIT_THRESH = -0.12; FLOOR = 0.10
        gate_pos = 1.0 if dd_2y_val >= EXIT_THRESH else (FLOOR * 2 if dd_2y_val >= EXIT_THRESH - 0.05 else FLOOR)
        HS300_CLOSE = float(latest_close)
        DD_2Y = dd_2y_val
        print(f'HS300={HS300_CLOSE:.0f} DD={DD_2Y*100:.1f}% → gate={gate_pos*100:.0f}%')
    except Exception as e:
        print(f'门禁降级: {e}')

# ── 2. 选股 ──
CAPITAL = 100000; TOP_N = 20
picks = []

if HAS_AK:
    try:
        # 用中证500成分股做选股池
        time.sleep(SLEEP)
        try:
            zz500 = ak.index_stock_cons(symbol='000905')
            stock_codes = zz500['品种代码'].tolist()[:150]
        except:
            time.sleep(SLEEP)
            hs300_c = ak.index_stock_cons(symbol='000300')
            stock_codes = hs300_c['品种代码'].tolist()[:150]
        print(f'选股池: {len(stock_codes)}只')

        scores = []
        for i, code in enumerate(stock_codes):
            try:
                time.sleep(SLEEP)
                kl = ak.stock_zh_a_hist(symbol=code, period='daily',
                    start_date=(TODAY - timedelta(days=100)).strftime('%Y%m%d'),
                    end_date=TODAY.strftime('%Y%m%d'), adjust='qfq')
                if kl is None or len(kl) < 20:
                    continue
                close = kl['收盘'].astype(float).values
                vol = kl['成交量'].astype(float).values
                cur = close[-1]
                # Simple scores
                ret_1m = cur / close[-22] - 1 if len(close) >= 22 else 0
                ret_3m = cur / close[-66] - 1 if len(close) >= 66 else 0
                vol_ratio = vol[-1] / vol[-20:].mean() if len(vol) >= 20 else 1
                ma20 = close[-20:].mean()
                div_ma20 = cur / ma20 - 1
                rsi_rough = sum(1 for x in close[-14:] if x > cur) / 14  # rough
                score = ret_1m * 0.3 + ret_3m * 0.1 - abs(div_ma20) * 0.2 + vol_ratio * 0.1 + (1 - rsi_rough) * 0.3
                scores.append({'code': code, 'close': cur, 'score': score,
                               'ret_1m': ret_1m, 'div_ma20': div_ma20})
            except:
                continue

        scores_df = pd.DataFrame(scores).nlargest(TOP_N, 'score')
        SLIPPAGE = 0.001; COMM = 0.00033
        cash_per = CAPITAL * gate_pos / TOP_N

        for _, r in scores_df.iterrows():
            price = r['close']
            shares = max(1, int(cash_per / (price * (1 + SLIPPAGE)) / 100)) * 100
            bp = price * (1 + SLIPPAGE)
            cost = shares * bp * (1 + COMM)
            picks.append({
                'code': str(r['code']), 'shares': shares, 'price': float(price),
                'buy_price': float(bp), 'cost': float(cost), 'score': float(r['score'])
            })
            print(f'  {r[\"code\"]} score={r[\"score\"]:.3f} price={price:.2f} {shares}股')
    except Exception as e:
        print(f'选股失败: {e}')

# ── 3. 保存 ──
pf = {
    'strategy': '圆桌会议_云端纸交',
    'date': LTD,
    'capital': CAPITAL,
    'gate': {'position': gate_pos, 'hs300': HS300_CLOSE, 'dd_2y': DD_2Y},
    'cash': round(CAPITAL - sum(p['cost'] for p in picks), 2),
    'invested': round(sum(p['cost'] for p in picks), 2),
    'positions': picks,
    'total_value': CAPITAL,
}

with open(PF_FILE, 'w', encoding='utf-8') as f:
    json.dump(pf, f, ensure_ascii=False, indent=2)
print(f'\n✅ paper_portfolio.json ({len(picks)}只)')

# 日志
with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write(f'| {LTD} | {len(picks)}只 | {CAPITAL:,} |\n')

print('Done.')
