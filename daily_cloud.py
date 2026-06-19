# -*- coding: utf-8 -*-
"""
QuantLab 云端纸交引擎 — GitHub Actions 版
===========================================
零本地依赖: 全部数据来自AKShare, 慢速爬取防封
3秒间隔 × 约15次API调用 ≈ 45秒完成
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import warnings; warnings.filterwarnings('ignore')

import akshare as ak
import pandas as pd
import numpy as np

SLEEP = 4  # API间隔秒数, 越慢越安全
TODAY = date.today()
OUTPUT = 'paper_daily_signal.json'

print(f'QuantLab Cloud — {TODAY.isoformat()}')
print(f'API间隔: {SLEEP}s\n')

# ============================================================
# 0. 交易日检测
# ============================================================
print('[0/4] 交易日检测...')
time.sleep(SLEEP)
cal = ak.tool_trade_date_hist_sina()
cal_dates = set(cal['trade_date'].astype(str).values)

if TODAY.isoformat() not in cal_dates:
    # 找最近交易日
    for td in sorted(cal_dates, reverse=True):
        if td <= TODAY.isoformat():
            LTD = td
            break
    else:
        LTD = TODAY.isoformat()
    if TODAY.isoformat() != LTD:
        print(f'  非交易日, 最近: {LTD}')
        sys.exit(0)

LTD = TODAY.isoformat()
print(f'  交易日: {LTD}')

# ============================================================
# 1. 加载今日市场数据 (慢速, 只取最近60天用于因子计算)
# ============================================================
print('\n[1/4] 加载沪深300+个股行情...')

# 沪深300 (择时)
time.sleep(SLEEP)
hs300 = ak.stock_zh_index_daily(symbol='sh000300')
hs300['trade_date'] = pd.to_datetime(hs300['date'])
hs300 = hs300.sort_values('trade_date')
hs300_px = hs300.set_index('trade_date')['close']
latest_close = hs300_px.iloc[-1]
ma200 = hs300_px.rolling(200).mean().iloc[-1]
is_bull = latest_close > ma200
print(f'  CSI300: {latest_close:.0f} MA200: {ma200:.0f} → {"BULL" if is_bull else "BEAR"}')

# 中证500成分股 (选股池, 约500只)
print('  加载中证500成分股...')
time.sleep(SLEEP)
try:
    zz500 = ak.index_stock_cons(symbol='000905')
    stock_codes = zz500['品种代码'].tolist()[:300]  # 限300只, 减少API调用
    print(f'  成分股: {len(stock_codes)}只 (限300)')
except:
    # fallback: 用沪深300
    time.sleep(SLEEP)
    hs300_c = ak.index_stock_cons(symbol='000300')
    stock_codes = hs300_c['品种代码'].tolist()[:300]
    print(f'  fallback沪深300: {len(stock_codes)}只')

# ============================================================
# 2. 计算技术因子 (不需要历史数据库, 直接从AKShare取60天K线)
# ============================================================
print(f'\n[2/4] 计算技术因子 (慢速, {len(stock_codes)}只, 每只{SLEEP}s)...')

def compute_factors(code):
    """从AKShare取单只股票60天K线, 计算12个技术因子"""
    try:
        time.sleep(SLEEP)
        kline = ak.stock_zh_a_hist(symbol=code, period='daily',
                                    start_date=(TODAY-timedelta(days=120)).strftime('%Y%m%d'),
                                    end_date=TODAY.strftime('%Y%m%d'),
                                    adjust='qfq')
        if kline is None or len(kline) < 30:
            return None

        close = kline['收盘'].astype(float)
        vol = kline['成交量'].astype(float)

        # 12技术因子
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain_6 = gain.rolling(6).mean()
        avg_loss_6 = loss.rolling(6).mean()
        rs6 = avg_gain_6 / avg_loss_6.replace(0, 1)
        rsi6 = 100 - (100 / (1 + rs6))

        avg_gain_14 = gain.rolling(14).mean()
        avg_loss_14 = loss.rolling(14).mean()
        rs14 = avg_gain_14 / avg_loss_14.replace(0, 1)
        rsi14 = 100 - (100 / (1 + rs14))

        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma120 = close.rolling(120).mean()
        std20 = close.rolling(20).std()
        boll_pos = (close - (ma20 - 2*std20)) / (4*std20).replace(0, 1)
        boll_width = (4*std20) / ma20.replace(0, 1)

        return {
            'code': code,
            'close': float(close.iloc[-1]),
            'rsi6': float(rsi6.iloc[-1]) if not pd.isna(rsi6.iloc[-1]) else 50,
            'rsi14': float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50,
            'boll_pos': float(boll_pos.iloc[-1]) if not pd.isna(boll_pos.iloc[-1]) else 0.5,
            'boll_width': float(boll_width.iloc[-1]) if not pd.isna(boll_width.iloc[-1]) else 0.02,
            'div_ma20': float((close.iloc[-1]/ma20.iloc[-1]-1)) if not pd.isna(ma20.iloc[-1]) else 0,
            'div_ma60': float((close.iloc[-1]/ma60.iloc[-1]-1)) if not pd.isna(ma60.iloc[-1]) else 0,
            'div_ma120': float((close.iloc[-1]/ma120.iloc[-1]-1)) if not pd.isna(ma120.iloc[-1]) else 0,
            'vol_ratio': float(vol.iloc[-1]/vol.rolling(20).mean().iloc[-1]) if vol.rolling(20).mean().iloc[-1]>0 else 1,
            'ma_score': float(((close>ma20).astype(int)+(close>ma60).astype(int)+(close>ma120).astype(int)).iloc[-1]/3),
            'rsi_extreme': float(abs(rsi14.iloc[-1]-50)/50) if not pd.isna(rsi14.iloc[-1]) else 0,
            'margin_panic': float(-delta.rolling(5).std().iloc[-1]) if delta.rolling(5).std().iloc[-1]>0 else 0,
            'streak5_dn': float((delta<0).rolling(5).sum().iloc[-1]),
        }
    except Exception as e:
        return None

# 只取前100只做因子计算 (控制API调用总量)
MAX_STOCKS = 100
print(f'  实际计算: {MAX_STOCKS}只 (~{MAX_STOCKS*SLEEP}s)')
factors_list = []
for i, code in enumerate(stock_codes[:MAX_STOCKS]):
    if i % 20 == 0:
        print(f'    {i}/{MAX_STOCKS}...')
    result = compute_factors(code)
    if result:
        factors_list.append(result)

if not factors_list:
    print('  因子计算失败, 退出')
    sys.exit(1)

factors_df = pd.DataFrame(factors_list)
print(f'  有效: {len(factors_df)}只')

# ============================================================
# 3. 信号生成 (简化的规则打分, 替代LightGBM)
# ============================================================
print('\n[3/4] 生成信号...')

# 等权打分 (12因子 → 标准化 → 求和)
FEATS = ['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
         'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

for f in FEATS:
    if f in factors_df.columns:
        mu = factors_df[f].mean()
        std = factors_df[f].std()
        if std > 0:
            factors_df[f'{f}_z'] = (factors_df[f] - mu) / std
        else:
            factors_df[f'{f}_z'] = 0

score_cols = [f'{f}_z' for f in FEATS if f'{f}_z' in factors_df.columns]
factors_df['score'] = factors_df[score_cols].sum(axis=1)
factors_df = factors_df.nlargest(20, 'score')

stock_picks = []
for _, row in factors_df.head(15).iterrows():
    stock_picks.append({
        'ts_code': row['code'],
        'score': round(float(row['score']), 2),
        'close': round(float(row['close']), 2),
    })
    print(f'    {row["code"]}: score={row["score"]:.2f} close={row["close"]:.2f}')

# ============================================================
# 4. ETF信号
# ============================================================
print('\n[4/4] ETF信号...')
if is_bull:
    etf_picks = ['电子ETF(159997)', '通信ETF(515880)', '计算机ETF(512720)',
                 '国防ETF(512670)', '机械ETF(159886)']
else:
    etf_picks = ['国债ETF(511010)', '黄金ETF(518880)', '纳指ETF(513100)']
print(f'  {"BULL→进攻" if is_bull else "BEAR→防御"}: {", ".join(etf_picks)}')

# ============================================================
# 输出
# ============================================================
report = {
    'date': LTD,
    'generated': datetime.now().isoformat(),
    'regime': 'BULL' if is_bull else 'BEAR',
    'csi300': round(float(latest_close), 1),
    'ma200': round(float(ma200), 1),
    'stock_picks': stock_picks,
    'etf_picks': etf_picks,
}

os.makedirs(os.path.dirname(OUTPUT) if os.path.dirname(OUTPUT) else '.', exist_ok=True)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'\n✓ 信号已保存: {OUTPUT}')
print(f'  个股: {len(stock_picks)}只 | ETF: {len(etf_picks)}个')
