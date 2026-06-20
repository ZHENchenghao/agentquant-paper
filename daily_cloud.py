# -*- coding: utf-8 -*-
"""
QuantLab 云端纸交引擎 v2.0 — GitHub Actions 版
===============================================
三策略: 小众战法Top30 + ETF轮动 + ML选股(保留对比)
零本地依赖: 全部数据来自AKShare, 慢速爬取防封
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import warnings; warnings.filterwarnings('ignore')
import akshare as ak
import pandas as pd
import numpy as np

SLEEP = 4; TODAY = date.today(); OUTPUT = 'paper_daily_signal.json'
print('QuantLab Cloud v2.0 — %s' % TODAY.isoformat())

# ============================================================
# 0. 交易日检测
# ============================================================
print('[0/5] 交易日检测...')
time.sleep(SLEEP)
cal = ak.tool_trade_date_hist_sina()
cal_dates = set(cal['trade_date'].astype(str).values)
if TODAY.isoformat() not in cal_dates:
    for td in sorted(cal_dates, reverse=True):
        if td <= TODAY.isoformat(): LTD = td; break
    else: LTD = TODAY.isoformat()
    if TODAY.isoformat() != LTD:
        print('  非交易日, 最近: %s' % LTD); sys.exit(0)
LTD = TODAY.isoformat()
print('  交易日: %s' % LTD)

# ============================================================
# 1. 市场状态 + 沪深300
# ============================================================
print('\n[1/5] 市场状态...')
time.sleep(SLEEP)
hs300 = ak.stock_zh_index_daily(symbol='sh000300')
hs300['trade_date'] = pd.to_datetime(hs300['date'])
hs300 = hs300.sort_values('trade_date')
px = hs300.set_index('trade_date')['close']
latest_close = px.iloc[-1]; ma200 = px.rolling(200).mean().iloc[-1]
ma50 = px.rolling(50).mean().iloc[-1]
high_2y = px.rolling(504).max().iloc[-1]; low_1y = px.rolling(252).min().iloc[-1]
dd_2y = latest_close/high_2y - 1; recovery = latest_close/low_1y - 1
is_bull = latest_close > ma200

# DD_SMART v2
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
if dd_2y >= EXIT_THRESH: gate_pos, gate_msg = 1.0, 'FULL'
elif dd_2y >= EXIT_THRESH - 0.05: gate_pos, gate_msg = FLOOR*2, 'REDUCE'
else: gate_pos, gate_msg = FLOOR, 'CRASH'

print('  HS300: %.0f MA50: %.0f MA200: %.0f' % (latest_close, ma50, ma200))
print('  DD_SMART: DD=%.1f%% REC=%.1f%% -> %s(%.0f%%)' % (dd_2y*100, recovery*100, gate_msg, gate_pos*100))

# ============================================================
# 2. 小众战法 Top30 (核心策略)
# ============================================================
print('\n[2/5] 小众战法 Top30...')
print('  加载中证500成分股...')
time.sleep(SLEEP)
try:
    zz500 = ak.index_stock_cons(symbol='000905')
    stock_codes = zz500['品种代码'].tolist()[:500]
except:
    try:
        hs300_c = ak.index_stock_cons(symbol='000300')
        stock_codes = hs300_c['品种代码'].tolist()[:300]
    except:
        # fallback: 用全A股spot取前200只
        spot = ak.stock_zh_a_spot_em()
        stock_codes = spot['代码'].tolist()[:200]
print('  股票池: %d只' % len(stock_codes))

# 计算6因子
MAX_STOCKS = 200  # 控制API总量: 200只×4s=800s≈13分钟
print('  计算6因子 (%d只, ~%ds)...' % (MAX_STOCKS, MAX_STOCKS*SLEEP))

def compute_6factors(code):
    """从AKShare取120天K线, 计算小众战法6因子"""
    try:
        time.sleep(SLEEP)
        kline = ak.stock_zh_a_hist(symbol=code, period='daily',
            start_date=(TODAY-timedelta(days=150)).strftime('%Y%m%d'),
            end_date=TODAY.strftime('%Y%m%d'), adjust='qfq')
        if kline is None or len(kline) < 40: return None

        close = kline['收盘'].astype(float).values
        vol = kline['成交量'].astype(float).values
        amount = kline['成交额'].astype(float).values if '成交额' in kline.columns else close * vol
        high = kline['最高'].astype(float).values
        low = kline['最低'].astype(float).values
        open_p = kline['开盘'].astype(float).values
        turnover = kline['换手率'].astype(float).values if '换手率' in kline.columns else vol / 1e8
        turnover = np.nan_to_num(turnover, 0.001)

        if len(close) < 30: return None

        # 日收益率
        ret_1d = np.diff(close) / close[:-1]
        ret_1d = np.insert(ret_1d, 0, 0)

        # 1. Amihud (log, 20日)
        dollar_vol = np.maximum(amount, vol * close)
        illiq = np.abs(ret_1d) / np.maximum(dollar_vol, 1) * 1e10
        illiq_20 = np.convolve(illiq, np.ones(20)/20, mode='same')
        amihud = np.log(1.0 + illiq_20)

        # 2. Max_Rev (20日最大收益的负值)
        max_ret_20 = np.array([np.max(ret_1d[max(0,i-19):i+1]) for i in range(len(ret_1d))])
        max_rev = -max_ret_20

        # 3. Price_Rev (负收盘价)
        price_rev = -close

        # 4. Turnover_Rev (20日均换手率负值)
        turnover_20 = np.convolve(turnover, np.ones(20)/20, mode='same')
        turnover_rev = -turnover_20

        # 5. SR5 (5日反转)
        close_5 = np.roll(close, 5); close_5[:5] = close[:5]
        sr5 = -(close / close_5 - 1)

        # 6. VP_Corr (10日量价相关性)
        vol_change = np.diff(vol) / (vol[:-1] + 1)
        vol_change = np.insert(vol_change, 0, 0)
        vp_corr = np.zeros(len(close))
        for i in range(10, len(close)):
            r = ret_1d[i-9:i+1]; v = vol_change[i-9:i+1]
            if np.std(r) > 0 and np.std(v) > 0:
                vp_corr[i] = np.corrcoef(r, v)[0,1]

        return {
            'code': code, 'close': float(close[-1]),
            'amihud': float(amihud[-1]), 'max_rev': float(max_rev[-1]),
            'price_rev': float(price_rev[-1]), 'turnover_rev': float(turnover_rev[-1]),
            'sr5': float(sr5[-1]), 'vp_corr': float(vp_corr[-1])
        }
    except:
        return None

factors_list = []
for i, code in enumerate(stock_codes[:MAX_STOCKS]):
    if i % 50 == 0: print('    %d/%d...' % (i, min(MAX_STOCKS, len(stock_codes))))
    result = compute_6factors(str(code))
    if result: factors_list.append(result)

if len(factors_list) < 60:
    print('  因子不足(%d只), 退出' % len(factors_list)); sys.exit(1)

factors_df = pd.DataFrame(factors_list)
print('  有效: %d只' % len(factors_df))

# 乘法选股
ALL_PAIRS = [('amihud','turnover_rev'),('amihud','max_rev'),('amihud','sr5'),('turnover_rev','sr5')]
all_f = list(set([x for p in ALL_PAIRS for x in p]))
for f in all_f:
    if f in factors_df.columns:
        factors_df[f+'_r'] = factors_df[f].rank(pct=True)

factors_df['score'] = 0
for fa, fb in ALL_PAIRS:
    if fa+'_r' in factors_df.columns and fb+'_r' in factors_df.columns:
        factors_df['score'] += factors_df[fa+'_r'] * factors_df[fb+'_r']

# 市值代理过滤(用成交额)
factors_df['mcap_r'] = factors_df['close'].rank(pct=True)
factors_df = factors_df[factors_df['mcap_r'] >= 0.20]
top = factors_df.nlargest(30, 'score')

xz_picks = []
for _, row in top.iterrows():
    xz_picks.append({
        'ts_code': row['code'], 'score': round(float(row['score']), 3),
        'close': round(float(row['close']), 2)
    })
print('  小众战法: %d只 (得分%.2f~%.2f)' % (len(xz_picks), top['score'].min(), top['score'].max()))
for p in xz_picks[:5]: print('    %s: %.3f @ %.2f' % (p['ts_code'], p['score'], p['close']))
if len(xz_picks) > 5: print('    ... +%d只' % (len(xz_picks)-5))

# ============================================================
# 3. ML选股(保留对比, 简化版)
# ============================================================
print('\n[3/5] ML选股(对比)...')
FEATS_ML = ['rsi6','rsi14','boll_pos','div_ma20','div_ma60','vol_ratio','ma_score','rsi_extreme','streak5_dn']

def compute_ml_factors(code):
    try:
        time.sleep(1.5)  # 更快, 因为因子更简单
        kline = ak.stock_zh_a_hist(symbol=code, period='daily',
            start_date=(TODAY-timedelta(days=150)).strftime('%Y%m%d'),
            end_date=TODAY.strftime('%Y%m%d'), adjust='qfq')
        if kline is None or len(kline) < 40: return None
        close = kline['收盘'].astype(float); vol = kline['成交量'].astype(float)
        delta = close.diff()
        gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        rs6 = gain.rolling(6).mean() / loss.rolling(6).mean().replace(0,1)
        rs14 = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0,1)
        ma20 = close.rolling(20).mean(); ma60 = close.rolling(60).mean()
        std20 = close.rolling(20).std()
        return {
            'code': code, 'close': float(close.iloc[-1]),
            'rsi6': float(100-100/(1+rs6.iloc[-1])) if not pd.isna(rs6.iloc[-1]) else 50,
            'rsi14': float(100-100/(1+rs14.iloc[-1])) if not pd.isna(rs14.iloc[-1]) else 50,
            'boll_pos': float((close.iloc[-1]-(ma20.iloc[-1]-2*std20.iloc[-1]))/(4*std20.iloc[-1]+1)),
            'div_ma20': float(close.iloc[-1]/ma20.iloc[-1]-1),
            'div_ma60': float(close.iloc[-1]/ma60.iloc[-1]-1),
            'vol_ratio': float(vol.iloc[-1]/vol.rolling(20).mean().iloc[-1]),
            'ma_score': float(((close>ma20).astype(int)+(close>ma60).astype(int)).iloc[-1]/2),
            'rsi_extreme': float(abs(100-100/(1+rs14.iloc[-1])-50)/50),
            'streak5_dn': float((delta<0).rolling(5).sum().iloc[-1])
        }
    except: return None

ml_list = []
for code in stock_codes[:100]:  # 只取100只做ML对比
    result = compute_ml_factors(str(code))
    if result: ml_list.append(result)

ml_picks = []
if ml_list:
    ml_df = pd.DataFrame(ml_list)
    for f in FEATS_ML:
        if f in ml_df.columns:
            mu, std = ml_df[f].mean(), ml_df[f].std()
            ml_df[f+'_z'] = (ml_df[f]-mu)/std if std>0 else 0
    score_cols = [f+'_z' for f in FEATS_ML if f+'_z' in ml_df.columns]
    ml_df['score'] = ml_df[score_cols].sum(axis=1)
    ml_top = ml_df.nlargest(15, 'score')
    for _, row in ml_top.iterrows():
        ml_picks.append({'ts_code': row['code'], 'score': round(float(row['score']),2)})
    print('  ML: %d只' % len(ml_picks))

# ============================================================
# 4. ETF信号
# ============================================================
print('\n[4/5] ETF信号...')
if is_bull:
    etf_picks = ['电子ETF(159997)','通信ETF(515880)','计算机ETF(512720)','国防ETF(512670)','机械ETF(159886)']
else:
    etf_picks = ['国债ETF(511010)','黄金ETF(518880)','纳指ETF(513100)']
print('  %s: %s' % ('BULL->进攻' if is_bull else 'BEAR->防御', ','.join(etf_picks)))

# ============================================================
# 5. 输出
# ============================================================
report = {
    'date': LTD, 'generated': datetime.now().isoformat(),
    'regime': 'BULL' if is_bull else 'BEAR',
    'csi300': round(float(latest_close), 1), 'ma200': round(float(ma200), 1),
    'dd_smart': {'dd_2y': round(dd_2y*100,1), 'gate': gate_msg, 'position': gate_pos},
    'xiaozhong': {'picks': xz_picks, 'count': len(xz_picks), 'pairs': ['amihud_x_turnover','amihud_x_maxrev','amihud_x_sr5','turnover_x_sr5']},
    'ml_stock': {'picks': ml_picks, 'count': len(ml_picks)},
    'etf': {'picks': etf_picks}
}
os.makedirs(os.path.dirname(OUTPUT) if os.path.dirname(OUTPUT) else '.', exist_ok=True)
with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('\n' + '='*50)
print('信号已保存: %s' % OUTPUT)
print('  小众战法: %d只 | ML: %d只 | ETF: %d个' % (len(xz_picks), len(ml_picks), len(etf_picks)))
print('  门禁: %s(%.0f%%)' % (gate_msg, gate_pos*100))
