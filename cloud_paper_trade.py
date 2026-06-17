# -*- coding: utf-8 -*-
"""GitHub Actions 云端纸交 — 纯parquet, 无DuckDB依赖"""
import sys, io, os, json, time
import numpy as np
import pandas as pd
from datetime import date, timedelta
from lightgbm import LGBMRegressor

CACHE_DIR = 'cache'
CLOUD_DIR = 'cloud_data'
TOP_N = 30
INIT_CASH = 100000

def load_parquet(path):
    for alt in [path, f'{CLOUD_DIR}/{path.split("/")[-1]}']:
        if os.path.exists(alt):
            return pd.read_parquet(alt)
    print(f'⚠ 缺失: {path}')
    return None

def std_ts_code(s):
    s = str(s)
    if '.SZ' in s: return 'sz' + s.replace('.SZ','')
    if '.SH' in s: return 'sh' + s.replace('.SH','')
    if '.BJ' in s: return 'bj' + s.replace('.BJ','')
    return s

def load_portfolio():
    if os.path.exists('paper_portfolio.json'):
        with open('paper_portfolio.json','r',encoding='utf-8') as f:
            return json.load(f)
    return {'cash': INIT_CASH, 'positions': {}, 'history': []}

# ── 加载数据 ──
print('加载数据...')
factors = load_parquet('cache/factors_all.parquet')
target  = load_parquet('cache/target_60d.parquet')
kline   = load_parquet('cloud_data/kline_90d.parquet')
fin     = load_parquet('cloud_data/fin_snapshot.parquet')
macro   = load_parquet('cloud_data/macro_90d.parquet')
margin  = load_parquet('cloud_data/margin_90d.parquet')
stocks  = load_parquet('cloud_data/stock_basic.parquet')

if factors is None:
    print('❌ 因子缓存缺失')
    sys.exit(0)

# ── 最新交易日 ──
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
latest_date = str(kline['trade_date'].max().date())
print(f'最新交易日: {latest_date}')

# ── 合并数据+训练 ──
print('合并...')
data = factors.merge(target[['ts_code','trade_date','excess_ret']], on=['ts_code','trade_date'], how='inner')
feat_cols = [c for c in data.columns if c not in ('ts_code','trade_date','excess_ret','fwd_ret','factor_group','report_date','_k','close')]

# 二次项中性化 (简化版: 用np.linalg.lstsq)
if 'log_mcap' in data.columns:
    for col in [c for c in feat_cols if c != 'log_mcap']:
        y = data[col].values.astype(float)
        x = data['log_mcap'].values.astype(float)
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 100: continue
        x2 = x[mask] * x[mask]
        X = np.column_stack([np.ones(mask.sum()), x[mask], x2])
        coeffs = np.linalg.lstsq(X, y[mask], rcond=None)[0]
        data[col] = y - (coeffs[0] + coeffs[1] * x + coeffs[2] * x * x)

print(f'训练样本: {len(data)}')

# ── ML训练 ──
train_data = data[data['trade_date'] < latest_date].dropna(subset=feat_cols+['excess_ret'])
if len(train_data) < 5000:
    print('训练样本不足')
    sys.exit(0)

X_train = train_data[feat_cols].fillna(train_data[feat_cols].median())
y_train = train_data['excess_ret']

model = LGBMRegressor(
    objective='regression', metric='rmse',
    learning_rate=0.05, num_leaves=63, max_depth=10,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_estimators=300, verbose=-1, random_state=42, n_jobs=1
)
model.fit(X_train, y_train)
print('ML训练完成')

# ── 选股 ──
today_data = data[data['trade_date'] == latest_date].dropna(subset=feat_cols)
if len(today_data) < 100:
    # 使用缓存中最近日期
    latest_in_cache = str(data['trade_date'].max()).split()[0]
    today_data = data[data['trade_date'] == latest_in_cache].dropna(subset=feat_cols)
    print(f'回退到缓存日: {latest_in_cache}')

X_today = today_data[feat_cols].fillna(train_data[feat_cols].median())
scores = model.predict(X_today)
today_data = today_data.copy()
today_data['ml_score'] = scores

# 市值底线
if 'log_mcap' in today_data.columns:
    floor = today_data['log_mcap'].quantile(0.20)
    today_data = today_data[today_data['log_mcap'] >= floor]
    print(f'市值底线: {len(today_data)}只')

top_n = today_data.drop_duplicates(subset=['ts_code']).nlargest(TOP_N, 'ml_score')

# ── 获取执行价格 ──
kline_today = kline[kline['trade_date'] == latest_date]
if kline_today.empty:
    kline_today = kline[kline['trade_date'] == str(kline['trade_date'].max())]
kline_today['ts_code_std'] = kline_today['ts_code'].apply(std_ts_code)
price_map = dict(zip(kline_today['ts_code_std'], kline_today['close']))

# ── 更新纸交组合 ──
pf = load_portfolio()
valid_buy = [c for c in top_n['ts_code'].tolist() if c in price_map and price_map[c] > 0]

# 卖出不在新选股的
for code in list(pf['positions'].keys()):
    if code not in valid_buy:
        px = price_map.get(code, 0)
        if px > 0:
            shares = pf['positions'][code]['shares']
            pf['cash'] += shares * px * 0.9985  # 扣除卖出成本
            pf['history'].append({'date': latest_date, 'action': 'SELL', 'code': code,
                                  'shares': shares, 'price': px})
        del pf['positions'][code]

# 等权买入
n_buy = min(len(valid_buy), TOP_N)
if n_buy > 0 and pf['cash'] > 1000:
    cap = pf['cash'] / n_buy
    for code in valid_buy[:n_buy]:
        px = price_map[code]
        shares = int(cap / (px * 1.0015) / 100) * 100  # T+1买入成本
        if shares > 0:
            cost = shares * px * 1.0015
            if cost <= pf['cash']:
                pf['cash'] -= cost
                pf['positions'][code] = {'shares': shares, 'buy_price': px, 'buy_date': latest_date}
                pf['history'].append({'date': latest_date, 'action': 'BUY', 'code': code,
                                      'shares': shares, 'price': px})

# ── 估值 ──
total_mv = pf['cash']
for code, pos in pf['positions'].items():
    px = price_map.get(code, pos.get('buy_price', 0))
    total_mv += pos['shares'] * px

with open('paper_portfolio.json', 'w', encoding='utf-8') as f:
    json.dump(pf, f, ensure_ascii=False, indent=2)

# ── 日志 ──
with open('paper_daily_log.md', 'a', encoding='utf-8') as f:
    f.write(f'| {latest_date} | {len(pf["positions"])}只 | ¥{total_mv:,.0f} | {(total_mv/INIT_CASH-1)*100:+.2f}% |\n')

print(f'纸交完成: {len(pf["positions"])}只, 资产¥{total_mv:,.0f}, 收益{(total_mv/INIT_CASH-1)*100:+.2f}%')
