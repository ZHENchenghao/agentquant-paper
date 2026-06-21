# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
import json

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

CHAINS = [
    {'id':'WTI->OilGas',  'leader':'wti',    'follower':'sz399441', 'dir':'+', 'lag':(0,5),  'thresh':3.0},
    {'id':'WTI->Aviation', 'leader':'wti',    'follower':'sz399959', 'dir':'-', 'lag':(1,8),  'thresh':3.0},
    {'id':'Copper->Metal', 'leader':'copper', 'follower':'sh000819', 'dir':'+', 'lag':(1,10), 'thresh':2.0},
    {'id':'US10Y->STAR50', 'leader':'us10y',  'follower':'sh000688', 'dir':'-', 'lag':(1,10), 'thresh':0.15},
    {'id':'US10Y->Bank',   'leader':'us10y',  'follower':'sz399986', 'dir':'+', 'lag':(1,8),  'thresh':0.15},
    {'id':'Gold->Metal',   'leader':'gold',   'follower':'sh000819', 'dir':'+', 'lag':(0,5),  'thresh':1.5},
    {'id':'SPX->HS300',    'leader':'spx',    'follower':'sh000300', 'dir':'+', 'lag':(0,3),  'thresh':1.0},
    {'id':'VIX->HS300',    'leader':'vix',    'follower':'sh000300', 'dir':'-', 'lag':(0,5),  'thresh':3.0},
]

print('=' * 70)
print('Conduction Chain Event Prediction Backtest')
print('=' * 70)

con = duckdb.connect(DB, read_only=True)

# Macro
macro = con.execute("""
    SELECT trade_date, wti, copper, gold, us10y, vix
    FROM macro_indicators WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
for c in ['wti', 'copper', 'gold', 'us10y', 'vix']:
    macro[c] = macro[c].ffill().bfill()

# SPX
spx_df = con.execute("""
    SELECT trade_date, close FROM global_index_daily
    WHERE index_code = '.INX' AND trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
spx_df.columns = ['trade_date', 'spx']

# Compute sector returns from individual stocks
# Map chain follower to industry name
CHAIN_INDUSTRY_MAP = {
    'sz399441': '石油石化', 'sz399959': '航空',
    'sh000819': '有色金属', 'sz399986': '银行',
    'sh000688': '半导体',  # 科创50 proxy
    'sz399133': '建筑材料', 'sh000300': '沪深300',
}

# Get industry mapping
ind_map = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map
    ) WHERE rn = 1
""").df()

# Daily stock returns
stock_ret = con.execute("""
    SELECT ts_code, trade_date,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df()
stock_ret = stock_ret.dropna(subset=['ret'])

# Merge industry
stock_ret = stock_ret.merge(ind_map, on='ts_code', how='left')
stock_ret['ind_name'] = stock_ret['ind_name'].fillna('Other')

# For HS300, use actual index
hs300 = con.execute("""
    SELECT trade_date, close / LAG(close) OVER(ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE ts_code = 'sh000300' AND trade_date >= '2016-01-01'
""").df().dropna()

# For STAR50 (科创50 proxy), use semiconductor industry
semi = stock_ret[stock_ret['ind_name'] == '电子'].groupby('trade_date')['ret'].mean().reset_index()
semi.columns = ['trade_date', 'ret']

con.close()

# Build sector return series for each chain
sector = {}
# HS300 from index
sector['sh000300'] = hs300.set_index('trade_date')['ret']
# STAR50 from semiconductor+electronics
sector['sh000688'] = semi.set_index('trade_date')['ret']

# Other sectors from industry stock averages
for code, ind_name in CHAIN_INDUSTRY_MAP.items():
    if code in ('sh000300', 'sh000688'):
        continue
    ind_ret = stock_ret[stock_ret['ind_name'] == ind_name].groupby('trade_date')['ret'].mean()
    if len(ind_ret) > 100:
        sector[code] = ind_ret

print('Computed sectors: %d' % len(sector))
for k, v in sector.items():
    print('  %s: %d days' % (k, len(v)))

macro = macro.merge(spx_df, on='trade_date', how='left')
macro['spx'] = macro['spx'].ffill()

print('Data: %d macro days, %d sectors' % (len(macro), len(sector)))

# Test each chain
print('\n%-20s %8s %8s %7s %8s %8s' % ('Chain', 'N', 'Acc%', 'Ret%', 'Lag', 'Verdict'))
print('-' * 65)

results = []
for c in CHAINS:
    cid = c['id']
    if c['leader'] not in macro.columns:
        continue
    if c['follower'] not in sector:
        continue

    ldr = macro.set_index('trade_date')[c['leader']].dropna()
    # Leader daily return
    lr = ldr.pct_change().dropna()

    flr = sector[c['follower']]  # already daily returns
    common = lr.index.intersection(flr.index)
    if len(common) < 200:
        continue
    lr = lr.loc[common]
    fr = flr.loc[common]

    lag_min, lag_max = c['lag']
    best_lag, best_acc, best_ret, best_n = None, 0, 0, 0

    for lag in range(lag_min, lag_max + 1):
        if lag == 0:
            cl = lr.index.intersection(fr.index)
            lrl, frl = lr.loc[cl], fr.loc[cl]
        else:
            lrl = lr.iloc[:-lag]
            frl = fr.iloc[lag:]
            cl = lrl.index.intersection(frl.index)
            lrl, frl = lrl.loc[cl], frl.loc[cl]

        if len(lrl) < 100:
            continue

        # Threshold trigger
        if c['leader'] in ('us10y', 'vix'):
            trigger = lrl.abs() > c['thresh'] / 100
        else:
            trigger = lrl.abs() * 100 > c['thresh']

        if trigger.sum() < 20:
            continue

        # Prediction
        if c['dir'] == '+':
            pred_up = lrl[trigger] > 0
        else:
            pred_up = lrl[trigger] < 0

        actual_up = frl[trigger] > 0
        acc = (pred_up == actual_up).mean()

        # Return when signal fires
        if c['dir'] == '+':
            pos = frl[trigger] * (2 * (lrl[trigger] > 0).astype(float) - 1)
        else:
            pos = frl[trigger] * (2 * (lrl[trigger] < 0).astype(float) - 1)
        avg_ret = pos.mean() * 100

        if acc > best_acc:
            best_acc, best_lag, best_ret, best_n = acc, lag, avg_ret, trigger.sum()

    if best_lag is None:
        continue

    if best_acc >= 0.58:
        verdict = 'STRONG'
    elif best_acc >= 0.54:
        verdict = 'WEAK'
    else:
        verdict = 'NOISE'

    results.append({**c, 'best_lag':best_lag, 'acc':best_acc, 'avg_ret':best_ret, 'n':best_n, 'verdict':verdict})
    print('%-20s %8d %7.1f%% %+7.2f%% %8d %8s' % (cid, best_n, best_acc*100, best_ret, best_lag, verdict))

# Summary
print('\n' + '=' * 70)
strong = [r for r in results if r['verdict']=='STRONG']
weak   = [r for r in results if r['verdict']=='WEAK']
noise  = [r for r in results if r['verdict']=='NOISE']

print('STRONG (>=58%%): %d  WEAK (54-58%%): %d  NOISE (<54%%): %d' % (len(strong), len(weak), len(noise)))
if strong:
    print('\nActionable signals:')
    for r in strong:
        dir_name = 'follow' if r['dir']=='+' else 'reverse'
        print('  %s: when %s moves >%.1f%%, %s sector %s within %dd (acc=%.1f%%, avg_ret=%+.2f%%)' % (
            r['id'], r['leader'], r['thresh'], dir_name, r['follower'], r['best_lag'], r['acc']*100, r['avg_ret']))

print('\nDone.')
