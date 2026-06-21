# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

print('=' * 80)
print('Conduction Table Backtest: 25 links')
print('=' * 80)

# Load table
links = pd.read_parquet('cache/conduction_table.parquet')
print('Links loaded: %d' % len(links))

# Data
con = duckdb.connect(DB, read_only=True)

# Macro
macro = con.execute("""
    SELECT trade_date, wti, copper, gold, us10y, vix
    FROM macro_indicators WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
for c in ['wti', 'copper', 'gold', 'us10y', 'vix']:
    macro[c] = macro[c].ffill().bfill()

# SPX
spx = con.execute("""
    SELECT trade_date, MAX(close) AS spx FROM global_index_daily
    WHERE index_code = '.INX' AND trade_date >= '2016-01-01'
    GROUP BY trade_date ORDER BY trade_date
""").df()
macro = macro.merge(spx, on='trade_date', how='left')
macro['spx'] = macro['spx'].ffill().bfill()

# Industry returns
stock_ret = con.execute("""
    SELECT ts_code, trade_date,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df().dropna(subset=['ret'])

ind_map = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map
    ) WHERE rn = 1
""").df()

stock_ret = stock_ret.merge(ind_map, on='ts_code', how='left')
stock_ret['ind_name'] = stock_ret['ind_name'].fillna('Other')

# Pre-compute all industry daily returns
ind_rets = stock_ret.groupby(['trade_date', 'ind_name'])['ret'].mean().reset_index()
con.close()

print('Data ready: %d macro days, %d industry-day rows' % (len(macro), len(ind_rets)))

# ============================================================
# Backtest each link
# ============================================================
print('\n%-28s %6s %6s %8s %8s %8s %8s %8s' % (
    'Link', 'N', 'Lag', 'Acc%', 'Ret(bp)', 'T-stat', 'OrigConf', 'Result'))
print('-' * 90)

results = []

for idx, link in links.iterrows():
    macro_src = link['macro_source']
    industry = link['industry']
    direction = link['direction']
    lag_min, lag_max = int(link['lag_min']), int(link['lag_max'])
    threshold = link['threshold_pct']
    orig_conf = link['confidence']

    link_name = '%s->%s' % (link['macro_var'], link['industry'])

    if macro_src not in macro.columns:
        continue

    # Get industry return series
    ind_ret = ind_rets[ind_rets['ind_name'] == industry].set_index('trade_date')['ret']
    if len(ind_ret) < 200:
        continue

    # Leader daily return
    ldr = macro.set_index('trade_date')[macro_src]
    lr = ldr.pct_change().dropna()

    common = lr.index.intersection(ind_ret.index)
    lr = lr.loc[common]
    fr = ind_ret.loc[common]

    best_lag, best_acc, best_ret, best_tstat, best_n = None, 0, 0, 0, 0

    for lag in range(lag_min, lag_max + 1):
        if lag == 0:
            cl = lr.index.intersection(fr.index)
            lrl, frl = lr.loc[cl], fr.loc[cl]
        else:
            if len(lr) <= lag:
                continue
            lrl = lr.iloc[:-lag]
            frl = fr.iloc[lag:]
            cl = lrl.index.intersection(frl.index)
            lrl, frl = lrl.loc[cl], frl.loc[cl]

        if len(lrl) < 100:
            continue

        # Threshold: macro variable dependent
        if macro_src in ('us10y',):
            trigger = lrl.abs() > threshold / 100  # bp to decimal
        elif macro_src == 'vix':
            trigger = lrl.abs() > threshold / 100  # points to %
        else:
            trigger = lrl.abs() * 100 > threshold

        n_signals = trigger.sum()
        if n_signals < 20:
            continue

        # Predicted up/down based on direction
        if direction == 1:
            pred_up = lrl[trigger] > 0
        else:
            pred_up = lrl[trigger] < 0

        actual_up = frl[trigger] > 0
        acc = (pred_up == actual_up).mean()

        if acc < 0.45:  # worse than random, try opposite direction
            continue

        # Position returns
        if direction == 1:
            pos = frl[trigger] * (2 * (lrl[trigger] > 0).astype(float) - 1)
        else:
            pos = frl[trigger] * (2 * (lrl[trigger] < 0).astype(float) - 1)

        avg_ret = pos.mean() * 10000  # bp
        tstat = pos.mean() / (pos.std() / np.sqrt(len(pos))) if pos.std() > 0 else 0

        if acc > best_acc:
            best_acc, best_lag, best_ret, best_tstat, best_n = acc, lag, avg_ret, tstat, n_signals

    if best_lag is None:
        continue

    # Verdict
    if best_acc >= 0.60 and best_tstat > 1.5:
        result = 'VALIDATED'
    elif best_acc >= 0.55:
        result = 'WEAK'
    elif best_acc >= 0.50:
        result = 'MARGINAL'
    else:
        result = 'REJECTED'

    results.append({
        'link_id': idx,
        'link_name': link_name,
        'best_lag': best_lag,
        'accuracy': round(best_acc, 3),
        'avg_ret_bp': round(best_ret, 1),
        't_stat': round(best_tstat, 2),
        'n_signals': best_n,
        'orig_confidence': orig_conf,
        'result': result,
    })

    print('%-28s %6d %6d %7.1f%% %+8.1f %8.2f %8s %8s' % (
        link_name[:27], best_n, best_lag, best_acc*100, best_ret,
        best_tstat, orig_conf, result))

# ============================================================
# Summary
# ============================================================
print('\n' + '=' * 80)
res_df = pd.DataFrame(results)

for r in ['VALIDATED', 'WEAK', 'MARGINAL', 'REJECTED']:
    n = (res_df['result'] == r).sum()
    if n > 0:
        sub = res_df[res_df['result'] == r]
        print('%s (%d):' % (r, n))
        for _, row in sub.iterrows():
            print('  %s: lag=%dd acc=%.1f%% ret=%+.1fbp t=%.2f' % (
                row['link_name'], row['best_lag'], row['accuracy']*100,
                row['avg_ret_bp'], row['t_stat']))

# Save
res_df.to_parquet('cache/conduction_backtest_results.parquet')
print('\nSaved to cache/conduction_backtest_results.parquet')
print('Done.')
