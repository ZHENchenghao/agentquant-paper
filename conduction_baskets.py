# -*- coding: utf-8 -*-
"""
Custom stock baskets for conduction analysis.
Each basket = stocks with SAME economic exposure to a macro variable.
NOT the same Shenwan industry.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

# ============================================================
# Economic baskets: stocks grouped by exposure, not by classification
# ============================================================

BASKETS = {
    # ── Petroleum chain ──
    '石油开采(上游)': {
        'description': 'WTI up = profit up (elasticity 2-3x). Direct revenue channel.',
        'stocks': ['sh601857', 'sh600028', 'sh600938', 'sh601808', 'sh600583'],
    },
    '石油炼化(中下游)': {
        'description': 'WTI up = cost up but pass-through to products. Net effect: mild positive.',
        'stocks': ['sh600346', 'sh000301', 'sh600688', 'sz002493'],
    },
    '航空': {
        'description': 'WTI up = jet fuel cost up (30-35% of opex). Most sensitive victim.',
        'stocks': ['sh600029', 'sh601111', 'sh600115', 'sh601021', 'sh603885'],
    },
    '物流运输': {
        'description': 'WTI up = diesel cost up (20-25% of opex). Less elastic than airlines.',
        'stocks': ['sh002352', 'sh600233', 'sh603056', 'sh600057'],
    },
    '化纤': {
        'description': 'WTI up = PX/PTA/EG cost up -> margin squeeze. Lag 7-14d.',
        'stocks': ['sh600346', 'sh601233', 'sh000703', 'sz002493'],
    },

    # ── Metal chain ──
    '黄金矿业': {
        'description': 'Gold up = profit up (leverage 2.5-3x). Pure play on gold price.',
        'stocks': ['sh601899', 'sh600547', 'sh600489', 'sz002155', 'sh600988', 'sz000975'],
    },
    '铜矿': {
        'description': 'Copper up = mining profit up. Dr.Copper - economic bellwether.',
        'stocks': ['sh600362', 'sz000630', 'sh601168', 'sz002203', 'sh603799'],
    },
    '铝业': {
        'description': 'Aluminum up = profit up. Different from gold/copper - not macro-sensitive.',
        'stocks': ['sh601600', 'sh000807', 'sz002532', 'sh603993'],
    },
    '钢铁': {
        'description': 'Iron ore / rebar driven. Iron ore up = cost pressure for converters.',
        'stocks': ['sh600019', 'sh600010', 'sh000932', 'sh600585'],
    },

    # ── Interest rate sensitive ──
    '国有大行': {
        'description': 'Rate up = NIM expansion = profit up. But rate too fast = bond losses.',
        'stocks': ['sh601398', 'sh601939', 'sh601288', 'sh601988', 'sh601328'],
    },
    '股份制银行': {
        'description': 'More rate sensitive than big banks. Higher beta to rate cycle.',
        'stocks': ['sh600036', 'sh601166', 'sh000001', 'sh002142', 'sh600015'],
    },
    '高估值成长': {
        'description': 'Rate up = DCF denominator up = valuation compression. Long duration.',
        'stocks': ['sh688981', 'sh688012', 'sz300750', 'sh688111', 'sz002371', 'sh603501'],
    },

    # ── VIX / risk-off ──
    '北向重仓': {
        'description': 'VIX up = foreign outflow = these get sold first.',
        'stocks': ['sh600519', 'sh000858', 'sh601318', 'sh600036', 'sh000333'],
    },
    '高股息防御': {
        'description': 'VIX up = flight to safety = dividend plays benefit.',
        'stocks': ['sh601088', 'sh600900', 'sh601857', 'sh601398', 'sh600585'],
    },
    '高Beta科技': {
        'description': 'VIX up = risk-off = high beta gets crushed hardest.',
        'stocks': ['sh688981', 'sz300750', 'sh688111', 'sz300274', 'sh688012'],
    },
}

# ============================================================
# Backtest each basket against macro variables
# ============================================================

# Define which macro->basket pairs to test
TESTS = [
    # WTI chains
    ('wti', '石油开采(上游)', '+', 0, 5, 3.0),
    ('wti', '石油炼化(中下游)', '+', 0, 5, 3.0),
    ('wti', '航空', '-', 1, 8, 3.0),
    ('wti', '物流运输', '-', 1, 8, 3.0),
    ('wti', '化纤', '-', 3, 15, 5.0),
    # Gold chains
    ('gold', '黄金矿业', '+', 0, 3, 1.5),
    # Copper chains
    ('copper', '铜矿', '+', 1, 7, 2.0),
    ('copper', '铝业', '+', 1, 7, 2.0),
    # Rate chains
    ('us10y', '国有大行', '+', 1, 8, 0.10),
    ('us10y', '股份制银行', '+', 1, 8, 0.10),
    ('us10y', '高估值成长', '-', 1, 8, 0.15),
    # VIX chains
    ('vix', '北向重仓', '-', 0, 5, 3.0),
    ('vix', '高股息防御', '+', 0, 5, 3.0),
    ('vix', '高Beta科技', '-', 0, 5, 3.0),
]

print('=' * 80)
print('Custom Basket Conduction Backtest')
print('=' * 80)

con = duckdb.connect(DB, read_only=True)

# Macro
macro = con.execute("""
    SELECT trade_date, wti, copper, gold, us10y, vix
    FROM macro_indicators WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
for c in ['wti', 'copper', 'gold', 'us10y', 'vix']:
    macro[c] = macro[c].ffill().bfill()

# All stock daily returns
stock_ret = con.execute("""
    SELECT ts_code, trade_date,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df().dropna(subset=['ret'])

con.close()

# Filter to only stocks in our baskets
all_basket_stocks = set()
for b in BASKETS.values():
    all_basket_stocks.update(b['stocks'])

stock_ret = stock_ret[stock_ret['ts_code'].isin(all_basket_stocks)]
found = stock_ret['ts_code'].nunique()
print('Basket stocks found in data: %d/%d' % (found, len(all_basket_stocks)))

# Test each pair
print('\n%-30s %6s %6s %8s %8s %8s %8s' % (
    'Test', 'N', 'Lag', 'Acc%', 'Ret(bp)', 'T-stat', 'Verdict'))
print('-' * 85)

results = []

for macro_src, basket_name, direction, lag_min, lag_max, threshold in TESTS:
    if basket_name not in BASKETS:
        continue

    basket_stocks = BASKETS[basket_name]['stocks']
    basket_ret = stock_ret[stock_ret['ts_code'].isin(basket_stocks)]

    if len(basket_ret) < 100:
        continue

    # Equal-weight basket daily return
    basket_daily = basket_ret.groupby('trade_date')['ret'].mean()

    # Leader return
    ldr = macro.set_index('trade_date')[macro_src]
    lr = ldr.pct_change().dropna()

    common = lr.index.intersection(basket_daily.index)
    lr = lr.loc[common]
    fr = basket_daily.loc[common]

    best_lag, best_acc, best_ret, best_t, best_n = None, 0, 0, 0, 0

    for lag in range(lag_min, lag_max + 1):
        if lag == 0:
            cl = lr.index.intersection(fr.index)
            lrl, frl = lr.loc[cl], fr.loc[cl]
        else:
            if len(lr) <= lag: continue
            lrl = lr.iloc[:-lag]
            frl = fr.iloc[lag:]
            cl = lrl.index.intersection(frl.index)
            lrl, frl = lrl.loc[cl], frl.loc[cl]

        if len(lrl) < 100: continue

        # Threshold
        if macro_src in ('us10y', 'vix'):
            trigger = lrl.abs() > threshold / 100
        else:
            trigger = lrl.abs() * 100 > threshold

        n_signals = trigger.sum()
        if n_signals < 15: continue

        # Prediction
        if direction == '+':
            pred_up = lrl[trigger] > 0
            pos = frl[trigger] * (2 * (lrl[trigger] > 0).astype(float) - 1)
        else:
            pred_up = lrl[trigger] < 0
            pos = frl[trigger] * (2 * (lrl[trigger] < 0).astype(float) - 1)

        actual_up = frl[trigger] > 0
        acc = (pred_up == actual_up).mean()

        avg_ret = pos.mean() * 10000  # bp
        tstat = pos.mean() / (pos.std() / np.sqrt(len(pos))) if pos.std() > 0 else 0

        if acc > best_acc:
            best_acc, best_lag, best_ret, best_t, best_n = acc, lag, avg_ret, tstat, n_signals

    if best_lag is None: continue

    if best_acc >= 0.62 and best_t > 2.0:
        verdict = 'STRONG'
    elif best_acc >= 0.58 and best_t > 1.5:
        verdict = 'VALID'
    elif best_acc >= 0.55:
        verdict = 'WEAK'
    elif best_acc >= 0.52:
        verdict = 'MARGINAL'
    else:
        verdict = 'NOISE'

    test_name = '%s->%s' % (macro_src.upper(), basket_name)
    results.append({
        'test': test_name, 'macro': macro_src, 'basket': basket_name,
        'n': best_n, 'lag': best_lag, 'acc': best_acc, 'ret_bp': best_ret,
        't': best_t, 'verdict': verdict,
    })

    print('%-30s %6d %6d %7.1f%% %+8.1f %8.2f %8s' % (
        test_name[:29], best_n, best_lag, best_acc*100, best_ret, best_t, verdict))

# Summary
print('\n' + '=' * 80)
for v in ['STRONG', 'VALID', 'WEAK', 'MARGINAL', 'NOISE']:
    items = [r for r in results if r['verdict'] == v]
    if items:
        print('%s (%d):' % (v, len(items)))
        for r in items:
            print('  %s  n=%d lag=%dd acc=%.1f%% ret=%+.1fbp t=%.2f' % (
                r['test'], r['n'], r['lag'], r['acc']*100, r['ret_bp'], r['t']))

print('\nDone.')
