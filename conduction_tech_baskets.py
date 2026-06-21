# -*- coding: utf-8 -*-
"""
AI/Tech supply chain conduction backtest.
费城半导体/NASDAQ -> A股 AI产业链各环节
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

# ============================================================
# AI/半导体 产业链篮子
# ============================================================

BASKETS = {
    '光模块CPO': {
        'desc': '数据中心互联: 800G/1.6T光模块。英伟达GPU出货->光模块需求。A股最纯的AI标的。',
        'stocks': ['sz300308', 'sz300502', 'sz300394', 'sz002281', 'sz300570'],
    },
    'AI服务器': {
        'desc': '算力基础设施: AI服务器组装。工业富联给英伟达代工GPU模块。',
        'stocks': ['sh601138', 'sz000977', 'sh603019', 'sz000938'],
    },
    'PCB载板': {
        'desc': '高端PCB/IC载板: AI服务器用高多层板。沪电/深南是英伟达PCB供应商。',
        'stocks': ['sz002463', 'sz002916', 'sz002938', 'sz300476', 'sz002384'],
    },
    '芯片设计': {
        'desc': 'AI芯片设计: 寒武纪(训练)/海光(x86兼容)/GPU国产替代。直接对标英伟达。',
        'stocks': ['sh688256', 'sh688041', 'sh603501', 'sh603986', 'sh688008'],
    },
    '半导体设备': {
        'desc': '晶圆制造设备: 中微(刻蚀)/北方华创(沉积)/盛美(清洗)。受益于全球扩产周期。',
        'stocks': ['sh688012', 'sz002371', 'sh688082', 'sh688120'],
    },
    '先进封装': {
        'desc': 'Chiplet/CoWoS封装: 长电/通富/华天。AI芯片封装需求爆发。',
        'stocks': ['sh600584', 'sz002156', 'sz002185', 'sh688981'],
    },
    '存储芯片': {
        'desc': 'HBM/DRAM: AI算力瓶颈在存储带宽。HBM供给决定GPU出货量。',
        'stocks': ['sz002049', 'sh603893', 'sh688525', 'sz300672'],
    },
}

# Also add some comparison baskets
# 纳斯达克 -> A股科技 (broad)
BASKETS['创业板权重'] = {
    'desc': '创业板大市值: 宁德/迈瑞/东财/汇川。纳指情绪传导到A股成长股。',
    'stocks': ['sz300750', 'sz300760', 'sz300059', 'sz300124'],
}
BASKETS['科创50权重'] = {
    'desc': '科创50核心: 半导体设备+芯片+生物医药。纳指->科创50传导。',
    'stocks': ['sh688981', 'sh688012', 'sh688041', 'sh688256', 'sh688111'],
}

# ============================================================
# Test pairs: macro -> basket
# ============================================================

TESTS = [
    # SOX -> AI supply chain
    ('sox', '光模块CPO', '+', 0, 3, 1.5),
    ('sox', 'AI服务器', '+', 0, 3, 1.5),
    ('sox', 'PCB载板', '+', 0, 3, 1.5),
    ('sox', '芯片设计', '+', 0, 3, 1.5),
    ('sox', '半导体设备', '+', 0, 5, 1.5),
    ('sox', '先进封装', '+', 0, 3, 1.5),
    ('sox', '存储芯片', '+', 0, 3, 1.5),
    # NASDAQ -> broad tech
    ('nasdaq', '光模块CPO', '+', 0, 3, 1.5),
    ('nasdaq', '科创50权重', '+', 0, 3, 1.0),
    ('nasdaq', '创业板权重', '+', 0, 3, 1.0),
]

print('=' * 80)
print('AI Supply Chain Conduction Backtest')
print('=' * 80)

con = duckdb.connect(DB, read_only=True)

# SOX and NASDAQ data
sox = con.execute("""
    SELECT trade_date, close FROM global_index_daily
    WHERE index_code = '.SOX' AND trade_date >= '2019-01-01' ORDER BY trade_date
""").df()
sox.columns = ['trade_date', 'sox']

nasdaq = con.execute("""
    SELECT trade_date, close FROM global_index_daily
    WHERE index_code = '.IXIC' AND trade_date >= '2019-01-01' ORDER BY trade_date
""").df()
nasdaq.columns = ['trade_date', 'nasdaq']

# All stock daily returns (only basket stocks)
all_stocks = set()
for b in BASKETS.values():
    all_stocks.update(b['stocks'])

stock_ret = con.execute("""
    SELECT ts_code, trade_date,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE trade_date >= '2019-01-01'
""").df().dropna(subset=['ret'])

con.close()

stock_ret = stock_ret[stock_ret['ts_code'].isin(all_stocks)]
found = stock_ret['ts_code'].nunique()
print('Basket stocks found: %d/%d' % (found, len(all_stocks)))

# ============================================================
# Test
# ============================================================
results = []

def run_test(leader_df, leader_col, basket_name, direction, lag_min, lag_max, threshold):
    if basket_name not in BASKETS:
        return None
    basket_stocks = BASKETS[basket_name]['stocks']
    basket_ret = stock_ret[stock_ret['ts_code'].isin(basket_stocks)]
    if len(basket_ret) < 100: return None

    basket_daily = basket_ret.groupby('trade_date')['ret'].mean()

    ldr = leader_df.set_index('trade_date')[leader_col]
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

        if len(lrl) < 50: continue

        trigger = lrl.abs() * 100 > threshold
        n_signals = trigger.sum()
        if n_signals < 10: continue

        if direction == '+':
            pred_up = lrl[trigger] > 0
            pos = frl[trigger] * (2 * (lrl[trigger] > 0).astype(float) - 1)
        else:
            pred_up = lrl[trigger] < 0
            pos = frl[trigger] * (2 * (lrl[trigger] < 0).astype(float) - 1)

        actual_up = frl[trigger] > 0
        acc = (pred_up == actual_up).mean()

        avg_ret = pos.mean() * 10000
        tstat = abs(pos.mean()) / (pos.std() / np.sqrt(len(pos))) if pos.std() > 0 else 0

        if acc > best_acc:
            best_acc, best_lag, best_ret, best_t, best_n = acc, lag, avg_ret, tstat, n_signals

    if best_lag is None: return None

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

    return {
        'leader': leader_col, 'basket': basket_name,
        'n': best_n, 'lag': best_lag, 'acc': best_acc, 'ret_bp': best_ret,
        't': best_t, 'verdict': verdict,
    }

# Build leader data dict
leaders = {'sox': (sox, 'sox'), 'nasdaq': (nasdaq, 'nasdaq')}

for leader_col, basket_name, direction, lag_min, lag_max, threshold in TESTS:
    if leader_col not in leaders: continue
    leader_df, col = leaders[leader_col]

    r = run_test(leader_df, col, basket_name, direction, lag_min, lag_max, threshold)
    if r is None: continue
    results.append(r)

    test_name = '%s -> %s' % (leader_col.upper(), basket_name)
    print('%-35s n=%-5d lag=%-3d acc=%-5.1f%% ret=%+7.1fbp t=%-5.2f %s' % (
        test_name, r['n'], r['lag'], r['acc']*100, r['ret_bp'], r['t'], r['verdict']))

# ============================================================
# Summary
# ============================================================
print('\n' + '=' * 80)
print('AI Supply Chain Results')
print('=' * 80)

for v in ['STRONG', 'VALID', 'WEAK', 'MARGINAL', 'NOISE']:
    items = [r for r in results if r['verdict'] == v]
    if items:
        print('%s (%d):' % (v, len(items)))
        for r in sorted(items, key=lambda x: -x['acc']):
            print('  %s -> %s  n=%d lag=%dd acc=%.1f%% ret=%+.1fbp t=%.2f  %s' % (
                r['leader'].upper(), r['basket'], r['n'], r['lag'],
                r['acc']*100, r['ret_bp'], r['t'], BASKETS[r['basket']]['desc'][:60]))

# Save
df_results = pd.DataFrame(results)
df_results.to_parquet('cache/tech_conduction_results.parquet')
print('\nSaved to cache/tech_conduction_results.parquet')
print('Done.')
