# -*- coding: utf-8 -*-
"""
Validate 5-stage consensus framework against real A-share data.
Test cycles: 核心资产(2019-2021), 新能源(2020-2022), AI算力(2023-至今)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

CYCLES = {
    '核心资产/茅指数': {
        'period': ('2019-01-01', '2021-12-31'),
        'stocks': ['sh600519','sz000858','sz000333','sh600276','sz002415','sh601318','sh600036'],
        'sector': '食品饮料',
        'trigger': '外资流入+确定性溢价',
        'collapse_trigger': '美10Y急升+反垄断',
    },
    '新能源': {
        'period': ('2020-01-01', '2022-12-31'),
        'stocks': ['sz300750','sh601012','sz300274','sz002594','sh600438','sz300014'],
        'sector': '电力设备',
        'trigger': '碳中和+补贴',
        'collapse_trigger': '供给过剩+补贴退坡',
    },
    'AI算力': {
        'period': ('2023-01-01', '2026-06-18'),
        'stocks': ['sz300308','sz300502','sz300394','sh688256','sh601138','sh688981'],
        'sector': '电子',
        'trigger': 'ChatGPT+英伟达',
        'collapse_trigger': '?',
    },
}

con = duckdb.connect(DB, read_only=True)

# Get industry mapping
ind_map = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn = 1
""").df()

# CSI 300 reference
hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code = 'sh000300' AND trade_date >= '2019-01-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])

# Macro for context
macro = con.execute("""
    SELECT trade_date, us10y, vix FROM macro_indicators
    WHERE trade_date >= '2019-01-01' ORDER BY trade_date
""").df()

con.close()

print('=' * 80)
print('Validate: 5-Stage Consensus Framework')
print('=' * 80)

for cycle_name, cfg in CYCLES.items():
    print('\n' + '=' * 80)
    print('CYCLE: %s (%s ~ %s)' % (cycle_name, cfg['period'][0], cfg['period'][1]))
    print('=' * 80)

    start, end = cfg['period']

    # Get basket stock data
    con2 = duckdb.connect(DB, read_only=True)
    prices = con2.execute("""
        SELECT ts_code, trade_date, close, vol
        FROM kline_daily
        WHERE ts_code IN ({}) AND trade_date BETWEEN '{}' AND '{}'
        ORDER BY trade_date
    """.format(','.join("'%s'" % s for s in cfg['stocks']), start, end)).df()
    con2.close()

    if prices.empty: continue

    prices['trade_date'] = pd.to_datetime(prices['trade_date'])

    # Equal-weight basket
    basket = prices.groupby('trade_date')['close'].mean()
    vol_basket = prices.groupby('trade_date')['vol'].sum()

    # Monthly aggregation for cleaner signal
    basket_monthly = basket.resample('M').last()
    basket_ret = basket_monthly.pct_change()

    # Cumulative return
    cum_ret = (basket / basket.iloc[0] - 1) * 100

    # CSI 300 comparison
    hs300_period = hs300[(hs300['trade_date'] >= pd.Timestamp(start)) & (hs300['trade_date'] <= pd.Timestamp(end))]
    hs300_ret = (hs300_period.set_index('trade_date')['close'] / hs300_period['close'].iloc[0] - 1) * 100

    # === Stage identification ===
    # Break the cycle into stages based on return characteristics
    # Simplified: split into quartiles and analyze each

    n = len(cum_ret)
    stages_data = []

    # Stage detection: use drawdown/return patterns
    cum_vals = cum_ret.values
    peak_val = cum_vals[0]
    peak_idx = 0
    stage = 0

    for i in range(1, len(cum_vals)):
        if cum_vals[i] > peak_val * 1.1 + 10:  # New peak with 10% gap
            # Entered new stage
            if stage < 5:
                stages_data.append({
                    'stage': stage + 1,
                    'start': cum_ret.index[peak_idx],
                    'end': cum_ret.index[i],
                    'return': round(cum_vals[i] - cum_vals[peak_idx], 1),
                })
            stage += 1
            peak_val = cum_vals[i]
            peak_idx = i

    # Last stage
    if stage < 5:
        stages_data.append({
            'stage': stage + 1,
            'start': cum_ret.index[peak_idx],
            'end': cum_ret.index[-1],
            'return': round(cum_vals[-1] - cum_vals[peak_idx], 1),
        })

    # === Key metrics per stage ===
    print('\n  Identified stages:')
    for sd in stages_data[:5]:
        s_start = sd['start']
        s_end = sd['end']

        # Volume change
        vol_start = float(vol_basket.loc[s_start:s_end].iloc[:5].mean()) if s_start in vol_basket.index else 0
        vol_end = float(vol_basket.loc[s_start:s_end].iloc[-5:].mean()) if s_end in vol_basket.index else 0
        vol_change = (vol_end / vol_start - 1) * 100 if vol_start > 0 else 0

        # Monthly return volatility (risk numbness = low vol despite high returns?)
        mo_ret = basket.loc[s_start:s_end].resample('M').last().pct_change().dropna()
        mo_vol = mo_ret.std() * np.sqrt(12) * 100 if len(mo_ret) > 1 else 0
        mo_sharpe = mo_ret.mean() / mo_ret.std() * np.sqrt(12) if len(mo_ret) > 1 and mo_ret.std() > 0 else 0

        # Max drawdown in stage
        peak_in_stage = basket.loc[s_start:s_end].cummax()
        dd_in_stage = (basket.loc[s_start:s_end] / peak_in_stage - 1).min() * 100

        stage_name = ['','驱动萌芽','预期差','资金共振','风险钝化','共识松动'][min(sd['stage'],5)]

        print('  Stage%d %s: %s->%s ret=%+.0f%% vol_chg=%+.0f%% vol=%.0f%% Sharpe=%.1f maxDD=%.0f%%' % (
            sd['stage'], stage_name,
            str(sd['start'].date())[:7], str(sd['end'].date())[:7],
            sd['return'], vol_change, mo_vol, mo_sharpe, dd_in_stage))

    # === Stage 4 vs Stage 5 comparison ===
    if len(stages_data) >= 2:
        last_two = stages_data[-2:]

        # Check: does Stage 4 show higher returns + lower vol (risk numbness)?
        print('\n  Key test: Stage 4 (risk numbness) characteristics:')
        for sd in last_two:
            s_start = sd['start']
            s_end = sd['end']
            mo_ret = basket.loc[s_start:s_end].resample('M').last().pct_change().dropna()
            pos_months = (mo_ret > 0).mean() * 100
            best_month = mo_ret.max() * 100
            worst_month = mo_ret.min() * 100
            print('  Stage%d (%s): months=%d pos_rate=%.0f%% best=+%.0f%% worst=%.0f%%' % (
                sd['stage'], str(s_start.date())[:7], len(mo_ret), pos_months, best_month, worst_month))

    # Sector outperformance
    sector_ret = cum_ret.iloc[-1]
    hs300_ret_val = hs300_ret.iloc[-1] if len(hs300_ret) > 0 else 0
    print('\n  Full cycle: basket %+.0f%% vs CSI300 %+.0f%% = excess %+.0f%%' % (
        sector_ret, hs300_ret_val, sector_ret - hs300_ret_val))
    print('  Trigger: %s' % cfg['trigger'])
    print('  Collapse: %s' % cfg['collapse_trigger'])

print('\nDone.')
