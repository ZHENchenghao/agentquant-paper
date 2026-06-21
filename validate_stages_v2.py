# -*- coding: utf-8 -*-
"""Validate: does Stage 4 (risk numbness) exist in the data?
Tests: returns accelerate + vol declines + drawdowns shallow + bad news ignored
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'

con = duckdb.connect(DB, read_only=True)

# Get stock data for key stocks in each cycle
CYCLES = [
    ('核心资产', ['sh600519','sz000858','sz000333','sh600276','sz002415','sh601318'], '2018-06-01','2022-06-01'),
    ('新能源',   ['sz300750','sh601012','sz300274','sz002594','sh600438','sz300014'], '2019-06-01','2023-06-01'),
    ('AI算力',   ['sz300308','sz300502','sz300394','sh688256','sh601138','sh688981'], '2022-06-01','2026-06-18'),
]

for name, stocks, start, end in CYCLES:
    print('\n' + '=' * 80)
    print('%s (%s ~ %s)' % (name, start[:4], end[:4]))
    print('=' * 80)

    # Load basket price + volume
    prices = con.execute("""
        SELECT ts_code, trade_date, close, vol
        FROM kline_daily
        WHERE ts_code IN ({}) AND trade_date BETWEEN '{}' AND '{}'
        ORDER BY trade_date
    """.format(','.join("'%s'"%s for s in stocks), start, end)).df()

    if prices.empty: continue
    prices['trade_date'] = pd.to_datetime(prices['trade_date'])

    # Equal-weight basket, monthly resample
    basket = prices.groupby('trade_date')['close'].mean()
    vol = prices.groupby('trade_date')['vol'].sum()

    monthly = basket.resample('ME').agg(['first','last','max','min'])
    monthly['ret'] = monthly['last'] / monthly['first'].shift(1) - 1
    monthly = monthly.dropna()

    # Split into halves: first half vs second half
    n = len(monthly)
    half = n // 2
    first = monthly.iloc[:half]
    second = monthly.iloc[half:]

    # === Key metrics ===
    print('  %-20s %12s %12s %12s' % ('Metric', 'First Half', 'Second Half', 'Stage4?'))
    print('  ' + '-' * 60)

    # 1. Monthly return
    ret1 = first['ret'].mean() * 100
    ret2 = second['ret'].mean() * 100
    print('  %-20s %+11.1f%% %+11.1f%% %s' % ('Avg Monthly Ret', ret1, ret2, 'YES' if ret2 > ret1 else 'no'))

    # 2. Monthly volatility
    vol1 = first['ret'].std() * 100
    vol2 = second['ret'].std() * 100
    risk_numb = 'YES' if ret2 > ret1 and vol2 < vol1 else ('partial' if ret2 > ret1 else 'no')
    print('  %-20s %+11.1f%% %+11.1f%% %s' % ('Monthly Vol', vol1, vol2, risk_numb))

    # 3. Sharpe
    sh1 = ret1 / vol1 if vol1 > 0 else 0
    sh2 = ret2 / vol2 if vol2 > 0 else 0
    print('  %-20s %11.2f %11.2f %s' % ('Monthly Sharpe', sh1, sh2, 'YES' if sh2 > sh1 else 'no'))

    # 4. Win rate
    wr1 = (first['ret'] > 0).mean() * 100
    wr2 = (second['ret'] > 0).mean() * 100
    print('  %-20s %10.0f%% %10.0f%% %s' % ('Win Rate', wr1, wr2, 'YES' if wr2 > wr1 else 'no'))

    # 5. Max monthly drawdown
    dd1 = first['ret'].min() * 100
    dd2 = second['ret'].min() * 100
    print('  %-20s %+11.1f%% %+11.1f%% %s' % ('Worst Month', dd1, dd2, 'YES' if dd2 > dd1 else 'no'))

    # 6. Max drawdown (peak to trough within period)
    cum1 = (1 + first['ret']).cumprod()
    cum2 = (1 + second['ret']).cumprod()
    mdd1 = (cum1 / cum1.cummax() - 1).min() * 100
    mdd2 = (cum2 / cum2.cummax() - 1).min() * 100
    print('  %-20s %+11.1f%% %+11.1f%% %s' % ('Max Drawdown', mdd1, mdd2, 'YES' if mdd2 > mdd1 else 'no'))

    # 7. Volume trend
    vol_monthly = vol.resample('ME').mean()
    vol_chg1 = (vol_monthly.iloc[half-1] / vol_monthly.iloc[0] - 1) * 100 if len(vol_monthly) > half else 0
    vol_chg2 = (vol_monthly.iloc[-1] / vol_monthly.iloc[half] - 1) * 100 if len(vol_monthly) > half else 0
    print('  %-20s %+11.0f%% %+11.0f%% %s' % ('Volume Change', vol_chg1, vol_chg2, 'YES' if vol_chg2 > 0 else 'no'))

    # 8. Cumulative return
    cum_ret1 = (cum1.iloc[-1] - 1) * 100
    cum_ret2 = (cum2.iloc[-1] - 1) * 100
    print('  %-20s %+11.1f%% %+11.1f%% %s' % ('Cumulative Ret', cum_ret1, cum_ret2, 'YES' if cum_ret2 > cum_ret1 else 'no'))

    # Stage assessment
    tests = [ret2 > ret1, vol2 < vol1, wr2 > wr1, dd2 > dd1, mdd2 > mdd1]
    score = sum(tests)
    risk_numbness = 'STRONG' if score >= 4 else ('MODERATE' if score >= 3 else ('WEAK' if score >= 2 else 'NONE'))
    print('\n  => Risk Numbness Evidence: %s (%d/5 tests passed)' % (risk_numbness, score))
    print('  Higher return + lower vol + higher win rate + shallower drawdowns')

con.close()
print('\nDone.')
