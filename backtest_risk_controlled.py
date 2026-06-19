# -*- coding: utf-8 -*-
"""
Risk-Controlled Backtest — Post-Processing Layer
=================================================
在production_final基础上, 叠加MA200择时后处理:
- 读production_final_monthly.parquet
- 对熊市月收益归零 (MA200 regime)
- 输出风险控制版 vs 原版对比

运行: python backtest_risk_controlled.py
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
print('=' * 80)
print('Risk-Controlled Layer: MA200 Regime Filter + Momentum Inflection')
print('=' * 80)

# Load original monthly results
orig = pd.read_parquet('cache/production_final_monthly.parquet')

# Compute MA200 regime from CSI300
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date BETWEEN '2005-06-01' AND '2026-06-19'
    ORDER BY trade_date
""").df()
con.close()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300_px = hs300.set_index('trade_date')['close']

ma200 = hs300_px.rolling(200, min_periods=200).mean()
slope = ma200.pct_change(5)
raw_signal = (hs300_px > ma200) & (slope > -0.01)

initial = raw_signal.iloc[:3].mean() >= 0.5
regime = pd.Series('BULL' if initial else 'BEAR', index=raw_signal.index)
bull_s = 0; bear_s = 0; curr = regime.iloc[0]
for i in range(len(raw_signal)):
    if raw_signal.iloc[i]: bull_s += 1; bear_s = 0
    else: bear_s += 1; bull_s = 0
    if bull_s >= 3 and curr != 'BULL': curr = 'BULL'
    if bear_s >= 3 and curr != 'BEAR': curr = 'BEAR'
    regime.iloc[i] = curr

monthly_regime = (regime == 'BEAR').resample('M').mean()

# Apply MA200 filter: bear months → zero return
orig_ret = orig.copy()
# month column is like '2008-01' → pd.Period → end_time
orig_ret['month_dt'] = orig_ret['month'].apply(lambda x: pd.Period(x).end_time)
orig_ret = orig_ret.set_index('month_dt').sort_index()

# Align: monthly_regime is DatetimeIndex from resample('M')
aligned = monthly_regime.reindex(orig_ret.index, method='nearest')
bear_mask = aligned >= 0.5

filtered_ret = orig_ret['ret'].copy()
filtered_ret[bear_mask] = 0  # 熊市月归零(现金)

filtered_gross = orig_ret['ret_gross'].copy()
filtered_gross[bear_mask] = 0

# === Compare ===
def calc_stats(rets_series):
    r = rets_series.dropna().values
    if len(r) < 3: return {'ann':0,'sh':0,'mdd':0}
    ann = np.mean(r) * 12
    sh = ann / (np.std(r)*np.sqrt(12)) if np.std(r)>0 else 0
    cum = np.cumprod(1+r)
    mdd = np.min(cum / np.maximum.accumulate(cum) - 1)
    return {'ann':ann*100,'sh':sh,'mdd':mdd*100}

orig_stats = calc_stats(orig_ret['ret'])
filt_stats = calc_stats(filtered_ret)

print(f'\n  {"Metric":<20} {"Original":>12} {"+MA200":>12} {"Delta":>10}')
print('  ' + '-'*56)
for label, o_key, f_key in [
    ('Annual Return', 'ann', 'ann'),
    ('Sharpe', 'sh', 'sh'),
    ('MDD', 'mdd', 'mdd'),
]:
    o = orig_stats[o_key]; f = filt_stats[f_key]
    d = f - o
    print(f'  {label:<20} {o:>+11.1f}% {f:>+11.1f}% {d:>+10.1f}%' if 'Return' in label or 'MDD' in label else
          f'  {label:<20} {o:>11.2f} {f:>11.2f} {d:>+10.2f}')

# 分年
print(f'\n  {"Year":<6} {"Original":>10} {"+MA200":>10} {"Bear%":>8}')
print('  ' + '-'*38)
orig_yearly = orig_ret['ret'].groupby(orig_ret.index.year).mean() * 12
filt_yearly = filtered_ret.groupby(filtered_ret.index.year).mean() * 12

for yr in sorted(set(orig_yearly.index) | set(filt_yearly.index)):
    o = orig_yearly.get(yr, 0)
    f = filt_yearly.get(yr, 0)
    b = monthly_regime[monthly_regime.index.year == yr]
    bpct = b.mean()*100 if len(b)>0 else 0
    print(f'  {yr:<6} {o*100:>+9.0f}% {f*100:>+9.0f}% {bpct:>7.0f}%')

# 近5年
late = [yr for yr in range(2020,2025)]
late_orig = np.mean([orig_yearly.get(yr,0) for yr in late])
late_filt = np.mean([filt_yearly.get(yr,0) for yr in late])
print(f'\n  Late 5yr (20-24): Orig={late_orig:+.0f}% MA200={late_filt:+.0f}%')

elapsed = time.time() - t0
print(f'\n  Time: {elapsed:.0f}s')
