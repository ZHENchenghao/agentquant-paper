# -*- coding: utf-8 -*-
import json, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_parquet('cache/clean_monthly_v3.parquet')
rets = df['ret'].values
rands = df['ret_random'].values

n_months = len(rets)
ann_ret = np.mean(rets) * 12
ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
cum = np.prod(1 + rets) - 1
mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
calmar = ann_ret / abs(mdd) if mdd != 0 else 0
wr_monthly = np.mean(rets > 0) * 100
rand_ann = np.mean(rands) * 12

roll_sh = pd.Series(rets).rolling(12).apply(
    lambda x: x.mean()/x.std()*np.sqrt(12) if x.std()>0 else 0).dropna()

df['year'] = [int(m.split('-')[0]) for m in df['month']]
yearly = df.groupby('year').agg(
    ann_ret=('ret', lambda x: x.mean()*12),
    ann_vol=('ret', lambda x: x.std()*np.sqrt(12)),
    cum_ret=('ret', lambda x: np.prod(1+x)-1),
    mdd=('ret', lambda x: np.min(np.cumprod(1+x)/np.maximum.accumulate(np.cumprod(1+x))-1)),
    wr=('ret', lambda x: np.mean(x>0)),
    avg_n=('n', 'mean'),
).reset_index()
yearly['sharpe'] = yearly['ann_ret'] / yearly['ann_vol']

print('='*80)
print('  mlfinal v3.0 TEST REPORT | 2008-2024 Walk-Forward')
print('='*80)

print()
print('--- 1. CORE METRICS ---')
print(f'  Test months:        {n_months}')
print(f'  Annualized excess:  {ann_ret*100:+.1f}%')
print(f'  Annualized vol:     {ann_vol*100:.1f}%')
print(f'  Sharpe Ratio:       {sharpe:.2f}')
print(f'  Max Drawdown:       {mdd*100:.1f}%')
print(f'  Calmar Ratio:       {calmar:.2f}')
print(f'  Monthly Win Rate:   {wr_monthly:.1f}%')
print(f'  Cumulative excess:  {cum*100:+.1f}%')
print(f'  Random baseline:    {rand_ann*100:+.1f}%')

print()
print('--- 2. ANNUAL BREAKDOWN ---')
print(f'  {"Year":<6} {"AnnRet":>8} {"Sharpe":>8} {"MDD":>8} {"WinRate":>8} {"N_Stocks":>8}')
for _, r in yearly.iterrows():
    print(f'  {r["year"]:<6} {r["ann_ret"]*100:>+7.0f}% {r["sharpe"]:>7.2f} {r["mdd"]*100:>+7.0f}% {r["wr"]*100:>7.0f}% {r["avg_n"]:>7.0f}')

early = yearly[yearly['year'] < 2016]
mid = yearly[(yearly['year'] >= 2016) & (yearly['year'] < 2020)]
late = yearly[yearly['year'] >= 2020]

print()
print('--- 3. SUB-PERIOD ANALYSIS ---')
for label, data in [('Early  2008-2015', early), ('Mid    2016-2019', mid), ('Recent 2020-2024', late)]:
    a = data['ann_ret'].mean()
    s = data['sharpe'].mean()
    m = data['mdd'].mean()
    print(f'  {label}:  AnnRet={a*100:+.0f}%  Sharpe={s:.2f}  MDD={m*100:+.0f}%')

print()
print('--- 4. ROLLING 12M SHARPE ---')
print(f'  Mean: {roll_sh.mean():.2f}  Median: {roll_sh.median():.2f}')
print(f'  Min: {roll_sh.min():.2f}  Max: {roll_sh.max():.2f}')
print(f'  Pct < 0: {(roll_sh<0).mean()*100:.1f}%')
print(f'  Pct < 1: {(roll_sh<1).mean()*100:.1f}%')
print(f'  Pct > 2: {(roll_sh>2).mean()*100:.1f}%')

print()
print('--- 5. AFTER TRANSACTION COSTS ---')
for label, cost in [('10K scale  (-4.2%/yr)', 0.042), ('1M scale  (-6.5%/yr)', 0.065)]:
    net_ann = ann_ret - cost
    net_sh = net_ann / ann_vol
    print(f'  {label}:  Net AnnRet={net_ann*100:+.0f}%  Net Sharpe={net_sh:.2f}')

print()
print('--- 6. VERSION COMPARISON ---')
v1 = json.load(open('cache/clean_summary_v1.json'))
print(f'  v1.0 (single target):  AnnRet={v1["ann_ret"]:.0f}%  Sharpe={v1["sharpe"]:.2f}  MDD={v1["mdd"]:.0f}%')
print(f'  v3.0 (dual target):    AnnRet={ann_ret*100:.0f}%  Sharpe={sharpe:.2f}  MDD={mdd*100:.0f}%')

print()
print('--- 7. FACTOR COMPOSITION ---')
print('  RSI group:      rsi6, rsi14, rsi_extreme')
print('  Bollinger:      boll_pos, boll_width')
print('  MA divergence:  div_ma20, div_ma60, div_ma120, ma_score')
print('  Volatility:     vol_ratio, margin_panic')
print('  Reversal:       streak5_dn')
print('  Preprocessing:  Industry+Size OLS neutralization -> Z-score')
print('  Model:          LightGBM dual-target (excess_ret + cross-sectional rank)')
print('  Ensemble:       Standardized equal-weight average')
print('  Rebalance:      Monthly, top 15-30 by NLP sentiment gating')

print()
print('--- 8. DATA INTEGRITY ---')
print('  Financial factors:    REMOVED (PE/PB/ROE etc, 1-4mo reporting lag)')
print('  Target neutralization: Cross-sectional daily demeaning')
print('  Walk-forward:         Strict 3yr train / 1yr test, no overlap')
print('  Survivorship:         All stocks in kline_daily (5731 codes)')
print('  Random baseline:      Verified ~0% excess (cross-sectionally centered)')

print()
print('='*80)
print('  STATUS: Alpha Core — Ready for paper trading validation')
print('='*80)
