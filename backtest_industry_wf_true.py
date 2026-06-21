# -*- coding: utf-8 -*-
"""行业轮动 · 真正Walk-Forward · 无前瞻偏差
==============================================
关键修正:
  每轮只用历史5年数据→选因子+定方向+定权重
  → 应用到下一年的月度调仓
  → 逐年前滚, 完全模拟真实交易环境
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("行业轮动 · 真实WF · 无前瞻偏差")
print("=" * 70)

# ============ 1. 数据 (同上) ============
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

raw = con.execute("""
    SELECT industry, trade_date, close
    FROM proxy_industry_daily WHERE trade_date >= DATE '2005-01-01'
    ORDER BY industry, trade_date
""").df()
raw['trade_date'] = pd.to_datetime(raw['trade_date'])

hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300 = hs300.set_index('trade_date')['close']
con.close()

# ============ 2. 日频特征 ============
raw = raw.sort_values(['industry', 'trade_date']).reset_index(drop=True)

for ind_name, grp in raw.groupby('industry'):
    idx = grp.index; close = grp['close'].values
    ret = np.concatenate([[np.nan], (close[1:]-close[:-1])/close[:-1]])
    def roll_ret(k):
        r = np.full(len(close), np.nan)
        for i in range(k, len(close)): r[i] = close[i]/close[i-k] - 1
        return r

    raw.loc[idx, 'ret_5d'] = roll_ret(5)
    raw.loc[idx, 'ret_20d'] = roll_ret(20)
    raw.loc[idx, 'ret_60d'] = roll_ret(60)
    raw.loc[idx, 'ret_120d'] = roll_ret(120)
    raw.loc[idx, 'ret_240d'] = roll_ret(240)

    for k in [20, 60]:
        raw.loc[idx, f'vol_{k}d'] = pd.Series(ret).rolling(k).std().values * np.sqrt(252)

    raw.loc[idx, 'down_vol_60d'] = pd.Series(np.where(ret<0, ret, 0)).rolling(60).std().values*np.sqrt(252)
    raw.loc[idx, 'max_ret_20d'] = pd.Series(ret).rolling(20).max().values

    cum = np.cumprod(1+np.nan_to_num(ret,0)); h_max = np.maximum.accumulate(cum)
    raw.loc[idx, 'mdd_240d'] = pd.Series(cum/h_max-1).rolling(240).min().values

    h_52w = pd.Series(close).rolling(240).max().values
    l_52w = pd.Series(close).rolling(240).min().values
    raw.loc[idx, 'pct_52w_high'] = close/h_52w - 1
    raw.loc[idx, 'pct_52w_low'] = close/l_52w - 1

# ============ 3. 月度因子 ============
raw['month'] = raw['trade_date'].dt.to_period('M')
monthly = raw.groupby(['industry', 'month']).agg(
    close=('close', 'last'),
    mom_1m=('ret_20d', 'last'),
    mom_3m=('ret_60d', 'last'),
    mom_6m=('ret_120d', 'last'),
    mom_12m=('ret_240d', 'last'),
    rev_5d=('ret_5d', 'last'),
    rev_10d=('ret_5d', lambda x: x.tail(10).mean()),
    vol_1m=('vol_20d', 'last'),
    vol_3m=('vol_60d', 'last'),
    down_vol=('down_vol_60d', 'last'),
    max_ret=('max_ret_20d', 'last'),
    mdd_1y=('mdd_240d', 'last'),
    pct_high=('pct_52w_high', 'last'),
    pct_low=('pct_52w_low', 'last'),
    n_days=('trade_date', 'nunique'),
).reset_index()

monthly['mom_12m_1m'] = monthly['mom_1m'] - monthly['mom_12m']
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly['fwd_ret'] = monthly.groupby('industry')['mom_1m'].shift(-1)
monthly = monthly[monthly['n_days'] >= 15].dropna(subset=['fwd_ret'])

# 所有候选因子
ALL_FACTORS = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12m_1m',
               'rev_5d', 'rev_10d', 'vol_1m', 'vol_3m', 'down_vol',
               'max_ret', 'mdd_1y', 'pct_high', 'pct_low']

print(f"[1] 数据: {len(monthly)}行, {monthly['month'].nunique()}月, {monthly['industry'].nunique()}行业")

# ============ 4. 真正Walk-Forward ============
TRAIN_YEARS = 5
YEARS = sorted(set(d.year for d in monthly['month']))
WF_START = YEARS[0] + TRAIN_YEARS + 1  # 第6年开始测试

print(f"\n[2] 真正WF: 训练{TRAIN_YEARS}年 → 测试1年, {WF_START}-{YEARS[-1]}")
COST = 0.003  # 30bp/月

results = {
    'long': [],     # 做多组合月度收益
    'short': [],    # 做空组合月度收益
    'eq': [],       # 等权收益
    'years': [],    # 对应年份
    'factors': [],  # 每年入选的因子
    'ics': [],      # 每年每个因子IC
}

for test_yr in range(WF_START, YEARS[-1]+1):
    # 训练窗口: 前TRAIN_YEARS年
    train_start = pd.Timestamp(f'{test_yr - TRAIN_YEARS}-01-01')
    train_end = pd.Timestamp(f'{test_yr - 1}-12-31')  # 到去年底
    train = monthly[(monthly['month'] >= train_start) & (monthly['month'] <= train_end)]

    # 测试窗口: 本年
    test_start = pd.Timestamp(f'{test_yr}-01-01')
    test_end = pd.Timestamp(f'{test_yr}-12-31')
    test = monthly[(monthly['month'] >= test_start) & (monthly['month'] <= test_end)]

    if len(test) < 30:
        continue

    # === 步骤1: 在训练窗选因子(IC方向+t检验) ===
    factor_ics = {}
    for f in ALL_FACTORS:
        if f not in train.columns:
            continue
        ics = []
        for m, grp in train.groupby('month'):
            valid = grp.dropna(subset=[f, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic):
                    ics.append(ic)
        if len(ics) > 10:
            mi = np.mean(ics); std = np.std(ics)
            t = mi / std * np.sqrt(len(ics)) if std > 0 else 0
            factor_ics[f] = {'ic': mi, 't': t, 'n': len(ics)}

    # 选|t|>1.5的因子, 用实际IC方向
    selected = [(f, v['ic'], 1 if v['ic'] > 0 else -1)
                for f, v in sorted(factor_ics.items(), key=lambda x: abs(x[1]['t']), reverse=True)
                if abs(v['t']) > 1.5]
    selected = selected[:6]  # 最多6个

    if len(selected) < 2:
        # 至少选2个
        selected = [(f, v['ic'], 1 if v['ic'] > 0 else -1)
                    for f, v in sorted(factor_ics.items(), key=lambda x: abs(x[1]['t']), reverse=True)][:3]

    results['factors'].append((test_yr, [f for f, ic, d in selected]))
    results['ics'].append((test_yr, {f: factor_ics[f] for f in [s[0] for s in selected]}))

    # === 步骤2: 在测试窗按月调仓 ===
    test_copy = test.copy()
    # 计算综合得分
    for f, ic, direction in selected:
        if f in test_copy.columns:
            test_copy[f'{f}_rank'] = test_copy.groupby('month')[f].rank(pct=True) * direction

    rank_cols = [f'{f}_rank' for f, ic, d in selected if f'{f}_rank' in test_copy.columns]
    if not rank_cols:
        continue
    test_copy['combo'] = test_copy[rank_cols].mean(axis=1)

    for m, grp in test_copy.groupby('month'):
        if len(grp) < 10:
            continue
        n = max(1, len(grp)//4)
        top = grp.nlargest(n, 'combo')
        bot = grp.nsmallest(n, 'combo')

        results['long'].append(top['fwd_ret'].mean() - COST)
        results['short'].append(bot['fwd_ret'].mean() - COST)
        results['eq'].append(grp['fwd_ret'].mean())
        results['years'].append(test_yr)

# ============ 5. 评估 ============
long_arr = np.array(results['long'])
short_arr = np.array(results['short'])
eq_arr = np.array(results['eq'])
years_arr = np.array(results['years'])

if len(long_arr) < 10:
    print("数据不足"); exit()

def stats(rets, name):
    cum = np.prod(1+rets); n = len(rets)
    ann = cum ** (12/n) - 1
    vol = np.std(rets)*np.sqrt(12)
    sharpe = ann/vol if vol>0 else 0
    # MDD
    cum_series = np.cumprod(1+rets)
    peak = np.maximum.accumulate(cum_series)
    mdd = np.min(cum_series/peak-1)
    win = np.mean(rets>0)
    print(f"  {name:<12s} 年化{ann*100:+6.1f}%  累积{cum-1:+7.1%}  Sharpe{sharpe:5.2f}  MDD{mdd*100:+6.1f}%  胜率{win*100:4.0f}%")
    return cum, ann, sharpe, mdd

print(f"\n{'='*70}")
print(f"真实WF结果 ({WF_START}-{YEARS[-1]}, {len(long_arr)}个月)")
print(f"{'='*70}")

cum_l, ann_l, sh_l, mdd_l = stats(long_arr, '做多(Top25%)')
cum_s, ann_s, sh_s, mdd_s = stats(short_arr, '做空(Bot25%)')

ls_arr = long_arr - short_arr
cum_ls, ann_ls, sh_ls, mdd_ls = stats(ls_arr, '多空')
print(f"  多空命中率: {np.mean(ls_arr>0)*100:.0f}%")

cum_eq, ann_eq, sh_eq, mdd_eq = stats(eq_arr, '等权基准')

# 分年
print(f"\n{'年份':>6s} {'做多年':>8s} {'做空年':>8s} {'多空年':>8s} {'等权年':>8s} {'入选因子'}")
print("-" * 90)
for yr in range(WF_START, YEARS[-1]+1):
    mask = years_arr == yr
    if mask.sum() < 3:
        continue
    yr_long = np.prod(1+long_arr[mask]) - 1
    yr_short = np.prod(1+short_arr[mask]) - 1
    yr_ls = np.prod(1+ls_arr[mask]) - 1
    yr_eq = np.prod(1+eq_arr[mask]) - 1
    yr_factors = [r[1] for r in results['factors'] if r[0]==yr]
    factor_str = ','.join(yr_factors[0][:4]) if yr_factors else '-'
    print(f"  {yr:>4d}  {yr_long*100:+7.1f}% {yr_short*100:+7.1f}% {yr_ls*100:+7.1f}% {yr_eq*100:+7.1f}%  {factor_str}")

# 因子出场频率
from collections import Counter
factor_freq = Counter()
for yr, factors in results['factors']:
    for f in factors:
        factor_freq[f] += 1
print(f"\n因子出场频率({len(results['factors'])}年):")
for f, cnt in factor_freq.most_common():
    print(f"  {f:<16s}: {cnt}年 ({cnt/len(results['factors'])*100:.0f}%)")

# vs HS300
hs300_m = hs300.resample('ME').last()
print(f"\nvs 沪深300:")
for yr in range(WF_START, YEARS[-1]+1):
    mask = years_arr == yr
    if mask.sum() < 3:
        continue
    yr_long_cum = np.prod(1+long_arr[mask]) - 1
    hs300_yr = hs300_m[hs300_m.index.year==yr]
    if len(hs300_yr)>1:
        hs_ret = hs300_yr.iloc[-1]/hs300_yr.iloc[0]-1
        diff = yr_long_cum - hs_ret
        print(f"  {yr}: 策略{yr_long_cum*100:+5.1f}%  HS300{hs_ret*100:+5.1f}%  超额{diff*100:+5.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
