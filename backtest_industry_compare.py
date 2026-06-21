# -*- coding: utf-8 -*-
"""行业轮动 · 四策略对比(全WF, 无前瞻)"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
raw = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2005-01-01' ORDER BY industry, trade_date
""").df()
raw['trade_date'] = pd.to_datetime(raw['trade_date'])
hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300 = hs300.set_index('trade_date')['close']
con.close()

# 日频特征
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
    raw.loc[idx, 'down_vol_60d'] = pd.Series(np.where(ret<0, ret, 0)).rolling(60).std().values * np.sqrt(252)
    raw.loc[idx, 'max_ret_20d'] = pd.Series(ret).rolling(20).max().values
    cum = np.cumprod(1+np.nan_to_num(ret, 0)); h_max = np.maximum.accumulate(cum)
    raw.loc[idx, 'mdd_240d'] = pd.Series(cum/h_max-1).rolling(240).min().values
    h = pd.Series(close).rolling(240).max().values; l = pd.Series(close).rolling(240).min().values
    raw.loc[idx, 'pct_52w_high'] = close/h - 1; raw.loc[idx, 'pct_52w_low'] = close/l - 1

# 月度
raw['month'] = raw['trade_date'].dt.to_period('M')
monthly = raw.groupby(['industry', 'month']).agg(
    close=('close', 'last'),
    mom_1m=('ret_20d', 'last'), mom_3m=('ret_60d', 'last'),
    mom_6m=('ret_120d', 'last'), mom_12m=('ret_240d', 'last'),
    rev_5d=('ret_5d', 'last'), rev_10d=('ret_5d', lambda x: x.tail(10).mean()),
    vol_1m=('vol_20d', 'last'), vol_3m=('vol_60d', 'last'),
    down_vol=('down_vol_60d', 'last'), max_ret=('max_ret_20d', 'last'),
    mdd_1y=('mdd_240d', 'last'), pct_high=('pct_52w_high', 'last'),
    pct_low=('pct_52w_low', 'last'), n_days=('trade_date', 'nunique'),
).reset_index()
monthly['mom_12m_1m'] = monthly['mom_1m'] - monthly['mom_12m']
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly['fwd_ret'] = monthly.groupby('industry')['mom_1m'].shift(-1)
monthly = monthly[monthly['n_days'] >= 15].dropna(subset=['fwd_ret'])

COST = 0.003; TRAIN_YEARS = 5
YEARS = sorted(set(d.year for d in monthly['month'])); WF_START = YEARS[0] + TRAIN_YEARS + 1

ALL_F = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12m_1m',
         'rev_5d', 'rev_10d', 'vol_1m', 'vol_3m', 'down_vol',
         'max_ret', 'mdd_1y', 'pct_high', 'pct_low']

strategies = {
    '纯动量(1m)': ['mom_1m'],
    '动量(1m+3m)': ['mom_1m', 'mom_3m'],
    '动量(1m+3m+6m+12m)': ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m'],
    '动量+反转': ['mom_1m', 'mom_3m', 'rev_5d', 'rev_10d'],
    '全因子等权(14)': ALL_F,
}

all_results = {}
for sname, factors in strategies.items():
    long_rets = []; short_rets = []; ls_rets = []; eq_rets = []

    for test_yr in range(WF_START, YEARS[-1]+1):
        train_start = pd.Timestamp(f'{test_yr - TRAIN_YEARS}-01-01')
        train_end = pd.Timestamp(f'{test_yr - 1}-12-31')
        train = monthly[(monthly['month'] >= train_start) & (monthly['month'] <= train_end)]
        test = monthly[(monthly['month'] >= pd.Timestamp(f'{test_yr}-01-01')) &
                       (monthly['month'] <= pd.Timestamp(f'{test_yr}-12-31'))]
        if len(test) < 30: continue

        # 训练窗定方向
        factor_dirs = {}
        for f in factors:
            ics = []
            for m, grp in train.groupby('month'):
                valid = grp.dropna(subset=[f, 'fwd_ret'])
                if len(valid) > 5:
                    ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                    if not np.isnan(ic): ics.append(ic)
            factor_dirs[f] = 1 if (len(ics) > 10 and np.mean(ics) > 0) else -1

        # 计算得分
        test_copy = test.copy()
        for f in factors:
            if f in test_copy.columns:
                test_copy[f'{f}_r'] = test_copy.groupby('month')[f].rank(pct=True) * factor_dirs[f]
        rank_cols = [f'{f}_r' for f in factors if f'{f}_r' in test_copy.columns]
        if not rank_cols: continue
        test_copy['score'] = test_copy[rank_cols].mean(axis=1)

        for m, grp in test_copy.groupby('month'):
            if len(grp) < 10: continue
            n = max(1, len(grp) // 4)
            top = grp.nlargest(n, 'score')
            bot = grp.nsmallest(n, 'score')
            long_rets.append(top['fwd_ret'].mean() - COST)
            short_rets.append(bot['fwd_ret'].mean() - COST)
            ls_rets.append(top['fwd_ret'].mean() - bot['fwd_ret'].mean())
            eq_rets.append(grp['fwd_ret'].mean())

    long_arr = np.array(long_rets); ls_arr = np.array(ls_rets)
    n = len(long_arr)
    long_cum = np.prod(1+long_arr); ls_cum = np.prod(1+ls_arr)
    long_ann = long_cum ** (12/n) - 1; ls_ann = ls_cum ** (12/n) - 1

    def mdd_func(r):
        c = np.cumprod(1+r)
        return np.min(c / np.maximum.accumulate(c) - 1)

    ls_vol = np.std(ls_arr) * np.sqrt(12)
    ls_sh = ls_ann / ls_vol if ls_vol > 0 else 0

    all_results[sname] = {
        'long_ann': long_ann, 'long_cum': long_cum-1, 'long_mdd': mdd_func(long_arr),
        'ls_ann': ls_ann, 'ls_sh': ls_sh, 'ls_hit': np.mean(ls_arr > 0),
        'eq_ann': (np.prod(1+np.array(eq_rets)))**(12/len(eq_rets))-1,
        'n': n
    }

print("=" * 90)
print(f"{'策略':<24s} {'做多年化':>8s} {'做多累积':>8s} {'做多MDD':>7s} {'多空年化':>8s} {'多空Sharpe':>7s} {'多空命中':>7s} {'vs等权':>7s}")
print("-" * 90)
for sname, r in sorted(all_results.items(), key=lambda x: x[1]['ls_ann'], reverse=True):
    vs_eq = r['long_ann'] - r['eq_ann']
    print(f"{sname:<24s} {r['long_ann']*100:+7.1f}% {r['long_cum']*100:+7.1f}% {r['long_mdd']*100:+6.1f}% "
          f"{r['ls_ann']*100:+7.1f}% {r['ls_sh']:+6.2f} {r['ls_hit']*100:+6.0f}% {vs_eq*100:+6.1f}%")

# 对比小众战法基准
print(f"\n{'='*90}")
print(f"{'基准对比':<24s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>7s} {'备注'}")
print("-" * 90)
print(f"{'小众战法Top30':<24s} {'+14.8%':>8s} {'0.70':>7s} {'-31.9%':>7s} {'个股行为因子交互(2002-2026 WF)'}")
print(f"{'行业轮动(最优)':<24s} {all_results['动量+反转']['long_ann']*100:+7.1f}% "
      f"{'N/A':>7s} {all_results['动量+反转']['long_mdd']*100:+6.1f}% {'行业动量因子(2011-2026 WF)'}")

print(f"\n耗时: {time.time()-t0:.0f}s")
