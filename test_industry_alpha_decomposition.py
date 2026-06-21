# -*- coding: utf-8 -*-
"""行业回报分解 · 市场β剥离 → 纯行业α预测
=============================================
核心: R_ind = β*R_mkt + α_ind
      用滚动β剥离市场驱动 → 预测残差α → 这才是行业轮动能赚的钱
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats
t0 = time.time()

print("=" * 70)
print("行业回报分解 · β剥离 → 纯α预测")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ===== 1. 数据 =====
ind = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2005-01-01' ORDER BY industry, trade_date
""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])

# 用全A等权代替HS300(更代表市场)
market = con.execute("""
    SELECT trade_date, AVG(close) as mkt_close
    FROM kline_daily WHERE trade_date >= DATE '2005-01-01'
    GROUP BY trade_date ORDER BY trade_date
""").df()
market['trade_date'] = pd.to_datetime(market['trade_date'])
con.close()

# ===== 2. 月度收益 =====
ind['month'] = ind['trade_date'].dt.to_period('M')
ind_m = ind.groupby(['industry', 'month'])['close'].last().reset_index()
ind_m['month'] = ind_m['month'].dt.to_timestamp()
ind_m['ret'] = ind_m.groupby('industry')['close'].pct_change()

market['month'] = market['trade_date'].dt.to_period('M')
mkt_m = market.groupby('month')['mkt_close'].last().reset_index()
mkt_m['month'] = mkt_m['month'].dt.to_timestamp()
mkt_m['mkt_ret'] = mkt_m['mkt_close'].pct_change()

# ===== 3. 滚动36月β估计 → 残差α =====
merged = ind_m.merge(mkt_m[['month', 'mkt_ret']], on='month', how='inner')
merged = merged.dropna(subset=['ret', 'mkt_ret'])

# 对每个行业, 滚动36月窗口估计β, 然后计算残差
ROLL_WINDOW = 36
merged = merged.sort_values(['industry', 'month'])

for ind_name, grp in merged.groupby('industry'):
    idx = grp.index
    rets = grp['ret'].values
    mkt_rets = grp['mkt_ret'].values

    rolling_beta = np.full(len(grp), np.nan)
    rolling_alpha = np.full(len(grp), np.nan)
    residual = np.full(len(grp), np.nan)

    for i in range(ROLL_WINDOW, len(grp)):
        y = rets[i-ROLL_WINDOW:i]
        x = mkt_rets[i-ROLL_WINDOW:i]
        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() > 20:
            slope, intercept, r, p, se = stats.linregress(x[valid], y[valid])
            rolling_beta[i] = slope
            rolling_alpha[i] = intercept
            # 残差 = 实际收益 - (α + β*市场收益)
            if not np.isnan(rets[i]) and not np.isnan(mkt_rets[i]):
                residual[i] = rets[i] - (intercept + slope * mkt_rets[i])

    merged.loc[idx, 'beta'] = rolling_beta
    merged.loc[idx, 'alpha_raw'] = rolling_alpha
    merged.loc[idx, 'residual'] = residual

merged = merged.dropna(subset=['beta', 'residual'])
print(f"[1] 滚动β估计完成: {len(merged)}行, β均值{merged['beta'].mean():.2f}")

# 累积残差(行业α净值曲线)
merged['cum_residual'] = merged.groupby('industry')['residual'].transform(
    lambda x: (1+x).cumprod())

# ===== 4. 构建α因子 =====
# 在β剥离后的空间里重新测试因子
# 因子1: 残差动量 (过去1/3/6月残差)
for w in [1, 3, 6, 12]:
    merged[f'resid_mom_{w}m'] = merged.groupby('industry')['residual'].transform(
        lambda x: x.rolling(w).sum())

# 因子2: 残差波动率
merged['resid_vol_3m'] = merged.groupby('industry')['residual'].transform(
    lambda x: x.rolling(3).std())

# 因子3: β变化 (β在扩张→行业对市场更敏感)
merged['beta_chg'] = merged.groupby('industry')['beta'].transform(
    lambda x: x - x.shift(6))

# 因子4: α动量 (截距项的变化)
merged['alpha_chg'] = merged.groupby('industry')['alpha_raw'].transform(
    lambda x: x.rolling(3).mean() - x.rolling(12).mean())

# ===== 5. 目标: 下月残差 vs 下月总收益 =====
merged['fwd_residual'] = merged.groupby('industry')['residual'].shift(-1)
merged['fwd_ret'] = merged.groupby('industry')['ret'].shift(-1)
merged = merged.dropna(subset=['fwd_residual', 'fwd_ret'])

# 同时准备原始动量做基准
merged['mom_1m'] = merged.groupby('industry')['ret'].transform(lambda x: x.shift(1))

print(f"[2] 因子: {len(merged)}行")

# ===== 6. WF IC: 残差因子 vs 原始因子 =====
YEARS = sorted(set(d.year for d in merged['month']))
TRAIN = 5; WF_START = YEARS[0] + TRAIN + 1

FACTOR_TESTS = {
    # 残差空间因子
    'resid_mom_1m': ('残差动量1月', 'fwd_residual', 1),
    'resid_mom_3m': ('残差动量3月', 'fwd_residual', 1),
    'resid_mom_12m': ('残差动量12月', 'fwd_residual', 1),
    'resid_vol_3m': ('残差波动率', 'fwd_residual', -1),
    'beta_chg': ('β变化', 'fwd_residual', -1),
    'alpha_chg': ('α动量', 'fwd_residual', 1),
    # 原始空间因子(基准)
    'mom_1m': ('原始动量1月', 'fwd_ret', 1),
}

print(f"\n[3] WF IC ({WF_START}-{YEARS[-1]})")
print(f"{'因子上限':<20s} {'预测目标':>12s} {'IC':>8s} {'IR':>7s} {'t':>7s} {'方向'}")
print("-" * 65)

for f, (name, target, expected) in FACTOR_TESTS.items():
    if f not in merged.columns: continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[f, target])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid[target].rank())
                if not np.isnan(ic): ics.append(ic)
    if len(ics) > 10:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std>0 else 0
        ir = mi/std*np.sqrt(12) if std>0 else 0
        ok = (mi>0 and expected>0) or (mi<0 and expected<0)
        print(f"{name:<20s} {target:<12s} {mi*100:+7.2f}% {ir:+6.2f} {t:+6.2f} {'OK' if ok else 'XX'}")

# ===== 7. 策略WF: 预测残差 vs 预测总收益 =====
print(f"\n[4] 策略WF: 预测α vs 预测总收益")

def wf_strategy(df, factor, target, label):
    long_r = []
    for yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01'); train_e = pd.Timestamp(f'{yr-1}-12-31')
        train = df[(df['month']>=train_s)&(df['month']<=train_e)]
        test = df[(df['month']>=ts)&(df['month']<=te)]
        if len(test)<30 or len(train)<60: continue

        # 定方向
        dir_ics = []
        for m, grp in train.groupby('month'):
            v = grp.dropna(subset=[factor, target])
            if len(v)>5:
                ic = v[factor].rank().corr(v[target].rank())
                if not np.isnan(ic): dir_ics.append(ic)
        direction = 1 if (len(dir_ics)>8 and np.mean(dir_ics)>0) else -1

        for m, grp in test.groupby('month'):
            grp = grp.dropna(subset=[factor, target])
            if len(grp)<5: continue
            grp = grp.copy()
            grp['score'] = grp[factor].rank(pct=True) * direction
            top5 = grp.nlargest(5, 'score')
            # 关键是: 选出的行业实际总收益是多少?
            long_r.append(top5['fwd_ret'].mean() - 0.003)

    if long_r:
        arr = np.array(long_r); n = len(arr)
        cum = np.prod(1+arr); ann = cum**(12/n)-1
        c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
        vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
        print(f"  {label:<24s} 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.0f}% ({n}月)")

wf_strategy(merged, 'mom_1m', 'fwd_ret', '原始动量→总收益')
wf_strategy(merged, 'resid_mom_1m', 'fwd_residual', '残差动量→残差(但换总收益)')
wf_strategy(merged, 'resid_mom_3m', 'fwd_residual', '残差动量3M→残差')
wf_strategy(merged, 'alpha_chg', 'fwd_residual', 'α动量→残差')

# ===== 8. 双因子: 市场择时+行业选择 =====
print(f"\n[5] 结构化两层: 先判市场方向 → 再选行业")
# 用市场动量判断大方向(涨/跌), 涨时用动量选行业, 跌时用低波选
merged['mkt_mom'] = merged.groupby('month')['mkt_ret'].transform(lambda x: x)
two_layer_rets = []

for yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
    test = merged[(merged['month']>=ts)&(merged['month']<=te)]
    for m, grp in test.groupby('month'):
        if len(grp)<10: continue
        grp = grp.dropna(subset=['ret', 'residual', 'resid_mom_1m', 'resid_vol_3m'])
        if len(grp)<5: continue
        grp = grp.copy()

        # 判断市场方向: 用前6月市场动量
        mkt_mom = grp['mkt_ret'].iloc[0]  # 当月市场收益(我们不知道下月, 用近6月)
        # 简单: 用滚动12月市场动量
        prev = merged[(merged['month']<m)&(merged['month']>=m-pd.DateOffset(months=6))]
        mkt_trend = prev['mkt_ret'].mean() if len(prev)>0 else 0

        if mkt_trend > 0:
            # 涨市: 选高动量
            grp['score'] = grp['resid_mom_1m'].rank(pct=True)
        else:
            # 跌市: 选低波动
            grp['score'] = grp['resid_vol_3m'].rank(pct=True, ascending=False)

        top5 = grp.nlargest(5, 'score')
        two_layer_rets.append(top5['fwd_ret'].mean() - 0.003)

arr = np.array(two_layer_rets); n = len(arr)
cum = np.prod(1+arr); ann = cum**(12/n)-1
c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
print(f"  两层(涨市动量+跌市低波) 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.0f}% ({n}月)")

# 残差因子的实际经济含义
print(f"\n[6] 残差α的统计特征:")
print(f"  残差月均: {merged['residual'].mean()*100:+.2f}%")
print(f"  残差标准差: {merged['residual'].std()*100:.2f}%")
print(f"  残差自相关(1月): {merged.groupby('industry')['residual'].apply(lambda x: x.autocorr(1)).mean():.3f}")
print(f"  残差占总收益比: {merged['residual'].var()/merged['ret'].var()*100:.0f}%")

# 残差动量IC
print(f"\n  残差动量1月IC(分年):")
for yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{yr}-01-01'); te = pd.Timestamp(f'{yr}-12-31')
    test = merged[(merged['month']>=ts)&(merged['month']<=te)]
    yr_ics = []
    for m, grp in test.groupby('month'):
        valid = grp.dropna(subset=['resid_mom_1m', 'fwd_residual'])
        if len(valid)>5:
            ic = valid['resid_mom_1m'].rank().corr(valid['fwd_residual'].rank())
            if not np.isnan(ic): yr_ics.append(ic)
    if yr_ics:
        print(f"  {yr}: IC={np.mean(yr_ics)*100:+5.2f}%  t={np.mean(yr_ics)/np.std(yr_ics)*np.sqrt(len(yr_ics)):+5.2f}")

print(f"\n耗时: {time.time()-t0:.0f}s")
