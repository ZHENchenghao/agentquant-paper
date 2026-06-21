# -*- coding: utf-8 -*-
"""信用周期→行业轮动 · 方正/天风方法论
======================================
社融+M1-M2剪刀差 → 四象限 → 行业配置
WF: 每年用前5年数据做周期分类, 测试年按当前象限选行业
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("信用周期 → 行业轮动")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ===== 1. 信用周期数据 =====
macro = con.execute("""
    SELECT trade_date, social_finance, m1_growth, m2_growth, china_10y
    FROM macro_indicators WHERE trade_date >= DATE '2008-01-01'
    ORDER BY trade_date
""").df()
macro['trade_date'] = pd.to_datetime(macro['trade_date'])
macro = macro.set_index('trade_date')

# 月度化
macro_m = macro.resample('ME').agg({
    'social_finance': 'last',
    'm1_growth': 'last',
    'm2_growth': 'last',
    'china_10y': 'last',
})
macro_m['M1M2'] = macro_m['m1_growth'] - macro_m['m2_growth']

# 社融12月滚动求和(代表年度信用投放量)
macro_m['sf_12m'] = macro_m['social_finance'].rolling(12).sum()
# 社融方向: 3月MA的6月变化
macro_m['sf_ma3'] = macro_m['sf_12m'].rolling(3).mean()
macro_m['sf_dir'] = (macro_m['sf_ma3'] - macro_m['sf_ma3'].shift(6)) > 0  # True=扩张

# M1-M2方向
macro_m['m1m2_ma3'] = macro_m['M1M2'].rolling(3).mean()
macro_m['m1m2_dir'] = (macro_m['m1m2_ma3'] - macro_m['m1m2_ma3'].shift(6)) > 0  # True=扩张

# 四象限
def quadrant(row):
    sf_up = row['sf_dir']
    mm_up = row['m1m2_dir']
    if sf_up and mm_up: return '复苏'
    elif sf_up and not mm_up: return '过热'
    elif not sf_up and mm_up: return '滞胀'
    else: return '衰退'

macro_m['regime'] = macro_m.apply(quadrant, axis=1)
macro_m = macro_m.dropna(subset=['regime'])

# ===== 2. 行业指数 =====
ind = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2008-01-01' ORDER BY industry, trade_date
""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])
con.close()

ind['month'] = ind['trade_date'].dt.to_period('M')
ind_m = ind.groupby(['industry', 'month'])['close'].last().reset_index()
ind_m['month'] = ind_m['month'].dt.to_timestamp()
ind_m['ret_1m'] = ind_m.groupby('industry')['close'].pct_change()
ind_m['fwd_ret'] = ind_m.groupby('industry')['ret_1m'].shift(-1)
ind_m = ind_m.dropna(subset=['fwd_ret'])

# Merge with regime
ind_m['regime_month'] = ind_m['month']
# 周期状态滞后1月(月初已知上月社融)
merged = ind_m.copy()
merged['regime'] = None
for i, row in merged.iterrows():
    m = row['month']
    # 找最近的macro数据(上月或更早)
    prev_months = macro_m[macro_m.index < m]
    if len(prev_months) > 0:
        merged.at[i, 'regime'] = prev_months.iloc[-1]['regime']

merged = merged.dropna(subset=['regime'])
print(f"[1] 数据: {len(merged)}行, {merged['month'].nunique()}月, "
      f"{merged['industry'].nunique()}行业")

# 周期分布
print("\n周期分布:")
for r, grp in merged.groupby('regime'):
    print(f"  {r}: {grp['month'].nunique()}月 ({grp['month'].nunique()/merged['month'].nunique()*100:.0f}%)")

# ===== 3. 各周期阶段行业收益特征 =====
print(f"\n[2] 各周期阶段最优行业(WF训练窗)")
TRAIN_YEARS = 5
YEARS = sorted(set(d.year for d in merged['month']))
WF_START = YEARS[0] + TRAIN_YEARS + 1

# 滚动: 每年用前5年数据统计各周期阶段的行业收益
all_industry_returns = []
for test_yr in range(WF_START, YEARS[-1]+1):
    train_s = pd.Timestamp(f'{test_yr - TRAIN_YEARS}-01-01')
    train_e = pd.Timestamp(f'{test_yr - 1}-12-31')
    train = merged[(merged['month'] >= train_s) & (merged['month'] <= train_e)]

    # 统计训练窗内每个周期阶段各行业的月均收益
    regime_ind_ret = train.groupby(['regime', 'industry'])['fwd_ret'].mean().reset_index()
    regime_ind_ret['test_yr'] = test_yr
    all_industry_returns.append(regime_ind_ret)

# 展示最近一年的各周期最优行业
latest = all_industry_returns[-1]
print(f"\n各周期阶段Top行业(训练窗{WF_START+len(all_industry_returns)-1}年):")
for regime in ['复苏', '过热', '滞胀', '衰退']:
    r_data = latest[latest['regime'] == regime].nlargest(5, 'fwd_ret')
    if len(r_data) > 0:
        tops = ' '.join([f"{r['industry']}({r['fwd_ret']*100:+.1f}%)"
                         for _, r in r_data.iterrows()])
        print(f"  {regime}: {tops}")

# ===== 4. WF回测: 周期状态→选行业 =====
print(f"\n[3] WF回测: 信用周期→行业轮动 ({WF_START}-{YEARS[-1]})")

COST = 0.003

# 策略1: 当前周期 → Top5行业
strategy_rets = {'credit_top5': [], 'credit_top3': [], 'momentum_top5': [], 'eq': []}

for test_yr in range(WF_START, YEARS[-1]+1):
    train_s = pd.Timestamp(f'{test_yr - TRAIN_YEARS}-01-01')
    train_e = pd.Timestamp(f'{test_yr - 1}-12-31')
    test_s = pd.Timestamp(f'{test_yr}-01-01')
    test_e = pd.Timestamp(f'{test_yr}-12-31')

    train = merged[(merged['month'] >= train_s) & (merged['month'] <= train_e)]
    test = merged[(merged['month'] >= test_s) & (merged['month'] <= test_e)]
    if len(test) < 30: continue

    # 训练: 各周期->最优行业映射
    regime_tops = {}
    for regime in ['复苏', '过热', '滞胀', '衰退']:
        r_train = train[train['regime'] == regime]
        if len(r_train) > 20:
            avg_rets = r_train.groupby('industry')['fwd_ret'].mean().sort_values(ascending=False)
            regime_tops[regime] = avg_rets.head(10).index.tolist()

    # 动量基准: 上月Top行业
    test_mom = test.copy()
    test_mom['mom_score'] = test_mom.groupby('month')['ret_1m'].rank(pct=True)

    for m, grp in test.groupby('month'):
        regime = grp['regime'].iloc[0]
        if len(grp) < 5: continue

        # 信用周期策略
        if regime in regime_tops:
            top_inds = regime_tops[regime]
            top5 = grp[grp['industry'].isin(top_inds[:5])]
            top3 = grp[grp['industry'].isin(top_inds[:3])]
            if len(top5) > 0:
                strategy_rets['credit_top5'].append(top5['fwd_ret'].mean() - COST)
            if len(top3) > 0:
                strategy_rets['credit_top3'].append(top3['fwd_ret'].mean() - COST)
        else:
            # 未知周期: 等权
            strategy_rets['credit_top5'].append(grp['fwd_ret'].mean() - COST)
            strategy_rets['credit_top3'].append(grp['fwd_ret'].mean() - COST)

        # 动量基准
        mom_grp = test_mom[test_mom['month'] == m]
        if len(mom_grp) > 5:
            top5_mom = mom_grp.nlargest(5, 'mom_score')
            strategy_rets['momentum_top5'].append(top5_mom['fwd_ret'].mean() - COST)

        # 等权
        strategy_rets['eq'].append(grp['fwd_ret'].mean())

# ===== 5. 评估 =====
def evaluate(name, rets):
    arr = np.array(rets); n = len(arr)
    if n < 10: return
    cum = np.prod(1+arr); ann = cum**(12/n)-1
    vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
    hit = np.mean(arr>0)
    print(f"  {name:<20s} 年化{ann*100:+6.1f}%  Sharpe{sh:+5.2f}  MDD{mdd*100:+5.1f}%  月胜率{hit*100:4.0f}%  "
          f"累积{cum-1:+7.1%} ({n}月)")

print()
for name in ['credit_top5', 'credit_top3', 'momentum_top5', 'eq']:
    evaluate(name, strategy_rets[name])

# 分年
print(f"\n分年对比:")
print(f"{'年份':>6s} {'信用Top5':>8s} {'动量Top5':>8s} {'等权':>8s} {'周期阶段':>10s}")
for test_yr in range(WF_START, YEARS[-1]+1):
    test_s = pd.Timestamp(f'{test_yr}-01-01')
    test_e = pd.Timestamp(f'{test_yr}-12-31')
    test = merged[(merged['month'] >= test_s) & (merged['month'] <= test_e)]

    # 本年主要周期
    regime_counts = test['regime'].value_counts()
    main_regime = regime_counts.index[0] if len(regime_counts) > 0 else '?'

    # 本年各策略收益
    yr_data = test.copy()
    yr_rets = {}
    for name, fn in [('credit_top5', None), ('momentum_top5', None), ('eq', None)]:
        pass

    # 简化: 按月算
    yr_cr = []; yr_mom = []; yr_eq = []
    for m, grp in test.groupby('month'):
        regime = grp['regime'].iloc[0]
        if regime in regime_tops:
            top_inds = regime_tops[regime]
            top5 = grp[grp['industry'].isin(top_inds[:5])]
            if len(top5) > 0:
                yr_cr.append(top5['fwd_ret'].mean())
        # 动量
        grp_m = test_mom[test_mom['month'] == m] if 'test_mom' in dir() else grp.copy()
        grp_m['ms'] = grp.groupby('month')['ret_1m'].transform(lambda x: x.rank(pct=True))
        if len(grp_m) > 5:
            top5m = grp_m.nlargest(5, 'ms')
            yr_mom.append(top5m['fwd_ret'].mean())
        yr_eq.append(grp['fwd_ret'].mean())

    if yr_cr and yr_eq:
        cr_yr = np.prod(1+np.array(yr_cr)) - 1 if yr_cr else 0
        mom_yr = np.prod(1+np.array(yr_mom)) - 1 if yr_mom else 0
        eq_yr = np.prod(1+np.array(yr_eq)) - 1
        print(f"  {test_yr:>4d}  {cr_yr*100:+7.1f}% {mom_yr*100:+7.1f}% {eq_yr*100:+7.1f}%  {main_regime}")

# 周期判断准确率
print(f"\n[4] 周期判断效果分析:")
# 每个周期阶段的实际行业收益是否符合预期
for regime in ['复苏', '过热', '滞胀', '衰退']:
    r_data = merged[merged['regime'] == regime]
    if len(r_data) > 50:
        ind_rets = r_data.groupby('industry')['fwd_ret'].mean().sort_values(ascending=False)
        best3 = ind_rets.head(3)
        worst3 = ind_rets.tail(3)
        print(f"  {regime}({r_data['month'].nunique()}月):")
        print(f"    最优: {' '.join([f'{i}({r*100:+.1f}%)' for i,r in best3.items()])}")
        print(f"    最差: {' '.join([f'{i}({r*100:+.1f}%)' for i,r in worst3.items()])}")

print(f"\n耗时: {time.time()-t0:.0f}s")
