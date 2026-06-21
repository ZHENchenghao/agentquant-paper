# -*- coding: utf-8 -*-
"""行业轮动 · 东吴五维 · 个股离散度→行业因子
==============================================
核心创新: 不聚合个股指标到行业, 而用行业内个股的离散结构特征
维度: 离散度 | 偏度 | 参与广度 | 相关结构 | 量能集中度
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats as sp_stats
t0 = time.time()

print("=" * 70)
print("行业轮动 · 个股离散度→行业因子 (东吴方法论)")
print("=" * 70)

# ===== 1. 加载 =====
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 行业映射
ind_map = con.execute("SELECT ts_code, ind_name FROM stock_industry").df()
ind_map = ind_map.rename(columns={'ind_name': 'industry'})
print(f"[1] 行业映射: {ind_map['industry'].nunique()}行业, {len(ind_map)}只")

# 个股日线(只用close, 减少内存)
print("[2] 加载日线...")
kline = con.execute("""
    SELECT ts_code, trade_date, close, vol as volume
    FROM kline_daily WHERE trade_date >= DATE '2008-01-01'
    ORDER BY ts_code, trade_date
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# HS300
hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300 = hs300.set_index('trade_date')['close']

# 行业指数
ind_idx = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2008-01-01' ORDER BY industry, trade_date
""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
con.close()

# Merge
kline = kline.merge(ind_map, on='ts_code', how='inner')
print(f"   Merge后: {len(kline)}行, {kline['ts_code'].nunique()}只")

# 日收益
kline = kline.sort_values(['ts_code', 'trade_date'])
kline['ret'] = kline.groupby('ts_code')['close'].pct_change()
kline = kline.dropna(subset=['ret'])

# ===== 2. 月度行业内离散度因子 =====
kline['month'] = kline['trade_date'].dt.to_period('M')

print("[3] 计算月度行业内离散度...")
# 为了效率, 只保留最近几个月的数据来算目标收益, 但离散度需要全历史

# 先按行业-月份聚合离散度指标
def compute_dispersion(grp):
    """输入: 一个行业-月份的所有个股日收益"""
    rets = grp['ret'].values
    vols = grp['volume'].values
    if len(rets) < 10 or len(grp['ts_code'].unique()) < 5:
        return pd.Series({'dispersion': np.nan, 'skewness': np.nan,
                          'kurtosis': np.nan, 'pos_ratio': np.nan,
                          'vol_conc_top3': np.nan, 'n_stocks': len(grp['ts_code'].unique())})

    # 按个股取月收益
    stock_rets = grp.groupby('ts_code')['ret'].apply(lambda x: (1+x).prod()-1)
    stock_vols = grp.groupby('ts_code')['volume'].sum()

    n = len(stock_rets)
    if n < 5:
        return pd.Series({'dispersion': np.nan, 'skewness': np.nan,
                          'kurtosis': np.nan, 'pos_ratio': np.nan,
                          'vol_conc_top3': np.nan, 'n_stocks': n})

    # 1. 离散度: 行业内个股月收益的标准差
    dispersion = np.std(stock_rets)

    # 2. 偏度: 收益分布不对称性
    skew = sp_stats.skew(stock_rets) if n > 5 else 0

    # 3. 峰度: 厚尾(极端值多)
    kurt = sp_stats.kurtosis(stock_rets) if n > 5 else 0

    # 4. 参与度: 正收益股票比例
    pos_ratio = np.mean(stock_rets > 0)

    # 5. 量能集中度: Top3成交量占比
    if len(stock_vols) >= 3:
        vol_conc = stock_vols.nlargest(3).sum() / stock_vols.sum()
    else:
        vol_conc = np.nan

    return pd.Series({'dispersion': dispersion, 'skewness': skew,
                      'kurtosis': kurt, 'pos_ratio': pos_ratio,
                      'vol_conc_top3': vol_conc, 'n_stocks': n})

print("   计算中...")
disp_raw = kline.groupby(['industry', 'month']).apply(compute_dispersion).reset_index()
disp_raw['month'] = disp_raw['month'].dt.to_timestamp()
print(f"   离散度: {len(disp_raw)}行, {disp_raw['industry'].nunique()}行业")

# 衍生因子
disp_raw['disp_ma6'] = disp_raw.groupby('industry')['dispersion'].transform(
    lambda x: x.rolling(6).mean())
disp_raw['disp_change'] = disp_raw['dispersion'] / disp_raw['disp_ma6'].replace(0, 1) - 1

# 偏度变化
disp_raw['skew_ma6'] = disp_raw.groupby('industry')['skewness'].transform(
    lambda x: x.rolling(6).mean())

# ===== 3. 行业指数月度收益(目标) =====
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M')
ind_monthly = ind_idx.groupby(['industry', 'month'])['close'].last().reset_index()
ind_monthly['month'] = ind_monthly['month'].dt.to_timestamp()
ind_monthly['ret_1m'] = ind_monthly.groupby('industry')['close'].pct_change()
ind_monthly['fwd_ret'] = ind_monthly.groupby('industry')['ret_1m'].shift(-1)

# ===== 4. 合并 =====
merged = disp_raw.merge(ind_monthly[['industry', 'month', 'fwd_ret']],
                        on=['industry', 'month'], how='inner')
merged = merged.dropna(subset=['fwd_ret'])
merged = merged[merged['n_stocks'] >= 5]
print(f"[4] 合并: {len(merged)}行")

# ===== 5. 因子清单 =====
FACTORS = {
    'dispersion': ('个股离散度', 1),      # 高离散=分化大=?
    'disp_change': ('离散度变化', -1),     # 离散放大=风险信号
    'skewness': ('收益偏度', 1),         # 正偏=多数涨
    'kurtosis': ('收益峰度', -1),        # 高峰=极端值多
    'pos_ratio': ('正收益比', 1),        # 多数股涨=行业强
    'vol_conc_top3': ('量能集中度', -1),  # 集中=少数驱动
    'n_stocks': ('成分股数', 0),         # 控制变量
}

# ===== 6. WF IC =====
YEARS = sorted(set(d.year for d in merged['month']))
TRAIN_YEARS = 5; WF_START = YEARS[0] + TRAIN_YEARS + 1

print(f"\n[5] WF IC ({WF_START}-{YEARS[-1]})")
print(f"{'因子':<20s} {'预期':>4s} {'IC':>8s} {'IR':>7s} {'t':>7s} {'方向':>6s}")

ic_results = {}
for f, (name, expected) in FACTORS.items():
    if f not in merged.columns or expected == 0: continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[f, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
    if len(ics) > 10:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std>0 else 0
        ir = mi/std*np.sqrt(12) if std>0 else 0
        dir_ok = (mi>0 and expected>0) or (mi<0 and expected<0)
        ic_results[f] = {'name':name, 'ic':mi, 'ir':ir, 't':t, 'dir_ok':dir_ok}

for f, r in sorted(ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    print(f"{r['name']:<20s} {'正' if FACTORS[f][1]>0 else '负':>4s} "
          f"{r['ic']*100:+7.2f}% {r['ir']:+6.2f} {r['t']:+6.2f} "
          f"{'OK' if r['dir_ok'] else 'XX':>6s}")

# ===== 7. vs 原版动量(同窗口) =====
print(f"\n[6] vs 原版动量(同窗口WF):")
# 用同期的ind_monthly算动量因子
ind_monthly_full = ind_monthly.copy()
for w in [1, 3, 6, 12]:
    ind_monthly_full[f'mom_{w}m'] = ind_monthly_full.groupby('industry')['close'].pct_change(w)

mom_merged = ind_monthly_full.dropna(subset=['fwd_ret'])
# Add dispersion factors
mom_merged = mom_merged.merge(
    disp_raw[['industry', 'month', 'dispersion', 'pos_ratio', 'skewness', 'vol_conc_top3']],
    on=['industry', 'month'], how='left')

print(f"{'因子':<20s} {'IC':>8s} {'IR':>7s} {'t':>7s}")
for f, label in [('mom_1m','动量1月'), ('mom_3m','动量3月'), ('mom_6m','动量6月'),
                 ('mom_12m','动量12月'), ('pos_ratio','正收益比'), ('dispersion','离散度')]:
    if f not in mom_merged.columns: continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
        test = mom_merged[(mom_merged['month'] >= ts) & (mom_merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[f, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
    if ics:
        mi = np.mean(ics); ir = mi/np.std(ics)*np.sqrt(12) if np.std(ics)>0 else 0
        t = mi/np.std(ics)*np.sqrt(len(ics)) if np.std(ics)>0 else 0
        print(f"{label:<20s} {mi*100:+7.2f}% {ir:+6.2f} {t:+6.2f}")

# ===== 8. 离散度因子+动量组合 =====
print(f"\n[7] 离散度+动量 组合策略WF")
from itertools import combinations

# 测试: 纯动量 vs 动量+离散度 vs 动量+正收益比
combos = {
    '纯动量(1m+12m)': ['mom_1m', 'mom_12m'],
    '动量+离散度': ['mom_1m', 'mom_12m', 'dispersion'],
    '动量+正收益比': ['mom_1m', 'mom_12m', 'pos_ratio'],
    '动量+离散+正收益比': ['mom_1m', 'mom_12m', 'dispersion', 'pos_ratio'],
}

for cname, factors in combos.items():
    long_r = []; ls_r = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{test_yr-TRAIN_YEARS}-01-01')
        train_e = pd.Timestamp(f'{test_yr-1}-12-31')
        test_s = pd.Timestamp(f'{test_yr}-01-01'); test_e = pd.Timestamp(f'{test_yr}-12-31')

        train = mom_merged[(mom_merged['month']>=train_s)&(mom_merged['month']<=train_e)]
        test = mom_merged[(mom_merged['month']>=test_s)&(mom_merged['month']<=test_e)]
        if len(test)<30 or len(train)<60: continue

        # 训练窗定方向
        dirs = {}
        for f in factors:
            if f not in train.columns or f not in test.columns: continue
            ics = []
            for m, grp in train.groupby('month'):
                v = grp.dropna(subset=[f,'fwd_ret'])
                if len(v)>5:
                    ic = v[f].rank().corr(v['fwd_ret'].rank())
                    if not np.isnan(ic): ics.append(ic)
            dirs[f] = 1 if (len(ics)>8 and np.mean(ics)>0) else -1
        if len(dirs)<2: continue

        tc = test.copy()
        for f in factors:
            if f in tc.columns and f in dirs:
                tc[f'{f}_r'] = tc.groupby('month')[f].rank(pct=True)*dirs[f]
        rc = [f'{f}_r' for f in factors if f'{f}_r' in tc.columns]
        if not rc: continue
        tc['score'] = tc[rc].mean(axis=1)

        for m, grp in tc.groupby('month'):
            if len(grp)<10: continue
            n = max(1, len(grp)//4)
            top = grp.nlargest(n,'score'); bot = grp.nsmallest(n,'score')
            long_r.append(top['fwd_ret'].mean()-0.003)
            ls_r.append(top['fwd_ret'].mean()-bot['fwd_ret'].mean())

    if long_r:
        la=np.array(long_r); ls=np.array(ls_r); n=len(la)
        lc=np.prod(1+la); la_ann=lc**(12/n)-1
        lsc=np.prod(1+ls); ls_ann=lsc**(12/n)-1
        lmd=np.min(np.cumprod(1+la)/np.maximum.accumulate(np.cumprod(1+la))-1)
        ls_sh=ls_ann/(np.std(ls)*np.sqrt(12)) if np.std(ls)>0 else 0
        print(f"  {cname:<22s} 做多{la_ann*100:+5.1f}% MDD{lmd*100:+5.0f}%  多空{ls_ann*100:+5.1f}% Sh{ls_sh:+.2f}")

print(f"\n耗时: {time.time()-t0:.0f}s")
