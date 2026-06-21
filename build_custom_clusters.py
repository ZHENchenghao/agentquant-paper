# -*- coding: utf-8 -*-
"""数据驱动行业聚类 · 个股收益相关→100组 → 扩充轮动池
=====================================================
方法: PCA降维 + KMeans聚类 → 100个"伪行业"
每组的日收益=组内个股等权平均 → 形成100条净值曲线 → WF因子测试
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
t0 = time.time()

print("=" * 70)
print("数据驱动聚类 · 扩充行业轮动池 30→100")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ===== 1. 选样: 成交活跃+历史长的股票 =====
print("[1] 选样...")
# 用最近3年的数据筛选
stocks = con.execute("""
    SELECT ts_code, COUNT(*) as n_days, AVG(vol) as avg_vol
    FROM kline_daily
    WHERE trade_date >= DATE '2023-01-01' AND vol > 0
    GROUP BY ts_code
    HAVING n_days > 500 AND avg_vol > 1000000
""").df()
active_stocks = stocks['ts_code'].tolist()
print(f"  活跃股: {len(active_stocks)}只")

# 限制到2000只(计算可行性)
sample_stocks = active_stocks[:2000]

# 取日收益
# 分批查询避免SQL过长
kline_parts = []
batch_size = 500
for i in range(0, len(sample_stocks), batch_size):
    batch = sample_stocks[i:i+batch_size]
    placeholders = ','.join([f"'{s}'" for s in batch])
    part = con.execute(f"""
        SELECT ts_code, trade_date, close
        FROM kline_daily
        WHERE ts_code IN ({placeholders})
          AND trade_date >= DATE '2020-01-01'
    """).df()
    kline_parts.append(part)
kline = pd.concat(kline_parts, ignore_index=True)
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
kline = kline.sort_values(['ts_code', 'trade_date'])
kline['ret'] = kline.groupby('ts_code')['close'].pct_change()
kline = kline.dropna(subset=['ret'])
con.close()

# ===== 2. 计算收益相关性 → 聚类 =====
print("[2] 构建收益矩阵...")
# 用2023-2024年的数据做聚类(样本外之前)
train_data = kline[(kline['trade_date'] >= '2023-01-01') &
                   (kline['trade_date'] <= '2024-12-31')]

# 透视成股票×日期矩阵
ret_matrix = train_data.pivot(index='trade_date', columns='ts_code', values='ret')
# 删除缺失>30%的股票
ret_matrix = ret_matrix.loc[:, ret_matrix.isna().mean() < 0.3]
ret_matrix = ret_matrix.fillna(0)
print(f"  矩阵: {ret_matrix.shape[0]}天 × {ret_matrix.shape[1]}只")

# 计算股票间收益相关矩阵(用PCA近似避免完整相关矩阵)
# 对收益矩阵的转置做PCA
ret_array = ret_matrix.values.T  # stocks × days
n_stocks = ret_array.shape[0]
print(f"  聚类: {n_stocks}只 → 100组")

# PCA降维到50维
n_components = min(50, n_stocks - 1, ret_array.shape[1] - 1)
pca = PCA(n_components=n_components, random_state=42)
features = pca.fit_transform(ret_array)
print(f"  PCA: {ret_array.shape[1]}维 → {n_components}维 (解释方差{pca.explained_variance_ratio_.sum()*100:.0f}%)")

# KMeans 100组
n_clusters = 100
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(features)

# 股票→组映射
stock_cluster = dict(zip(ret_matrix.columns, clusters))
cluster_sizes = pd.Series(clusters).value_counts()
print(f"  聚类完成: {n_clusters}组, 平均每组{cluster_sizes.mean():.0f}只, 最少{cluster_sizes.min()}只, 最多{cluster_sizes.max()}只")

# ===== 3. 构建100条"行业"净值曲线(全历史) =====
print("[3] 构建100组净值...")
kline['cluster'] = kline['ts_code'].map(stock_cluster)
kline_valid = kline.dropna(subset=['cluster'])
kline_valid['cluster'] = kline_valid['cluster'].astype(int)

# 每组日收益=等权平均
cluster_daily = kline_valid.groupby(['cluster', 'trade_date'])['ret'].mean().reset_index()
cluster_daily = cluster_daily.rename(columns={'cluster': 'industry'})

# 累积净值
cluster_daily = cluster_daily.sort_values(['industry', 'trade_date'])
cluster_daily['close'] = cluster_daily.groupby('industry')['ret'].transform(
    lambda x: (1+x).cumprod())

# ===== 4. 月度因子+WF测试 =====
print("[4] 月度因子...")
cluster_daily['month'] = cluster_daily['trade_date'].dt.to_period('M')

monthly = cluster_daily.groupby(['industry', 'month']).agg(
    close=('close', 'last'),
    ret_1m=('ret', lambda x: (1+x).prod()-1),
    ret_3m=('ret', lambda x: (1+x.tail(63)).prod()-1 if len(x)>=40 else np.nan),
    ret_12m=('ret', lambda x: (1+x.tail(252)).prod()-1 if len(x)>=200 else np.nan),
    vol=('ret', lambda x: x.std()*np.sqrt(252)),
).reset_index()
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly['fwd_ret'] = monthly.groupby('industry')['ret_1m'].shift(-1)
monthly = monthly.dropna(subset=['fwd_ret', 'ret_1m'])
print(f"  月度: {len(monthly)}行, {monthly['industry'].nunique()}组, {monthly['month'].nunique()}月")

# WF测试
YEARS = sorted(set(d.year for d in monthly['month']))
TRAIN = 5
# 由于数据从2020开始, WF从2025开始太短
# 改用2023-2024训练, 2025-2026测试
# 实际上数据够2021-2024训练5年, 2025-2026测试2年...还是太短
# 我们用全样本来个快速的IC看看方向

print(f"\n[5] IC测试 (2021-2026):")
FACTORS = {
    'ret_1m': ('动量1月', 1),
    'ret_3m': ('动量3月', 1),
    'ret_12m': ('动量12月', 1),
    'vol': ('波动率', -1),
}

for f, (name, expected) in FACTORS.items():
    ics = []
    for m, grp in monthly.groupby('month'):
        valid = grp.dropna(subset=[f, 'fwd_ret'])
        if len(valid) > 10:
            ic = valid[f].rank().corr(valid['fwd_ret'].rank())
            if not np.isnan(ic): ics.append(ic)
    if ics:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std>0 else 0
        ir = mi/std*np.sqrt(12) if std>0 else 0
        print(f"  {name:<12s} IC={mi*100:+6.2f}% IR={ir:+5.2f} t={t:+5.2f} N={len(ics)}")

# 策略模拟
print(f"\n[6] 策略: 动量Top10:")
long_r = []
for m, grp in monthly.groupby('month'):
    valid = grp.dropna(subset=['ret_1m', 'fwd_ret'])
    if len(valid) < 20: continue
    valid = valid.copy()
    valid['score'] = valid.groupby('month')['ret_1m'].rank(pct=True)
    top10 = valid.nlargest(10, 'score')
    long_r.append(top10['fwd_ret'].mean())

if long_r:
    arr = np.array(long_r); n = len(arr)
    cum = np.prod(1+arr); ann = cum**(12/n)-1
    c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
    vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    # 等权基准
    eq_arr = np.array([grp['fwd_ret'].mean() for m, grp in monthly.groupby('month') if len(grp)>10])
    eq_cum = np.prod(1+eq_arr); eq_ann = eq_cum**(12/len(eq_arr))-1

    print(f"  动量Top10: 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.0f}% 累积{cum-1:+5.1%} ({n}月)")
    print(f"  等权基准:  年化{eq_ann*100:+5.1f}%")

# === 对比: 30行业同等条件 ===
print(f"\n[7] 对比: 30行业vs100组 (同窗口2021-2026)")
# 用原始30行业同期的IC
ind_orig = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
orig = ind_orig.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2020-01-01' ORDER BY industry, trade_date
""").df()
orig['trade_date'] = pd.to_datetime(orig['trade_date'])
ind_orig.close()

orig['month'] = orig['trade_date'].dt.to_period('M')
orig_m = orig.groupby(['industry', 'month'])['close'].last().reset_index()
orig_m['month'] = orig_m['month'].dt.to_timestamp()
orig_m['ret_1m'] = orig_m.groupby('industry')['close'].pct_change()
orig_m['fwd_ret'] = orig_m.groupby('industry')['ret_1m'].shift(-1)
orig_m = orig_m.dropna(subset=['fwd_ret'])

for f, (name, expected) in FACTORS.items():
    if f == 'ret_1m':
        ics = []
        for m, grp in orig_m.groupby('month'):
            valid = grp.dropna(subset=['ret_1m', 'fwd_ret'])
            if len(valid) > 5:
                ic = valid['ret_1m'].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
        if ics:
            mi = np.mean(ics); std = np.std(ics)
            t = mi/std*np.sqrt(len(ics)) if std>0 else 0
            print(f"  30行业-{name}: IC={mi*100:+6.2f}% t={t:+5.2f}")

print(f"\n耗时: {time.time()-t0:.0f}s")
