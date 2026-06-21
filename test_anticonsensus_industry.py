# -*- coding: utf-8 -*-
"""反共识剪刀差 · 行业方向预测力Walk-Forward检验
==============================================
核心假设: 当短期情绪(1m收益+量比)远超长期现实(12m收益+波动率)时,
         → 拥挤, 下月大概率跑输
         当短期情绪远低于长期现实时,
         → 冷门, 下月大概率跑赢

方法: 每月末计算每个申万行业剪刀差 → IC → 多空组合WF回测
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("反共识剪刀差 · 行业方向预测")
print("=" * 60)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# === 1. 加载行业指数 ===
ind = con.execute("""
    SELECT industry, trade_date, close
    FROM proxy_industry_daily
    WHERE trade_date >= DATE '2010-01-01'
    ORDER BY industry, trade_date
""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])
print(f"[1] 行业指数: {ind['industry'].nunique()}行业, {len(ind)}行")

# === 2. 月度信号构建 ===
ind['month'] = ind['trade_date'].dt.to_period('M')

# 每月末取收盘
monthly = ind.sort_values(['industry', 'trade_date']).groupby(['industry', 'month']).agg(
    close=('close', 'last'),
    close_ma5=('close', lambda x: x.iloc[-5:].mean() if len(x)>=5 else x.iloc[-1]),
    high_1m=('close', 'max'),
    low_1m=('close', 'min'),
    days=('trade_date', 'count')
).reset_index()

# 月度收益
monthly['ret_1m'] = monthly.groupby('industry')['close'].pct_change()
monthly['ret_3m'] = monthly.groupby('industry')['close'].pct_change(3)
monthly['ret_12m'] = monthly.groupby('industry')['close'].pct_change(12)

# === 3. 剪刀差信号 ===
# 共识代理: 1月收益(短期情绪) + 当月涨幅范围(日内波动代表情绪强度)
monthly['range_1m'] = (monthly['high_1m'] - monthly['low_1m']) / monthly['close_ma5']
monthly['consensus_raw'] = monthly['ret_1m'].fillna(0) * 0.6 + monthly['range_1m'].fillna(0) * 0.4

# 现实代理: 12月收益(长期趋势) + 波动率倒数(低波=稳健)
monthly['vol_12m'] = monthly.groupby('industry')['ret_1m'].transform(
    lambda x: x.rolling(12).std())
monthly['reality_raw'] = monthly['ret_12m'].fillna(0) * 0.5 + (1/monthly['vol_12m'].replace(0, 1)).fillna(0) * 0.5

# 标准化到横截面百分位(每月内排名)
monthly['consensus_z'] = monthly.groupby('month')['consensus_raw'].rank(pct=True)
monthly['reality_z'] = monthly.groupby('month')['reality_raw'].rank(pct=True)

# 剪刀差: 正=情绪过热, 负=情绪过冷
monthly['divergence'] = monthly['consensus_z'] - monthly['reality_z']

# 目标: 下月收益
monthly['fwd_ret'] = monthly.groupby('industry')['ret_1m'].shift(-1)

# 清洗
monthly = monthly.dropna(subset=['divergence', 'fwd_ret'])
monthly['month'] = monthly['month'].dt.to_timestamp()
print(f"[2] 月度数据: {len(monthly)}行, {monthly['month'].nunique()}月")

# === 4. Walk-Forward IC检验 ===
print("\n" + "=" * 60)
print("WF · 月度横截面IC")
print("=" * 60)

YEARS = sorted(set(d.year for d in monthly['month']))
TRAIN_YEARS = 5
FY = YEARS[0] + TRAIN_YEARS + 1

ic_list = []; dir_hit_list = []; long_ret_list = []; short_ret_list = []

for test_yr in range(FY, YEARS[-1] + 1):
    ts = pd.Timestamp(f'{test_yr}-01-01')
    te = pd.Timestamp(f'{test_yr}-12-31')
    test_data = monthly[(monthly['month'] >= ts) & (monthly['month'] <= te)]

    for m, grp in test_data.groupby('month'):
        if len(grp) < 5:
            continue
        # Rank IC
        ic = grp['divergence'].rank().corr(grp['fwd_ret'].rank())
        if not np.isnan(ic):
            ic_list.append(ic)

        # 方向: 反共识做多(剪刀差负=冷门) vs 拥挤做空(剪刀差正=热门)
        grp_sorted = grp.sort_values('divergence')
        n = len(grp_sorted)
        long_n = max(1, n // 3)
        long_ret = grp_sorted.head(long_n)['fwd_ret'].mean()
        short_ret = grp_sorted.tail(long_n)['fwd_ret'].mean()

        long_ret_list.append(long_ret)
        short_ret_list.append(short_ret)
        dir_hit_list.append(1 if long_ret > short_ret else 0)

# === 5. 评估 ===
mean_ic = np.mean(ic_list)
ic_ir = mean_ic / np.std(ic_list) * np.sqrt(12) if np.std(ic_list) > 0 else 0
dir_hit = np.mean(dir_hit_list)
long_avg = np.mean(long_ret_list) * 100
short_avg = np.mean(short_ret_list) * 100
spread = long_avg - short_avg

print(f"月度IC均值: {mean_ic:.4f}")
print(f"IC IR: {ic_ir:.2f}")
print(f"方向命中(冷门>拥挤): {dir_hit*100:.1f}% ({sum(dir_hit_list)}/{len(dir_hit_list)}月)")
print(f"冷门(低剪刀差)月均: {long_avg:+.2f}%")
print(f"拥挤(高剪刀差)月均: {short_avg:+.2f}%")
print(f"多空spread: {spread:+.2f}%")

# === 6. 累积多空收益 ===
long_cum = np.cumprod(1 + np.array(long_ret_list))
short_cum = np.cumprod(1 + np.array(short_ret_list))
print(f"\n累积(冷门做多): {long_cum[-1]-1:+.1%}")
print(f"累积(拥挤做多): {short_cum[-1]-1:+.1%}")

# === 7. 分层单调性 ===
print("\n分层单调性检验(按剪刀差5分位):")
monthly['div_q'] = monthly.groupby('month')['divergence'].transform(
    lambda x: pd.qcut(x, 5, labels=['Q1(冷门)', 'Q2', 'Q3', 'Q4', 'Q5(拥挤)'], duplicates='drop'))
q_ret = monthly.groupby('div_q')['fwd_ret'].mean() * 100
for q, r in q_ret.items():
    print(f"  {q}: {r:+.2f}%")

# 单调性: Q1应该最好, Q5应该最差
q_order = ['Q1(冷门)', 'Q2', 'Q3', 'Q4', 'Q5(拥挤)']
q_vals = [q_ret.get(q, 0) for q in q_order if q in q_ret.index]
mono = all(q_vals[i] >= q_vals[i+1] for i in range(len(q_vals)-1))
print(f"单调性: {'通过' if mono else '不通过'}(预期Q1>Q5)")

# === 8. vs 简单动量的增量信息 ===
print("\n增量信息检验(vs 1月动量):")
# 控制1月动量后，剪刀差还有预测力吗
monthly['ret_1m_z'] = monthly.groupby('month')['ret_1m'].rank(pct=True)
for m, grp in monthly.dropna(subset=['ret_1m_z']).groupby('month'):
    if len(grp) > 5:
        # 残差化: 剪刀差对1月动量回归后的残差
        from scipy import stats
        try:
            slope, intercept, r, p, se = stats.linregress(grp['ret_1m_z'], grp['divergence'])
            residual = grp['divergence'] - (slope * grp['ret_1m_z'] + intercept)
            monthly.loc[grp.index, 'div_residual'] = residual
        except:
            pass

if 'div_residual' in monthly.columns:
    monthly_valid = monthly.dropna(subset=['div_residual'])
    residual_ic = []
    for m, grp in monthly_valid.groupby('month'):
        if len(grp) > 5:
            ic = grp['div_residual'].rank().corr(grp['fwd_ret'].rank())
            if not np.isnan(ic):
                residual_ic.append(ic)
    if residual_ic:
        print(f"残差IC(控制动量后): {np.mean(residual_ic):.4f}")
        print(f"原始IC: {mean_ic:.4f}")
        print(f"增量: {np.mean(residual_ic) - mean_ic:+.4f}")

con.close()
print(f"\n耗时: {time.time()-t0:.0f}s")
