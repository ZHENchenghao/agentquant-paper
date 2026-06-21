# -*- coding: utf-8 -*-
"""行业间领先滞后 · 周频 · 清华论文方法论
==============================================
核心: 上游行业收益→预测下游行业未来1-4周收益
清华何晨宇(2023): A股产业链领先滞后在周频显著, 不在月频
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
from scipy import stats
t0 = time.time()

print("=" * 70)
print("行业领先滞后 · 周频测试 (清华方法论)")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 行业指数日线
ind = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2010-01-01' ORDER BY industry, trade_date
""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])
con.close()

# ===== 1. 周频收益 =====
ind = ind.sort_values(['industry', 'trade_date'])
ind['week'] = ind['trade_date'].dt.isocalendar().year.astype(str) + '-W' + \
              ind['trade_date'].dt.isocalendar().week.astype(str).str.zfill(2)

weekly = ind.groupby(['industry', 'week'])['close'].last().reset_index()
# 转为datetime
weekly['week_dt'] = pd.to_datetime(weekly['week'] + '-1', format='%Y-W%W-%w')
weekly = weekly.sort_values(['industry', 'week_dt'])

weekly['ret'] = weekly.groupby('industry')['close'].pct_change()
print(f"[1] 周频: {len(weekly)}行, {weekly['industry'].nunique()}行业, "
      f"{weekly['week_dt'].min().date()}~{weekly['week_dt'].max().date()}")

# ===== 2. 构建行业对(上下游) =====
# 基于券商产业链分类
SUPPLY_CHAIN_PAIRS = [
    # 上游→下游
    ('有色金属', '电力设备'),  # 锂/钴→电池/光伏
    ('有色金属', '电子'),      # 稀土/铜→电子元件
    ('石油石化', '基础化工'),   # 原油→化工品
    ('基础化工', '医药生物'),   # 原料药
    ('基础化工', '纺织服饰'),   # 化纤→纺织
    ('钢铁', '建筑装饰'),      # 钢材→基建
    ('钢铁', '机械设备'),      # 钢材→机械
    ('钢铁', '汽车'),         # 钢材→汽车
    ('煤炭', '公用事业'),      # 煤→电力
    ('煤炭', '钢铁'),         # 焦煤→钢厂
    # 中游→下游
    ('电子', '计算机'),       # 硬件→软件
    ('电子', '通信'),         # 元件→通信设备
    ('电子', '传媒'),         # 硬件→内容
    ('电力设备', '汽车'),     # 电池→电动车
    ('电力设备', '公用事业'),  # 光伏→电力
    ('机械设备', '汽车'),     # 设备→车厂
    ('机械设备', '国防军工'),  # 设备→军工
    # 服务链
    ('银行', '房地产'),       # 信贷→地产
    ('房地产', '建筑材料'),   # 地产→建材
    ('房地产', '家用电器'),   # 地产→家电
    ('交通运输', '商贸零售'),  # 物流→零售
]

# 去重
seen = set(); pairs = []
for up, down in SUPPLY_CHAIN_PAIRS:
    key = f'{up}->{down}'
    if key not in seen and up != down:
        seen.add(key)
        pairs.append((up, down, key))
print(f"\n[2] 产业链对: {len(pairs)}个")

# ===== 3. Walk-Forward: 滚动窗口计算领先滞后IC =====
TRAIN_WEEKS = 260  # 5年周
MIN_TRAIN = 100
FWD_WEEKS = [1, 2, 3, 4]  # 预测未来1-4周

wf_results = {fwd: [] for fwd in FWD_WEEKS}
wf_by_pair = {}

# 遍历所有可能的测试周
all_weeks = sorted(weekly['week_dt'].unique())
test_start_idx = TRAIN_WEEKS + 10

for test_idx in range(test_start_idx, len(all_weeks)):
    test_week = all_weeks[test_idx]
    train_end_idx = test_idx - 1
    train_start_idx = max(0, train_end_idx - TRAIN_WEEKS)

    for up_ind, down_ind, pair_name in pairs:
        # 取上游和下游的收益序列
        up_data = weekly[weekly['industry'] == up_ind].set_index('week_dt')['ret'].dropna()
        down_data = weekly[weekly['industry'] == down_ind].set_index('week_dt')['ret'].dropna()

        if len(up_data) < MIN_TRAIN or len(down_data) < MIN_TRAIN:
            continue

        # 上游当期收益
        if test_week not in up_data.index:
            continue
        # ensure scalar
        up_ret_val = up_data.loc[test_week]
        if hasattr(up_ret_val, '__len__') and not isinstance(up_ret_val, (int, float, np.floating)):
            up_ret_val = up_ret_val.iloc[0] if len(up_ret_val) > 0 else np.nan
        up_ret = float(up_ret_val)
        if np.isnan(up_ret):
            continue

        for fwd in FWD_WEEKS:
            # 未来第fwd周
            future_idx = test_idx + fwd
            if future_idx >= len(all_weeks):
                continue
            future_week = all_weeks[future_idx]
            if future_week not in down_data.index:
                continue
            down_fwd_val = down_data.loc[future_week]
            if hasattr(down_fwd_val, '__len__') and not isinstance(down_fwd_val, (int, float, np.floating)):
                down_fwd_val = down_fwd_val.iloc[0] if len(down_fwd_val) > 0 else np.nan
            down_fwd = float(down_fwd_val)
            if np.isnan(down_fwd):
                continue

            # 用训练窗数据算上游→下游的时序相关
            train_up = up_data[(up_data.index >= all_weeks[train_start_idx]) &
                               (up_data.index <= all_weeks[train_end_idx])]
            train_down = down_data[(down_data.index >= all_weeks[train_start_idx]) &
                                   (down_data.index <= all_weeks[train_end_idx])]

            # 对齐
            common = train_up.index.intersection(train_down.index)
            if len(common) < 50:
                continue

            # 计算训练窗相关性(上游→下游领先fwd周)
            # 上游t时刻 vs 下游t+fwd时刻
            up_vals = []; down_vals = []
            for i in range(len(common) - fwd):
                t_week = common[i]; t_fwd_week = common[i + fwd]
                if t_week in up_data.index and t_fwd_week in down_data.index:
                    uv = up_data[t_week]; dv = down_data[t_fwd_week]
                    if isinstance(uv, (int, float, np.floating)) and isinstance(dv, (int, float, np.floating)):
                        if not np.isnan(uv) and not np.isnan(dv):
                            up_vals.append(float(uv))
                            down_vals.append(float(dv))

            if len(up_vals) < 30:
                continue

            corr = np.corrcoef(np.array(up_vals), np.array(down_vals))[0, 1]
            if np.isnan(corr):
                continue

            # 信号: 上游本周收益 × 训练期相关性方向
            direction = 1 if corr > 0 else -1
            signal = up_ret * direction

            wf_results[fwd].append({
                'week': test_week,
                'pair': pair_name,
                'up_ret': up_ret,
                'down_fwd': down_fwd,
                'signal': signal,
                'train_corr': corr,
            })

# ===== 4. 评估 =====
print(f"\n[3] 领先滞后WF结果:")
print(f"{'前向':>5s} {'信号数':>7s} {'IC':>8s} {'方向命中':>8s} {'多空spread':>10s}")

for fwd in FWD_WEEKS:
    data = wf_results[fwd]
    if len(data) < 30:
        continue

    signals = np.array([d['signal'] for d in data])
    actuals = np.array([d['down_fwd'] for d in data])
    n = len(signals)

    # IC
    ic = np.corrcoef(signals, actuals)[0, 1]

    # 方向命中
    dir_hit = np.mean((signals > 0) == (actuals > 0))

    # 多空spread
    top_mask = signals >= np.percentile(signals, 67)
    bot_mask = signals <= np.percentile(signals, 33)
    top_ret = np.mean(actuals[top_mask]) * 100
    bot_ret = np.mean(actuals[bot_mask]) * 100

    print(f"  {fwd}周   {n:>7d} {ic:+8.4f} {dir_hit*100:>7.1f}% {top_ret-bot_ret:+9.2f}%")

# ===== 5. 最好的传导对 =====
print(f"\n[4] 最优产业链传导对:")
pair_perf = {}
for fwd in [1, 3]:
    data = wf_results[fwd]
    for d in data:
        pair = d['pair']
        if pair not in pair_perf:
            pair_perf[pair] = {'signals': [], 'actuals': [], 'fwd': fwd}
        pair_perf[pair]['signals'].append(d['signal'])
        pair_perf[pair]['actuals'].append(d['down_fwd'])

pair_ic = {}
for pair, perf in pair_perf.items():
    if len(perf['signals']) > 30:
        ic = np.corrcoef(perf['signals'], perf['actuals'])[0, 1]
        dir_hit = np.mean((np.array(perf['signals'])>0) == (np.array(perf['actuals'])>0))
        pair_ic[pair] = (ic, dir_hit, len(perf['signals']))

for pair, (ic, dh, n) in sorted(pair_ic.items(), key=lambda x: abs(x[1][0]), reverse=True)[:15]:
    print(f"  {pair:<30s} IC={ic:+7.4f}  方向={dh*100:.0f}%  n={n}")

# ===== 6. 行业轮动策略: 用领先行业信号 =====
print(f"\n[5] 产业链信号→行业轮动策略")

# 对每个行业, 计算其上游行业的加权信号
industries = sorted(weekly['industry'].unique())
# 构建上游-下游映射
upstream_map = {}  # downstream → [(upstream, fwd_weeks)]
for up, down, name in pairs:
    if down not in upstream_map:
        upstream_map[down] = []
    upstream_map[down].append(up)

# WF: 每年重算, 月度调仓
# 每月末, 对每个行业, 看过去4周的上游信号, 加权打分
monthly = ind.groupby(['industry', ind['trade_date'].dt.to_period('M')])['close'].last().reset_index()
monthly.columns = ['industry', 'month', 'close']
monthly['month'] = monthly['month'].dt.to_timestamp()
monthly['ret'] = monthly.groupby('industry')['close'].pct_change()
monthly['fwd_ret'] = monthly.groupby('industry')['ret'].shift(-1)
monthly = monthly.dropna(subset=['fwd_ret'])

YEARS = sorted(set(d.year for d in monthly['month']))
TRAIN_YEARS = 5; WF_START = YEARS[0] + TRAIN_YEARS + 1

# 简单策略: 用过去4周的上游行业收益作为signal
# 如果上游行业近4周涨→下游应该跟涨
chain_rets = []; mom_rets = []; eq_rets = []
for test_yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
    test = monthly[(monthly['month'] >= ts) & (monthly['month'] <= te)]
    for m, grp in test.groupby('month'):
        if len(grp) < 5: continue
        n = max(1, len(grp)//4)

        # 产业链信号: 上游近1月收益
        grp = grp.copy()
        grp['chain_signal'] = 0.0
        for _, row in grp.iterrows():
            ind_name = row['industry']
            if ind_name in upstream_map:
                upstream_inds = upstream_map[ind_name][:3]  # 最多3个上游
                upstream_rets = []
                for up in upstream_inds:
                    up_row = grp[grp['industry'] == up]
                    if len(up_row) > 0:
                        upstream_rets.append(up_row['ret'].iloc[0])
                if upstream_rets:
                    grp.loc[_, 'chain_signal'] = np.mean(upstream_rets)  # 使用当月实际收益

        top = grp.nlargest(n, 'chain_signal')
        if len(top) > 0:
            chain_rets.append(top['fwd_ret'].mean() - 0.003)

        # 动量基准
        grp['mom'] = grp.groupby('month')['ret'].rank(pct=True)
        top_mom = grp.nlargest(n, 'mom')
        mom_rets.append(top_mom['fwd_ret'].mean() - 0.003)

        eq_rets.append(grp['fwd_ret'].mean())

# 评估
print(f"\n{'策略':<20s} {'年化':>8s} {'Sharpe':>7s} {'MDD':>7s}")
for name, rets in [('产业链信号', chain_rets), ('动量', mom_rets), ('等权', eq_rets)]:
    arr = np.array(rets); n = len(arr)
    if n < 10: continue
    cum = np.prod(1+arr); ann = cum**(12/n)-1
    vol = np.std(arr)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    c = np.cumprod(1+arr); mdd = np.min(c/np.maximum.accumulate(c)-1)
    print(f"  {name:<20s} {ann*100:+7.1f}% {sh:+6.2f} {mdd*100:+6.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
