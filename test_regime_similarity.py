# -*- coding: utf-8 -*-
"""
Man Group 历史相似性体制检测
=============================
方法: 每月计算状态向量(7变量)→找历史上欧氏距离最近的K个月→看之后N月收益→生成多空信号
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

K_NEIGHBORS = 5    # 找5个最相似的历史月
LOOK_FWD = 3       # 看之后3个月收益
ROLLING_Z = 60     # 滚动60个月z-score标准化

print("="*60)
print("Man Group 相似性体制检测")
print("="*60)

# ===== 加载 =====
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# HS300
hs300 = con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close']
hs300_ret = hs300.pct_change(20)  # 20日动量

# 宏观变量
macro = con.execute("""
    SELECT trade_date, vix, usdcny, m2_growth, m1_growth, spx, gold, wti
    FROM macro_indicators ORDER BY trade_date
""").df()
macro['trade_date'] = pd.to_datetime(macro['trade_date']); macro = macro.set_index('trade_date')

# 北向
nb = con.execute("SELECT trade_date, net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date'] = pd.to_datetime(nb['trade_date']); nb = nb.set_index('trade_date')['net_flow']

# 两融
mg = con.execute("SELECT trade_date, margin_balance FROM margin_trading ORDER BY trade_date").df()
mg['trade_date'] = pd.to_datetime(mg['trade_date']); mg = mg.set_index('trade_date')['margin_balance']

# 全A股波动率
kline = con.execute("SELECT trade_date, close FROM kline_daily ORDER BY trade_date").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
market_ret = kline.groupby('trade_date')['close'].mean().pct_change()
market_vol = market_ret.rolling(20).std() * np.sqrt(252)
con.close()

# ===== 月度调仓日 =====
dates = sorted(set(hs300.index) & set(macro.index))
md = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    md.append(g.iloc[0])
md = sorted(md)
print("月度: %d (%s~%s)" % (len(md), md[0].date(), md[-1].date()))

# ===== 构建月度状态向量 =====
def align(series, mdates, fill_method='ffill'):
    s = series.sort_index().dropna()
    result = {}
    for d in mdates:
        vals = s[s.index <= d]
        if len(vals) > 0: result[d] = vals.iloc[-1]
    return pd.Series(result)

# 7个状态变量
state_vars = {}
state_vars['HS300_mom'] = align(hs300_ret, md)          # 市场动量
state_vars['VIX'] = align(macro['vix'], md)              # 恐慌
state_vars['USDCNY'] = align(macro['usdcny'], md)        # 汇率
state_vars['M1M2_spread'] = align(macro['m1_growth'] - macro['m2_growth'], md)  # 货币剪刀差
state_vars['Northbound'] = align(nb.rolling(60).sum(), md)  # 北向季度累计
state_vars['Margin_chg'] = align(mg.pct_change(3), md)   # 两融季度变化
state_vars['Volatility'] = align(market_vol, md)          # 市场波动率

state_df = pd.DataFrame(state_vars)
state_df = state_df.dropna()
print("有效月度: %d" % len(state_df))

# ===== 相似性匹配 =====
VAR_NAMES = list(state_df.columns)
print("变量: %s" % VAR_NAMES)

# 计算信号: 每月找K个最相似历史月→看后续收益
signals = {}
for i in range(ROLLING_Z + LOOK_FWD, len(state_df)):
    today_idx = state_df.index[i]

    # 滚动z-score标准化(只用今天之前的数据)
    lookback = state_df.iloc[i-ROLLING_Z:i]
    mu = lookback.mean(); std = lookback.std().replace(0, 1)
    today_z = (state_df.iloc[i] - mu) / std

    # 历史z-score(同样用滚动窗口标准化)
    hist_start = max(0, i - ROLLING_Z * 2)
    hist = state_df.iloc[hist_start:i-LOOK_FWD]  # 不包括最近几个月(避免信息泄漏)
    hist_z = (hist - mu) / std

    # 计算欧氏距离
    distances = np.sqrt(((hist_z - today_z.values) ** 2).sum(axis=1))

    # 找K个最近邻
    if len(distances) < K_NEIGHBORS: continue
    nearest_idx = distances.nsmallest(K_NEIGHBORS).index

    # 计算后续收益(这些相似月之后的LOOK_FWD月HS300收益)
    fwd_rets = []
    for hist_date in nearest_idx:
        hist_pos = list(state_df.index).index(hist_date)
        fwd_pos = hist_pos + LOOK_FWD
        if fwd_pos < len(state_df):
            fwd_rets.append(hs300[state_df.index[fwd_pos]] / hs300[state_df.index[hist_pos]] - 1)

    if fwd_rets:
        signals[today_idx] = np.mean(fwd_rets)

signals = pd.Series(signals)
print("信号: %d个月" % len(signals))

# ===== 评估: 信号 vs 实际后续收益 =====
actual_fwd = {}
for i in range(len(state_df) - LOOK_FWD):
    today = state_df.index[i]
    future = state_df.index[i + LOOK_FWD]
    if today in signals.index:
        actual_fwd[today] = hs300[future] / hs300[today] - 1

common = sorted(set(signals.index) & set(actual_fwd.keys()))
pred = np.array([signals[d] for d in common])
actual = np.array([actual_fwd[d] for d in common])

# IC
ic = np.corrcoef(pred, actual)[0,1] if len(pred) > 5 else 0
# 方向命中率
dir_hit = np.mean((pred > 0) == (actual > 0))
# 分位数测试
top_idx = pred >= np.percentile(pred, 67)
bot_idx = pred <= np.percentile(pred, 33)
top_ret = np.mean(actual[top_idx]) * 100 if top_idx.sum() > 0 else 0
bot_ret = np.mean(actual[bot_idx]) * 100 if bot_idx.sum() > 0 else 0

print("\n" + "="*60)
print("相似性体制信号评估")
print("="*60)
print("IC: %.4f | 方向命中: %.1f%% | 月数: %d" % (ic, dir_hit*100, len(common)))
print("信号高分位(看多)后%dm收益: %+.1f%%" % (LOOK_FWD, top_ret))
print("信号低分位(看空)后%dm收益: %+.1f%%" % (LOOK_FWD, bot_ret))
print("多空spread: %+.1f%%" % (top_ret - bot_ret))

# 按信号分桶
print("\n信号分桶(后%dmHS300收益):" % LOOK_FWD)
for pct in [0, 25, 50, 75, 100]:
    if pct == 0: mask = pred < np.percentile(pred, 25)
    elif pct == 25: mask = (pred >= np.percentile(pred, 25)) & (pred < np.percentile(pred, 50))
    elif pct == 50: mask = (pred >= np.percentile(pred, 50)) & (pred < np.percentile(pred, 75))
    else: mask = pred >= np.percentile(pred, 75)
    if mask.sum() > 0:
        avg_ret = np.mean(actual[mask]) * 100
        print("  %d%%分位: %+.1f%% (%d个月)" % (pct, avg_ret, mask.sum()))

# 对比: 和DD_SMART信号的相关性
print("\n信号vs DD_SMART:")
# DD_SMART: 2年高点回撤
hs300_m = align(hs300, md)
h2y = hs300_m.rolling(24).max()
dd = hs300_m / h2y - 1
dd_aligned = align(dd, md)
common_dd = sorted(set(signals.index) & set(dd_aligned.dropna().index))
sig_dd_corr = np.corrcoef([signals[d] for d in common_dd], [dd_aligned[d] for d in common_dd])[0,1]
print("  相似性信号 vs DD回撤 相关性: %.3f" % sig_dd_corr)
print("  (低相关=互补, 高相关=冗余)")

print("\n耗时: %.0fs" % (time.time()-t0))
