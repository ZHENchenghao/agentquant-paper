# -*- coding: utf-8 -*-
"""国家队护盘信号 · 从ETF异常量+尾盘拉升检测"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("国家队护盘信号构建")
print("=" * 60)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# === 1. 加载核心ETF + HS300基准 ===
NT_ETFS = ['510050.SH', '510300.SH', '510500.SH', '159915.SZ']  # 国家队主要工具

etf = con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, volume, name
    FROM etf_daily
    WHERE ts_code IN ('510050.SH','510300.SH','510500.SH','159915.SZ')
    ORDER BY ts_code, trade_date
""").df()
etf['trade_date'] = pd.to_datetime(etf['trade_date'])
print(f"[1] 国家队ETF: {etf['ts_code'].nunique()}只, {len(etf)}行")

hs300 = con.execute("SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300 = hs300.set_index('trade_date')['close']

con.close()

# === 2. 逐只ETF计算异常指标 ===
results = []

for code, grp in etf.groupby('ts_code'):
    grp = grp.sort_values('trade_date').set_index('trade_date')
    name = grp['name'].iloc[0]

    # 基础指标
    grp['ret'] = grp['close'].pct_change()
    grp['vol_ma20'] = grp['volume'].rolling(20).mean()
    grp['vol_ratio'] = grp['volume'] / grp['vol_ma20'].replace(0, np.nan)

    # 日内形态
    grp['intraday'] = (grp['close'] - grp['open']) / (grp['high'] - grp['low'] + 0.001)  # 收盘在日内位置
    grp['tail_lift'] = (grp['close'] - grp['open']) / grp['open']  # 日内涨幅(尾盘拉升)

    # 振幅
    grp['amplitude'] = (grp['high'] - grp['low']) / grp['close']

    # 国家队信号打分
    score = pd.Series(0.0, index=grp.index)

    # D1: 放量(量比>2) → +2分
    score += (grp['vol_ratio'] > 2.0).astype(float) * 2

    # D2: 尾盘拉升(日内收涨且收在日内高位>60%) → +2分
    score += ((grp['tail_lift'] > 0.002) & (grp['intraday'] > 0.6)).astype(float) * 2

    # D3: 逆势放量(ETF放量但当日跌) → +3分(护盘特征)
    score += ((grp['vol_ratio'] > 1.8) & (grp['ret'] < -0.005)).astype(float) * 3

    # D4: 连续放量(昨今皆放量) → +1分(持续护盘)
    score += ((grp['vol_ratio'] > 1.5) & (grp['vol_ratio'].shift(1) > 1.5)).astype(float) * 1

    # D5: 超大量(量比>4) → +3分(强力护盘)
    score += (grp['vol_ratio'] > 4.0).astype(float) * 3

    # 持仓量变化代理(连续3日净流入)
    grp['flow_3d'] = (grp['ret'] * grp['volume']).rolling(3).sum()
    score += ((grp['flow_3d'] > 0) & (grp['vol_ratio'] > 1.5)).astype(float) * 1

    grp['nt_score'] = score.clip(0, 10)

    results.append(grp[['close', 'volume', 'vol_ratio', 'tail_lift', 'intraday', 'nt_score', 'name']])

nt_df = pd.concat(results).reset_index()
print(f"[2] 单ETF信号完成: {len(nt_df)}行")

# === 3. 汇总国家队综合信号(多ETF合并) ===
daily_nt = nt_df.groupby('trade_date').agg(
    nt_score_sum=('nt_score', 'sum'),
    nt_score_max=('nt_score', 'max'),
    nt_etf_cnt=('nt_score', lambda x: (x > 3).sum()),  # 多少只ETF>3分
    vol_ratio_mean=('vol_ratio', 'mean'),
    tail_lift_mean=('tail_lift', 'mean'),
).reset_index()

# 综合评级
def classify_nt(row):
    s = row['nt_score_sum']
    cnt = row['nt_etf_cnt']
    if s >= 12 and cnt >= 2: return 'HEAVY_BUY'     # 强力护盘
    elif s >= 8 and cnt >= 1: return 'MODERATE_BUY'  # 中等护盘
    elif s >= 5: return 'LIGHT_BUY'                   # 轻度护盘
    elif s >= 2: return 'PRESENCE'                    # 有动作
    else: return 'DORMANT'                            # 休眠

daily_nt['nt_action'] = daily_nt.apply(classify_nt, axis=1)

# === 4. 合并HS300收益 ===
daily_nt = daily_nt.set_index('trade_date')
hs300_aligned = hs300.reindex(daily_nt.index)
daily_nt['hs300'] = hs300_aligned
daily_nt['hs300_ret'] = hs300_aligned.pct_change()
daily_nt['hs300_fwd5d'] = hs300_aligned.pct_change(5).shift(-5)
daily_nt['hs300_fwd20d'] = hs300_aligned.pct_change(20).shift(-20)

# === 5. 评估国家队信号的预测力 ===
print("\n" + "=" * 60)
print("国家队信号评估")
print("=" * 60)

# 信号分布
for action in ['HEAVY_BUY', 'MODERATE_BUY', 'LIGHT_BUY', 'PRESENCE', 'DORMANT']:
    cnt = (daily_nt['nt_action'] == action).sum()
    print(f"  {action}: {cnt}天 ({cnt/len(daily_nt)*100:.1f}%)")

# 信号日后N日收益
print("\n信号日后5日/20日HS300收益:")
for action in ['HEAVY_BUY', 'MODERATE_BUY', 'LIGHT_BUY']:
    mask = daily_nt['nt_action'] == action
    if mask.sum() > 5:
        fwd5 = daily_nt.loc[mask, 'hs300_fwd5d'].mean() * 100
        fwd20 = daily_nt.loc[mask, 'hs300_fwd20d'].mean() * 100
        print(f"  {action}: 5日={fwd5:+.2f}%  20日={fwd20:+.2f}% (信号{mask.sum()}次)")

# IC: 信号vs未来收益
valid = daily_nt.dropna(subset=['hs300_fwd5d', 'hs300_fwd20d'])
if len(valid) > 100:
    ic5 = valid['nt_score_sum'].corr(valid['hs300_fwd5d'])
    ic20 = valid['nt_score_sum'].corr(valid['hs300_fwd20d'])
    print(f"\nIC(nt_score vs fwd5d): {ic5:.4f}")
    print(f"IC(nt_score vs fwd20d): {ic20:.4f}")

# 国家队信号 vs DD_SMART(大盘回撤)的相关性
dd = hs300_aligned / hs300_aligned.rolling(504).max() - 1
common = daily_nt.index.intersection(dd.dropna().index)
corr_nt_dd = daily_nt.loc[common, 'nt_score_sum'].corr(dd.loc[common])
print(f"国家队信号 vs DD回撤相关: {corr_nt_dd:.3f} (负值→逆势护盘✅)")

# 看国家队干预后市场是否止跌 (在回撤>10%时介入)
stress_mask = dd < -0.10
stress_nt = daily_nt[stress_mask]
if len(stress_nt) > 10:
    heavy_stress = stress_nt[stress_nt['nt_action'].isin(['HEAVY_BUY', 'MODERATE_BUY'])]
    if len(heavy_stress) > 3:
        print(f"\n市场回撤>10%时国家队强力介入: {len(heavy_stress)}次")
        print(f"  介入后5日: {heavy_stress['hs300_fwd5d'].mean()*100:+.2f}%")
        print(f"  介入后20日: {heavy_stress['hs300_fwd20d'].mean()*100:+.2f}%")
        print(f"  命中率(5日正): {(heavy_stress['hs300_fwd5d']>0).mean()*100:.0f}%")

# === 6. 保存 ===
daily_nt = daily_nt.reset_index()
daily_nt.to_parquet('D:/AgentQuant/our/cache/national_team_signal.parquet')
print(f"\nSaved: cache/national_team_signal.parquet ({len(daily_nt)}行)")

# 最新信号
latest = daily_nt.sort_values('trade_date').iloc[-5:]
print("\n最新5日信号:")
for _, r in latest.iterrows():
    print(f"  {r['trade_date'].date()}  nt={r['nt_action']:14s} score={r['nt_score_sum']:.0f} etf_cnt={r['nt_etf_cnt']:.0f} vol_ratio={r['vol_ratio_mean']:.2f} tail={r['tail_lift_mean']*100:+.2f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
