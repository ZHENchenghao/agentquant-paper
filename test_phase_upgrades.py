# -*- coding: utf-8 -*-
"""因子升级路线图 · 三阶段数据测试
Phase 1: NLP情绪层
Phase 2: 跨资产传导
Phase 3: 因子失效检测
每阶段独立测试，用IC变化判定是否有效
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE = 'cache/factors_all.parquet'

print('=' * 80)
print('因子升级路线图 · 三阶段数据测试')
print('=' * 80)

# ============================================================
# 加载基础数据
# ============================================================
print('\n[0/4] 加载基础数据...')
factors = pd.read_parquet(CACHE)
con = duckdb.connect(DB, read_only=True)

# 构建target (20日超额收益)
target = con.execute("""
    SELECT s.ts_code, s.trade_date,
           (s.fc/s.close - 1) - (x.fc/x.close - 1) AS excess_ret
    FROM (
        SELECT ts_code, trade_date, close,
               LEAD(close, 20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
        FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16'
    ) s
    JOIN (
        SELECT trade_date, close,
               LEAD(close, 20) OVER(ORDER BY trade_date) AS fc
        FROM kline_daily WHERE ts_code='sh000300'
    ) x ON s.trade_date = x.trade_date
    WHERE s.fc IS NOT NULL
""").df()

print(f'  因子缓存: {factors.shape[0]}行 × {factors.shape[1]}列, {factors.ts_code.nunique()}只')
print(f'  Target覆盖: {target.trade_date.min()} ~ {target.trade_date.max()}')
print(f'  日期范围: {factors.trade_date.min()} ~ {factors.trade_date.max()}')

# ============================================================
# Phase 1: NLP情绪层
# ============================================================
print('\n' + '=' * 80)
print('Phase 1: NLP情绪层测试')
print('=' * 80)

# 1a. 从DuckDB取新闻数据
news = con.execute("""
    SELECT title, content, source, publish_date, sector_tags
    FROM news_articles
    WHERE publish_date >= '2026-04-01'
    ORDER BY publish_date
""").df()
print(f'\n[1a] 新闻数据: {len(news)}条, {news.publish_date.min()} ~ {news.publish_date.max()}')

# 1b. 简易情感词典打分 (不需要transformers, 用关键词)
BULL_WORDS = ['暴涨', '大涨', '飙升', '突破', '新高', '利好', '超预期', '强劲', '爆发',
              '涨停', '翻倍', '增持', '回购', '业绩增长', '放量', '主升浪', '机构买入',
              '上调', '买入评级', '订单', '中标', '量产', '盈利', '景气', '复苏']
BEAR_WORDS = ['暴跌', '崩盘', '闪崩', '跌停', '亏损', '爆雷', '退市', '减持', '利空',
              '制裁', '关税', '监管', '调查', '诉讼', '下调', '卖出评级', '违约',
              '资金链', '跑路', '造假', 'ST', '*ST', '业绩下滑', '低于预期', '踩踏']

def sentiment_score(text):
    """-1到+1情感分"""
    if pd.isna(text) or not text:
        return 0.0
    bull = sum(1 for w in BULL_WORDS if w in str(text))
    bear = sum(1 for w in BEAR_WORDS if w in str(text))
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / max(total, 1)

news['sentiment'] = news['title'].fillna('').apply(sentiment_score)
news['publish_date'] = pd.to_datetime(news['publish_date'])

# 按日期聚合市场情绪
daily_sent = news.groupby('publish_date')['sentiment'].agg(['mean', 'count', 'std']).reset_index()
daily_sent.columns = ['trade_date', 'sent_mean', 'sent_count', 'sent_std']
daily_sent['trade_date'] = pd.to_datetime(daily_sent['trade_date'])

print(f'[1b] 情感分布: mean={daily_sent.sent_mean.mean():+.3f}, '
      f'std={daily_sent.sent_mean.std():.3f}, '
      f'日均{daily_sent.sent_count.mean():.0f}条')
print(f'  多头日(sent>0): {(daily_sent.sent_mean>0).sum()}/{len(daily_sent)}')
print(f'  空头日(sent<0): {(daily_sent.sent_mean<0).sum()}/{len(daily_sent)}')

# 1c. 新闻情绪 vs 次日指数收益
idx_ret = con.execute("""
    SELECT trade_date, close/LAG(close) OVER(ORDER BY trade_date) - 1 AS ret_1d
    FROM kline_daily WHERE ts_code='sh000300'
    AND trade_date >= '2026-04-01'
""").df()

sent_vs_ret = daily_sent.merge(idx_ret, on='trade_date', how='inner')
if len(sent_vs_ret) > 10:
    ic_nlp, p_nlp = stats.spearmanr(sent_vs_ret['sent_mean'], sent_vs_ret['ret_1d'])
    print(f'\n[1c] 市场情绪 vs 次日沪深300收益: IC={ic_nlp:+.4f}, p={p_nlp:.4f}')

    # 极端情绪日表现
    sent_vs_ret['sent_extreme'] = pd.cut(sent_vs_ret['sent_mean'],
                                          bins=[-1, -0.3, 0.3, 1],
                                          labels=['空头', '中性', '多头'])
    for label in ['多头', '中性', '空头']:
        sub = sent_vs_ret[sent_vs_ret['sent_extreme'] == label]
        if len(sub) > 0:
            print(f'  {label}日({len(sub)}天): 次日平均收益={sub.ret_1d.mean():+.3%}')
else:
    print(f'\n[1c] 数据不足({len(sent_vs_ret)}天), 无法计算IC')
    ic_nlp = 0

# 1d. 新闻太少(2个月), 无法接入回测 — 结论: 需要补爬历史新闻
print(f'\n[1d] 结论: 新闻仅{len(news)}条({news.publish_date.min()}~{news.publish_date.max()})')
print(f'  覆盖{daily_sent.trade_date.nunique()}个交易日, 不足以接入回测')
print(f'  短期情绪IC={ic_nlp:+.4f} (需>0.02才有意义)')
print(f'  → 待办: 补爬2023-2026历史新闻 (akshare/news_feed或自建爬虫)')

nlp_ready = len(news) > 10000  # 至少1万条才能接入回测
print(f'  Phase 1 状态: {"✅ 可接入" if nlp_ready else "⏸ 需补数据"}')

# ============================================================
# Phase 2: 跨资产传导
# ============================================================
print('\n' + '=' * 80)
print('Phase 2: 跨资产传导测试')
print('=' * 80)

# 2a. 取宏观指标
macro = con.execute("""
    SELECT trade_date, wti, copper, us10y, usdcny, vix, gold, china_10y
    FROM macro_indicators
    WHERE trade_date >= '2016-01-01'
    ORDER BY trade_date
""").df()

# 计算动量和变化率
macro['wti_20d'] = macro['wti'].pct_change(20)
macro['copper_20d'] = macro['copper'].pct_change(20)
macro['us10y_20d_chg'] = macro['us10y'].diff(20)
macro['vix_20d_chg'] = macro['vix'].diff(20)
macro['gold_20d'] = macro['gold'].pct_change(20)
macro['dxy_proxy'] = macro['usdcny'].pct_change(20)  # 人民币走弱=美元走强

# 铜金比 (风险偏好指标)
macro['copper_gold'] = macro['copper'] / macro['gold']
macro['copper_gold_20d'] = macro['copper_gold'].pct_change(20)

# WTI/铜联动 (需求冲击 vs 供给冲击)
macro['wti_copper_ratio'] = macro['wti'] / macro['copper']
macro['wti_copper_20d'] = macro['wti_copper_ratio'].pct_change(20)

print(f'\n[2a] 宏观数据: {len(macro)}天, {macro.trade_date.min()} ~ {macro.trade_date.max()}')
print(f'  WTI覆盖: {macro.wti.notna().sum()}天')
print(f'  铜覆盖:   {macro.copper.notna().sum()}天')
print(f'  VIX覆盖:  {macro.vix.notna().sum()}天')

# 2b. 构建传导特征: 计算每个指标对申万行业的滞后相关性
# 简化: 用全市场等权收益代替行业, 先验证宏观→全市场传导有效
mkt_ret = con.execute("""
    SELECT trade_date,
           close / LAG(close) OVER(ORDER BY trade_date) - 1 AS mkt_ret
    FROM kline_daily
    WHERE ts_code = 'sh000300' AND trade_date >= '2016-01-01'
    ORDER BY trade_date
""").df()

macro_ret = macro.merge(mkt_ret, on='trade_date', how='inner')
macro_ret['fwd_5d'] = macro_ret['mkt_ret'].rolling(5).sum().shift(-5)
macro_ret['fwd_20d'] = macro_ret['mkt_ret'].rolling(20).sum().shift(-20)

# 各宏观因子的领先IC
print(f'\n[2b] 宏观因子 → 未来20日市场收益 IC:')
conduction_ics = {}
for col in ['wti_20d', 'copper_20d', 'us10y_20d_chg', 'vix_20d_chg',
            'gold_20d', 'copper_gold_20d', 'wti_copper_20d']:
    valid = macro_ret[[col, 'fwd_20d']].dropna()
    if len(valid) > 100:
        ic, p = stats.spearmanr(valid[col], valid['fwd_20d'])
        conduction_ics[col] = ic
        stars = '***' if abs(ic) > 0.1 else ('**' if abs(ic) > 0.05 else ('*' if abs(ic) > 0.02 else ''))
        print(f'  {col:<22s} IC={ic:+.4f} p={p:.4f} {stars}')
    else:
        print(f'  {col:<22s} 数据不足({len(valid)}行)')

# 2c. 构建4个传导特征并入因子表
# 用macro数据构建每日传导分数
conduction_feats = macro[['trade_date', 'wti_20d', 'copper_20d', 'us10y_20d_chg',
                          'vix_20d_chg', 'copper_gold_20d']].copy()
conduction_feats['trade_date'] = conduction_feats['trade_date'].astype(str)

# 合并到因子表
factors['trade_date'] = factors['trade_date'].astype(str)
conduction_feats['trade_date'] = conduction_feats['trade_date'].astype(str)
target['trade_date'] = target['trade_date'].astype(str)
factors_aug = factors.merge(conduction_feats, on='trade_date', how='left')

# 2d. 测试: baseline 24因子 vs baseline+传导特征
train_end = '2023-12-31'
test_start = '2024-01-01'

base_feats = [c for c in factors.columns if c not in
              ('ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date')]

# 加上传导特征
extra_feats = ['wti_20d', 'copper_20d', 'us10y_20d_chg', 'vix_20d_chg', 'copper_gold_20d']
all_feats = base_feats + [f for f in extra_feats if f in factors_aug.columns]

# 合并target
merged = factors_aug.merge(target, on=['ts_code', 'trade_date'], how='inner')

train = merged[merged['trade_date'] <= train_end]
test = merged[merged['trade_date'] >= test_start]

print(f'\n[2d] 训练/测试划分: train={len(train)}, test={len(test)}')

# 仅用base feats
valid_base = [c for c in base_feats if c in merged.columns and c != 'excess_ret']
X_tr_b = train[valid_base].fillna(train[valid_base].median())
y_tr = train['excess_ret']
X_te_b = test[valid_base].fillna(train[valid_base].median())
y_te = test['excess_ret']

m_base = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                        subsample=0.8, colsample_bytree=0.8,
                        n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
m_base.fit(X_tr_b, y_tr)
pred_base = m_base.predict(X_te_b)
mask = ~np.isnan(pred_base) & ~np.isnan(y_te.values)
ic_base, _ = stats.spearmanr(pred_base[mask], y_te.values[mask])

# 用base+传导feats
valid_all = [c for c in all_feats if c in merged.columns]
X_tr_a = train[valid_all].fillna(train[valid_all].median())
X_te_a = test[valid_all].fillna(train[valid_all].median())

m_aug = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                       subsample=0.8, colsample_bytree=0.8,
                       n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
m_aug.fit(X_tr_a, y_tr)
pred_aug = m_aug.predict(X_te_a)
ic_aug, _ = stats.spearmanr(pred_aug[mask], y_te.values[mask])

# 按月分组回测比较
def monthly_backtest(df, pred_col, n_top=30):
    df = df.copy()
    df['ym'] = pd.to_datetime(df['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in df.groupby('ym'):
        if len(g) < 30:
            continue
        top = g.nlargest(n_top, pred_col)
        mrets.append(top['excess_ret'].mean())
    if len(mrets) < 3:
        return {'sharpe': 0, 'mdd': 0, 'mean_ret': 0, 'n_months': len(mrets)}
    rets = np.array(mrets)
    ann = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12)
    sh = ann / vol if vol > 0 else 0
    mdd = np.min(np.cumprod(1+rets) / np.maximum.accumulate(np.cumprod(1+rets)) - 1)
    return {'sharpe': sh, 'mdd': mdd, 'mean_ret': np.mean(rets), 'n_months': len(rets)}

test['pred_base'] = pred_base
test['pred_aug'] = pred_aug
perf_base = monthly_backtest(test, 'pred_base')
perf_aug = monthly_backtest(test, 'pred_aug')

print(f'\n{"":25s} {"IC":>8s} {"Sharpe":>8s} {"MDD":>8s} {"月均超额":>8s} {"月数":>6s}')
print(f'  {"Baseline(24因子)":25s} {ic_base:+.4f} {perf_base["sharpe"]:8.3f} {perf_base["mdd"]:+7.1%} {perf_base["mean_ret"]:+7.3%} {perf_base["n_months"]:6d}')
print(f'  {"+传导(29因子)":25s} {ic_aug:+.4f} {perf_aug["sharpe"]:8.3f} {perf_aug["mdd"]:+7.1%} {perf_aug["mean_ret"]:+7.3%} {perf_aug["n_months"]:6d}')

ic_gain = ic_aug - ic_base
sh_gain = perf_aug['sharpe'] - perf_base['sharpe']
print(f'  {"Δ":25s} {ic_gain:+.4f} {sh_gain:+.3f} --')

if ic_gain > 0.01:
    print(f'  ✅ Phase 2有效, IC提升{ic_gain:+.4f}, 建议接入基线')
elif ic_gain > 0:
    print(f'  ⚠ Phase 2微弱正, IC仅+{ic_gain:.4f}, 需更长测试周期')
else:
    print(f'  ❌ Phase 2无效或反向, IC={ic_gain:+.4f}')

# ============================================================
# Phase 3: 因子失效检测
# ============================================================
print('\n' + '=' * 80)
print('Phase 3: 因子失效检测 (滚动IC动态权重)')
print('=' * 80)

# 3a. 计算每个因子过去60天滚动IC
merged['trade_date_dt'] = pd.to_datetime(merged['trade_date'])
dates_sorted = sorted(merged['trade_date_dt'].unique())

print(f'\n[3a] 计算{len(valid_base)}个因子滚动IC...')
factor_names = valid_base[:12]  # 用前12个主要因子演示

# 最近60天各因子IC
recent_start = pd.Timestamp('2026-03-01')
recent = merged[merged['trade_date_dt'] >= recent_start]
print(f'  最近60天 ({recent.trade_date_dt.min().date()} ~ {recent.trade_date_dt.max().date()})')
print(f'  {"因子":<20s} {"IC_60d":>8s} {"方向":>6s} {"状态":>10s}')
print(f'  {"-"*20} {"-"*8} {"-"*6} {"-"*10}')

decaying = []
strong = []
for fn in factor_names:
    if fn not in recent.columns:
        continue
    valid = recent[[fn, 'excess_ret']].dropna()
    if len(valid) < 100:
        continue
    ic, p = stats.spearmanr(valid[fn], valid['excess_ret'])
    status = '✅有效' if abs(ic) > 0.02 else ('⚠衰减' if abs(ic) > 0.01 else '❌失效')
    if abs(ic) < 0.01:
        decaying.append((fn, ic))
    elif abs(ic) > 0.03:
        strong.append((fn, ic))
    print(f'  {fn:<20s} {ic:+.4f} {"多头" if ic>0 else "空头":>6s} {status:>10s}')

# 3b. 模拟动态权重效果
# 规则: IC<0 → 权重减半, IC连续负 → 因子冻结
print(f'\n[3b] 动态权重回测 (2024全年)')
print(f'  失效因子: {len(decaying)}个 → 权重减半')
print(f'  强势因子: {len(strong)}个 → 保持全权')

# 简化: 用2024数据, baseline vs 动态权重
test24 = merged[(merged['trade_date'] >= '2024-01-01') & (merged['trade_date'] <= '2024-12-31')]

# 先算每个因子在训练期(2022-2023)的IC, 决定权重
train_for_w = merged[(merged['trade_date'] >= '2022-01-01') & (merged['trade_date'] <= '2023-12-31')]
factor_weights = {}
for fn in valid_base[:15]:
    if fn not in train_for_w.columns:
        continue
    valid = train_for_w[[fn, 'excess_ret']].dropna()
    if len(valid) < 100:
        factor_weights[fn] = 1.0
        continue
    ic, _ = stats.spearmanr(valid[fn], valid['excess_ret'])
    if abs(ic) < 0.005:
        factor_weights[fn] = 0.0  # 冻结
    elif ic < 0:
        factor_weights[fn] = 0.5  # 减半
    else:
        factor_weights[fn] = 1.0  # 全权

# 用权重缩放特征
valid_fw = [f for f in valid_base if f in factor_weights and f in test24.columns]
X_te_b24 = test24[valid_fw].fillna(train[valid_fw].median())

# Baseline预测 (等权特征)
m24 = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                     subsample=0.8, colsample_bytree=0.8,
                     n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
X_tr_b24 = train[valid_fw].fillna(train[valid_fw].median())
m24.fit(X_tr_b24, y_tr)
pred_b24 = m24.predict(X_te_b24)

# 动态权重: 缩放特征后再训练
X_tr_w = X_tr_b24.copy()
X_te_w = X_te_b24.copy()
for fn, w in factor_weights.items():
    if fn in X_tr_w.columns:
        X_tr_w[fn] = X_tr_w[fn] * w
        X_te_w[fn] = X_te_w[fn] * w

m_dw = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                      subsample=0.8, colsample_bytree=0.8,
                      n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
m_dw.fit(X_tr_w, y_tr)
pred_dw = m_dw.predict(X_te_w)

ic_b24, _ = stats.spearmanr(pred_b24[~np.isnan(pred_b24)],
                             test24['excess_ret'].values[~np.isnan(pred_b24)])
ic_dw, _ = stats.spearmanr(pred_dw[~np.isnan(pred_dw)],
                            test24['excess_ret'].values[~np.isnan(pred_dw)])

test24_b = test24.copy()
test24_b['pred_b'] = pred_b24
test24_b['pred_dw'] = pred_dw
perf_b24 = monthly_backtest(test24_b, 'pred_b')
perf_dw = monthly_backtest(test24_b, 'pred_dw')

print(f'\n  {"":25s} {"IC":>8s} {"Sharpe":>8s} {"MDD":>8s} {"月均超额":>8s}')
print(f'  {"Baseline(等权因子)":25s} {ic_b24:+.4f} {perf_b24["sharpe"]:8.3f} {perf_b24["mdd"]:+7.1%} {perf_b24["mean_ret"]:+7.3%}')
print(f'  {"+动态权重":25s} {ic_dw:+.4f} {perf_dw["sharpe"]:8.3f} {perf_dw["mdd"]:+7.1%} {perf_dw["mean_ret"]:+7.3%}')

mdd_improve = perf_dw['mdd'] - perf_b24['mdd']
if mdd_improve < 0:  # MDD降低=改善
    print(f'  ✅ Phase 3有效, MDD降低{mdd_improve:+.1%}')
else:
    print(f'  ⚠ Phase 3 MDD未改善({mdd_improve:+.1%}), 需更长测试')

# ============================================================
# 汇总
# ============================================================
print('\n' + '=' * 80)
print('三阶段测试汇总')
print('=' * 80)

results = [
    ('Phase 1 NLP情绪', ic_nlp if 'ic_nlp' in dir() else 0, '⏸ 需补爬2023-2026历史新闻'),
    ('Phase 2 跨资产传导', ic_gain, f'{"✅" if ic_gain > 0.01 else "⚠" if ic_gain > 0 else "❌"} IC增益{ic_gain:+.4f}'),
    ('Phase 3 因子动态权重', mdd_improve if 'mdd_improve' in dir() else 0, f'{"✅" if mdd_improve < 0 else "⚠"} MDD变化{mdd_improve:+.1%}'),
]

for name, metric, verdict in results:
    print(f'  {name:<25s} {verdict}')

print('\n执行顺序建议:')
print('  1. Phase 2(传导): 数据现成, 即刻接入 → 新基线 vFinal+')
print('  2. Phase 3(动态权重): 数据现成, 接入 → vFinal++')
print('  3. Phase 1(NLP): 需补爬新闻 → 补完再测')

con.close()
print('\nDone.')
