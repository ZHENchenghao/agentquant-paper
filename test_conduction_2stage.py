# -*- coding: utf-8 -*-
"""
Phase 2 重做: 两层传导模型
Stage 1: 宏观因子 → 预测行业未来20日超额 → 选Top5行业
Stage 2: 股票因子 → 在选中行业内选Top30股票
对比: 单层(全市场选股) vs 两层(先行业后选股)
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
print('Phase 2 两层传导模型测试')
print('=' * 80)

# ============================================================
# 0. 数据准备
# ============================================================
print('\n[0] Loading...')
con = duckdb.connect(DB, read_only=True)

# 行业映射
industry = con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name,
               ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map
    ) WHERE rn = 1
""").df()

# Target
target = con.execute("""
    SELECT s.ts_code, s.trade_date::VARCHAR AS trade_date,
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

# 宏观因子
macro = con.execute("""
    SELECT trade_date::VARCHAR AS trade_date,
           wti, copper, gold, us10y, vix,
           wti / LAG(wti, 20) OVER w - 1 AS wti_20d,
           copper / LAG(copper, 20) OVER w - 1 AS copper_20d,
           gold / LAG(gold, 20) OVER w - 1 AS gold_20d,
           us10y - LAG(us10y, 20) OVER w AS us10y_chg,
           vix - LAG(vix, 20) OVER w AS vix_chg
    FROM macro_indicators
    WHERE trade_date >= '2016-01-01'
    WINDOW w AS (ORDER BY trade_date)
""").df()

con.close()

macro_cols = ['wti', 'copper', 'gold', 'vix', 'us10y']
# 填充NaN: 前向填充 (先填再合并)
for mc in macro_cols:
    macro[mc] = macro[mc].ffill().bfill()

# 合并
factors = pd.read_parquet(CACHE)
factors['trade_date'] = factors['trade_date'].astype(str)
target['trade_date'] = target['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df['ind_name'] = df['ind_name'].fillna('其他')
df = df.merge(macro, on='trade_date', how='left')

print('  Merged: %d rows, %d stocks, %d industries' % (
    len(df), df.ts_code.nunique(), df.ind_name.nunique()))

# ============================================================
# 1. 训练单层模型 (Baseline)
# ============================================================
print('\n[1] Single-stage baseline...')

exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret', 'ind_name'] + macro_cols + ['wti', 'copper', 'gold', 'us10y', 'vix']
stock_feats = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
print('  Stock features: %d' % len(stock_feats))

# ============================================================
# Stage 1: 宏观 → 行业方向预测
# ============================================================
print('\n[2] Stage 1: Macro -> Industry direction...')

# 每个行业每日平均超额收益
ind_daily = df.groupby(['trade_date', 'ind_name'])['excess_ret'].mean().reset_index()
ind_daily = ind_daily.merge(macro, on='trade_date', how='inner')

# 未来10日行业超额 (减少NaN损失)
ind_daily['fwd'] = ind_daily.groupby('ind_name')['excess_ret'].transform(
    lambda x: x.rolling(10, min_periods=5).sum().shift(-10))

# 训练: 宏观因子 → 行业方向 (所有行业共用模型)
# 每个行业单独建模 (因为不同行业对宏观敏感度不同)
top_inds = df.groupby('ind_name').size().nlargest(15).index.tolist()
print('  Top15 industries: %s' % ', '.join(top_inds[:8]) + '...')

# 行业宏观敏感度模型
ind_models = {}
ind_sector_ic = {}

for ind in top_inds:
    sub = ind_daily[ind_daily['ind_name'] == ind].copy()
    valid = sub[macro_cols + ['fwd']].dropna()
    if len(valid) < 200:
        continue

    # 简单线性: 宏观因子对行业方向的IC
    best_ic = 0
    best_macro = ''
    for mc in macro_cols:
        ic, p = stats.spearmanr(valid[mc], valid['fwd'])
        if abs(ic) > abs(best_ic):
            best_ic = ic
            best_macro = mc

    ind_sector_ic[ind] = {'best_macro': best_macro, 'best_ic': best_ic}

    # 训练LightGBM预测行业方向
    X = valid[macro_cols].values
    y = valid['fwd'].values
    y_binary = (y > 0).astype(int)  # 方向预测

    if len(set(y_binary)) < 2:
        continue

    m = LGBMRegressor(n_estimators=50, max_depth=3, num_leaves=7,
                      verbose=-1, random_state=42)
    m.fit(X, y)
    ind_models[ind] = m

print('  Industries with models: %d/%d' % (len(ind_models), len(top_inds)))
print('\n  Sector macro sensitivity:')
for ind, info in sorted(ind_sector_ic.items(), key=lambda x: -abs(x[1]['best_ic']))[:8]:
    print('    %-12s %-12s IC=%+.4f' % (ind, info['best_macro'], info['best_ic']))

# ============================================================
# 2. 两层模型: 先选行业, 行业内选股
# ============================================================
print('\n[3] Two-stage: sector filter -> stock selection...')

def evaluate_stage2(train_start, train_end, test_start, test_end, label=''):
    """两层模型评估"""
    tr = df[(df['trade_date'] >= train_start) & (df['trade_date'] <= train_end)]
    te = df[(df['trade_date'] >= test_start) & (df['trade_date'] <= test_end)]

    if len(tr) < 5000 or len(te) < 1000:
        return None

    # === Stage 1: 宏观预测行业方向 ===
    # 用训练期数据refit行业模型
    tr_ind = ind_daily[(ind_daily['trade_date'] >= train_start) & (ind_daily['trade_date'] <= train_end)]

    # === Baseline: 全市场选股 ===
    feats = [f for f in stock_feats if f in tr.columns]
    X_tr = tr[feats].fillna(tr[feats].median())
    y_tr = tr['excess_ret'].fillna(0)
    X_te = te[feats].fillna(tr[feats].median())
    y_te = te['excess_ret'].fillna(0)

    m_bl = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                          subsample=0.8, colsample_bytree=0.8,
                          n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m_bl.fit(X_tr, y_tr)
    te_bl = te.copy()
    te_bl['pred'] = m_bl.predict(X_te)

    # === Two-stage ===
    # Stage 1: 每月用宏观信号选Top5行业
    # 训练行业模型 (用训练期的行业数据)
    ind_models_te = {}
    for ind in top_inds:
        sub = tr_ind[tr_ind['ind_name'] == ind]
        valid = sub[macro_cols + ['fwd']].dropna()
        if len(valid) < 100:
            continue
        X = valid[macro_cols].values
        y = valid['fwd'].values
        if len(set(y > 0)) < 2:
            continue
        m = LGBMRegressor(n_estimators=50, max_depth=3, num_leaves=7,
                          verbose=-1, random_state=42)
        m.fit(X, y)
        ind_models_te[ind] = m

    # Stage 2: 每月选行业, 行业内选股
    te_2s = te.copy()
    te_2s['pred_2s'] = 0.0

    # 按月处理
    te_2s['ym'] = pd.to_datetime(te_2s['trade_date']).dt.to_period('M')
    for mo, g in te_2s.groupby('ym'):
        if len(g) < 500:
            continue

        # 该月宏观特征 (取月内平均值)
        mo_macro = g[macro_cols].mean().values.reshape(1, -1)

        # 预测每个行业方向
        ind_scores = {}
        for ind, m in ind_models_te.items():
            try:
                score = m.predict(mo_macro)[0]
                ind_scores[ind] = score
            except:
                pass

        if not ind_scores:
            continue

        # 选Top5行业
        top5_inds = sorted(ind_scores, key=ind_scores.get, reverse=True)[:5]

        # 在Top5行业内选股 (用基础股票因子)
        g_selected = g[g['ind_name'].isin(top5_inds)]

        if len(g_selected) < 30:
            g_selected = g  # fallback to all

        # 训练股票模型 (用训练期数据)
        tr_stock = tr[tr['ind_name'].isin(top5_inds)]
        if len(tr_stock) < 1000:
            tr_stock = tr

        X_tr_s = tr_stock[feats].fillna(tr_stock[feats].median())
        y_tr_s = tr_stock['excess_ret'].fillna(0)
        X_te_s = g_selected[feats].fillna(tr_stock[feats].median())

        m_s = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                             subsample=0.8, colsample_bytree=0.8,
                             n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
        m_s.fit(X_tr_s, y_tr_s)
        g_pred = m_s.predict(X_te_s)

        # 写入预测值
        g_idx = g_selected.index
        for j, idx in enumerate(g_idx):
            if j < len(g_pred):
                te_2s.loc[idx, 'pred_2s'] = g_pred[j]

    # === 评估 ===
    # Baseline
    te_bl2 = te_bl.copy()
    te_bl2['ym'] = pd.to_datetime(te_bl2['trade_date']).dt.to_period('M')
    mrets_bl = []
    for mo, g in te_bl2.groupby('ym'):
        if len(g) < 30:
            continue
        top = g.nlargest(30, 'pred')
        mrets_bl.append(top['excess_ret'].mean())

    # Two-stage
    mrets_2s = []
    for mo, g in te_2s.groupby('ym'):
        if len(g) < 30:
            continue
        top = g.nlargest(30, 'pred_2s')
        mrets_2s.append(top['excess_ret'].mean())

    def calc_metrics(rets):
        if len(rets) < 3:
            return {'ic': 0, 'sh': 0, 'mdd': 0, 'mr': 0}
        rets = np.array(rets)
        ann = np.mean(rets) * 12
        vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets) > 2 else 0.01
        sh = ann / vol if vol > 0 else 0
        mdd = np.min(np.cumprod(1+rets) / np.maximum.accumulate(np.cumprod(1+rets)) - 1)
        return {'sh': sh, 'mdd': mdd, 'mr': np.mean(rets)}

    m_bl = calc_metrics(mrets_bl)
    m_2s = calc_metrics(mrets_2s)

    # IC (全量预测 vs 实际)
    mask_bl = ~np.isnan(te_bl['pred']) & ~np.isnan(te_bl['excess_ret'])
    ic_bl, _ = stats.spearmanr(te_bl.loc[mask_bl, 'pred'], te_bl.loc[mask_bl, 'excess_ret'])

    mask_2s = ~np.isnan(te_2s['pred_2s']) & ~np.isnan(te_2s['excess_ret'])
    ic_2s, _ = stats.spearmanr(te_2s.loc[mask_2s, 'pred_2s'], te_2s.loc[mask_2s, 'excess_ret'])

    return {
        'label': label,
        'bl_ic': ic_bl, 'bl_sh': m_bl['sh'], 'bl_mdd': m_bl['mdd'], 'bl_mr': m_bl['mr'],
        '2s_ic': ic_2s, '2s_sh': m_2s['sh'], '2s_mdd': m_2s['mdd'], '2s_mr': m_2s['mr'],
    }

# 测试3个窗口
# 简化: train_end连接test_start
windows = [
    ('22-23->24', '2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('21-22->23', '2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('23-24->25', '2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
]

print('\n  %-18s %8s %8s %8s | %8s %8s %8s' % (
    'Window', 'BL_IC', '2S_IC', 'Delta', 'BL_Sh', '2S_Sh', 'Win?'))
print('  ' + '-' * 75)

results = []
for label, tr_s, tr_e, te_s, te_e in windows:
    r = evaluate_stage2(tr_s, tr_e, te_s, te_e, label)
    if r is None:
        continue
    results.append(r)
    dic = r['2s_ic'] - r['bl_ic']
    winner = '2-Stage' if dic > 0 else 'Baseline'
    print('  %-18s %+.4f %+.4f %+.4f | %8.3f %8.3f %s' % (
        r['label'], r['bl_ic'], r['2s_ic'], dic, r['bl_sh'], r['2s_sh'], winner))

if results:
    avg_bl_ic = np.mean([r['bl_ic'] for r in results])
    avg_2s_ic = np.mean([r['2s_ic'] for r in results])
    avg_bl_sh = np.mean([r['bl_sh'] for r in results])
    avg_2s_sh = np.mean([r['2s_sh'] for r in results])
    print('  %-18s %+.4f %+.4f %+.4f | %8.3f %8.3f' % (
        'AVERAGE', avg_bl_ic, avg_2s_ic, avg_2s_ic-avg_bl_ic, avg_bl_sh, avg_2s_sh))

    pos = sum(1 for r in results if r['2s_ic'] > r['bl_ic'])
    print('\n  两层胜率: %d/%d' % (pos, len(results)))

# ============================================================
# 行业传导强度排名
# ============================================================
print('\n[4] 行业宏观敏感度总结:')
for ind, info in sorted(ind_sector_ic.items(), key=lambda x: -abs(x[1]['best_ic'])):
    ic = info['best_ic']
    direction = '跟随' if ic > 0 else '反向'
    print('  %-12s %-12s IC=%+.4f (%s)' % (ind, info['best_macro'], ic, direction))

print('\nDone.')
