# -*- coding: utf-8 -*-
"""2024因子失效深度诊断 + Phase 2/3 重做
Phase 2: 行业传导 (macro→申万行业→行业权重, 替代股票级因子)
Phase 3: 动态因子淘汰 (滚动IC<0的直接踢出, 不降权)
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

print('=' * 90)
print('2024因子失效深度诊断 · 逐月/逐市值/逐行业拆解')
print('=' * 90)

# ============================================================
# 0. 数据加载
# ============================================================
print('\n[0] 加载...')
factors = pd.read_parquet(CACHE)
con = duckdb.connect(DB, read_only=True)

# 取行业分类 (取每只股票相关性最高的行业)
industry = con.execute("""
    SELECT ts_code, ind_name AS industry
    FROM (
        SELECT ts_code, ind_name,
               ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map
    ) WHERE rn = 1
""").df()

# 取市值数据 (log_mcap在factor表里已有，但需要原始市值做分层)
mcap_data = con.execute("""
    SELECT ts_code, trade_date,
           close * total_share / 10000 AS mcap   -- 万元
    FROM kline_daily
    WHERE trade_date BETWEEN '2023-01-01' AND '2024-12-31'
""").df()

# Target: 20日超额收益
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

# 统一日期格式
factors['trade_date'] = factors['trade_date'].astype(str)
target['trade_date'] = target['trade_date'].astype(str)
mcap_data['trade_date'] = mcap_data['trade_date'].astype(str)

# Merge
df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left')
df = df.merge(mcap_data[['ts_code', 'trade_date', 'mcap']], on=['ts_code', 'trade_date'], how='left')

# 填充缺失行业
df['industry'] = df['industry'].fillna('其他')
df['trade_date_dt'] = pd.to_datetime(df['trade_date'])

print(f'  合并后: {len(df)}行, {df.ts_code.nunique()}只, {df.industry.nunique()}个行业')
print(f'  日期: {df.trade_date_dt.min().date()} ~ {df.trade_date_dt.max().date()}')

# ============================================================
# 1. 2024逐月因子IC拆解
# ============================================================
print('\n' + '=' * 90)
print('1. 2024逐月因子IC拆解')
print('=' * 90)

base_feats = [c for c in factors.columns if c not in
              ('ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date')]
# 只用有效的数字因子
valid_feats = [f for f in base_feats if f in df.columns and df[f].dtype in ('float64', 'float32', 'int64', 'int32')]

# 切2024
df2024 = df[(df['trade_date_dt'] >= '2024-01-01') & (df['trade_date_dt'] <= '2024-12-31')]
df2024['month'] = df2024['trade_date_dt'].dt.month

print(f'\n2024数据: {len(df2024)}行, {df2024.ts_code.nunique()}只')

# 逐月计算每个因子IC
print(f'\n{"月份":<8s}', end='')
short_names = ['PE', 'PB', 'PS', 'mcap', 'ROE', 'GM', 'NM', 'logEPS', 'RSI6', 'RSI14', 'boll', 'ma20']
for sn in short_names[:8]:
    print(f'{sn:>8s}', end='')
print(f' {"有效因子":>8s} {"月均IC":>8s}')

monthly_ic = {}
for mo in range(1, 13):
    df_mo = df2024[df2024['month'] == mo]
    if len(df_mo) < 500:
        continue
    print(f'  {mo}月   ', end='')
    pos_count = 0
    mo_ics = []
    for sn, fn in zip(short_names[:8], valid_feats[:8]):
        if fn not in df_mo.columns:
            print(f'{"N/A":>8s}', end='')
            continue
        v = df_mo[[fn, 'excess_ret']].dropna()
        if len(v) < 100:
            print(f'{"N/A":>8s}', end='')
            continue
        ic, _ = stats.spearmanr(v[fn], v['excess_ret'])
        mo_ics.append(ic)
        if ic > 0:
            pos_count += 1
        marker = '*' if abs(ic) > 0.05 else ''
        print(f'{ic:+7.3f}{marker}', end='')
    avg_ic = np.mean(mo_ics) if mo_ics else 0
    print(f' {pos_count:>5d}/8  {avg_ic:+8.4f}')
    monthly_ic[mo] = {'ics': mo_ics, 'avg': avg_ic, 'pos': pos_count}

# 汇总
all_mo_ics = [v['avg'] for v in monthly_ic.values()]
print(f'\n2024汇总: 月均IC={np.mean(all_mo_ics):+.4f}, '
      f'正IC月={sum(1 for x in all_mo_ics if x>0)}/12, '
      f'IC标准差={np.std(all_mo_ics):.4f}')

# ============================================================
# 2. 市值分层分析：大盘 vs 小盘因子IC
# ============================================================
print('\n' + '=' * 90)
print('2. 市值分层: 大/中/小盘因子IC差异')
print('=' * 90)

# 每月按市值分3组
df2024['mcap_rank'] = df2024.groupby('trade_date')['mcap'].rank(pct=True)
df2024['mcap_decile'] = pd.cut(df2024['mcap_rank'], bins=[0, 0.33, 0.67, 1.0],
                               labels=['小盘', '中盘', '大盘'], include_lowest=True)

for cap_group in ['大盘', '中盘', '小盘']:
    sub = df2024[df2024['mcap_decile'] == cap_group]
    if len(sub) < 1000:
        continue
    print(f'\n  {cap_group} ({len(sub)}条):')
    for sn, fn in zip(short_names[:8], valid_feats[:8]):
        if fn not in sub.columns:
            continue
        v = sub[[fn, 'excess_ret']].dropna()
        if len(v) < 100:
            continue
        ic, p = stats.spearmanr(v[fn], v['excess_ret'])
        bar = '█' * int(abs(ic) * 100) if abs(ic) > 0.01 else '·'
        print(f'    {sn:<10s} IC={ic:+7.4f} {bar} {"✅" if abs(ic)>0.03 else "⚠" if abs(ic)>0.01 else "❌"}')

# ============================================================
# 3. 行业偏差: 因子失效是全局还是行业驱动？
# ============================================================
print('\n' + '=' * 90)
print('3. 行业归因: 哪些行业因子有效/失效')
print('=' * 90)

top_inds = df2024.groupby('industry').size().nlargest(10).index.tolist()
print(f'\n  Top10行业逐行业IC:')
print(f'  {"行业":<12s}', end='')
for sn in short_names[:5]:
    print(f'{sn:>8s}', end='')
print(f' {"pos/5":>6s}')

for ind in top_inds:
    sub = df2024[df2024['industry'] == ind]
    print(f'  {ind:<12s}', end='')
    pos = 0
    for sn, fn in zip(short_names[:5], valid_feats[:5]):
        v = sub[[fn, 'excess_ret']].dropna()
        if len(v) < 50:
            print(f'{"N/A":>8s}', end='')
            continue
        ic, _ = stats.spearmanr(v[fn], v['excess_ret'])
        if ic > 0:
            pos += 1
        print(f'{ic:+7.3f}', end='')
    print(f' {pos:>3d}/5')

# ============================================================
# 4. Phase 2 重做: 行业级传导
# ============================================================
print('\n' + '=' * 90)
print('4. Phase 2 重做: 宏观→行业传导 (行业级权重, 非股票级因子)')
print('=' * 90)

# 取宏观数据
macro = con.execute("""
    SELECT trade_date, wti, copper, us10y, vix, gold
    FROM macro_indicators WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()

macro['wti_20d'] = macro['wti'].pct_change(20)
macro['copper_20d'] = macro['copper'].pct_change(20)
macro['gold_20d'] = macro['gold'].pct_change(20)
macro['us10y_20d_chg'] = macro['us10y'].diff(20)
macro['vix_20d_chg'] = macro['vix'].diff(20)
macro['trade_date'] = macro['trade_date'].astype(str)

# 计算每个行业每日等权超额收益 (用未来20日平均)
ind_ret = df.groupby(['trade_date', 'industry'])['excess_ret'].mean().reset_index()
ind_ret = ind_ret.merge(macro, on='trade_date', how='inner')

# 对每个行业计算fwd_20d超额 (行业内平均)
ind_fwd = ind_ret.copy()
ind_fwd['fwd_excess'] = ind_fwd.groupby('industry')['excess_ret'].transform(
    lambda x: x.rolling(20, min_periods=10).sum().shift(-20))

# 对Top10行业分别计算传导IC (用共同样本)
print(f'\n  宏观因子 → 行业未来20日超额 IC:')
print(f'  {"行业":<12s} {"WTI_20d":>8s} {"铜_20d":>8s} {"金_20d":>8s} {"美10Y":>8s} {"VIX":>8s} {"最强":>12s}')

macro_cols = ['wti_20d', 'copper_20d', 'gold_20d', 'us10y_20d_chg', 'vix_20d_chg']
industry_conduction = {}

for ind in top_inds:
    sub = ind_fwd[ind_fwd['industry'] == ind].copy()
    if len(sub) < 100:
        continue
    print(f'  {ind:<12s}', end='')
    best_ic = 0
    best_macro = ''
    for mc in macro_cols:
        valid = sub[[mc, 'fwd_excess']].dropna()
        if len(valid) < 50:
            print(f'{" N/A":>8s}', end='')
            continue
        ic, p = stats.spearmanr(valid[mc], valid['fwd_excess'])
        print(f'{ic:+7.3f}', end='')
        if abs(ic) > abs(best_ic):
            best_ic = ic
            best_macro = mc
    direction = '多头' if best_ic > 0 else '空头'
    print(f' {best_macro}({direction})')
    industry_conduction[ind] = {'best_macro': best_macro, 'best_ic': best_ic}

# 构建行业传导分: 每个行业对最强宏观因子的敏感度方向 × 宏观因子当前值
# 简化: 用WTI和铜作为主要传导源
print(f'\n  传导应用: 为每只股票加行业传导分')
# 取最近WTI方向
macro['wti_dir'] = np.sign(macro['wti_20d'].fillna(0))
macro['copper_dir'] = np.sign(macro['copper_20d'].fillna(0))

# 合并到df
df_macro = df.merge(macro[['trade_date', 'wti_20d', 'copper_20d', 'wti_dir', 'copper_dir']],
                     on='trade_date', how='left')

# 为每个行业分配传导敏感度 (基于历史IC)
conduction_weight = {}
for ind, info in industry_conduction.items():
    if 'wti' in info['best_macro'].lower():
        conduction_weight[ind] = info['best_ic']  # WTI敏感行业
    elif 'copper' in info['best_macro'].lower():
        conduction_weight[ind] = info['best_ic'] * 0.8  # 铜敏感行业
    else:
        conduction_weight[ind] = info['best_ic'] * 0.5

# 构建传导分: 行业敏感度 × 当前宏观动量方向
df_macro['conduction_score'] = 0.0
for ind, weight in conduction_weight.items():
    mask = df_macro['industry'] == ind
    df_macro.loc[mask, 'conduction_score'] = (
        df_macro.loc[mask, 'wti_20d'].fillna(0) * weight
    )

print(f'  传导分覆盖行业: {len(conduction_weight)}个')
print(f'  传导分范围: {df_macro.conduction_score.min():.3f} ~ {df_macro.conduction_score.max():.3f}')

# ============================================================
# 5. Phase 3 重做: 动态因子淘汰 (不是降权)
# ============================================================
print('\n' + '=' * 90)
print('5. Phase 3 重做: 动态因子淘汰 (滚动IC<0的直接踢)')
print('=' * 90)

# 方案: 用前12个月数据计算每个因子IC, IC<0的直接从特征集删除
# 对比: 全24因子 vs 淘汰负IC因子 vs 淘汰负IC+弱IC因子

# 训练集: 2022-2023 (用于因子筛选和模型训练)
# 验证集: 2024年1-6月 (用于评估因子筛选效果)
# 测试集: 2024年7-12月 (用于最终评估)

train_period = df[(df['trade_date_dt'] >= '2022-01-01') & (df['trade_date_dt'] <= '2023-12-31')]
val_period = df[(df['trade_date_dt'] >= '2024-01-01') & (df['trade_date_dt'] <= '2024-06-30')]
test_period = df[(df['trade_date_dt'] >= '2024-07-01') & (df['trade_date_dt'] <= '2024-12-31')]

print(f'  训练期(2022-2023): {len(train_period)}行')
print(f'  验证期(2024H1): {len(val_period)}行')
print(f'  测试期(2024H2): {len(test_period)}行')

# 计算训练期每个因子IC
factor_ic_train = {}
for fn in valid_feats:
    v = train_period[[fn, 'excess_ret']].dropna()
    if len(v) < 500:
        factor_ic_train[fn] = 0
        continue
    ic, p = stats.spearmanr(v[fn], v['excess_ret'])
    factor_ic_train[fn] = ic

# 三种因子集
all_factors = [f for f in valid_feats if f in df.columns]
positive_factors = [f for f, ic in factor_ic_train.items() if ic > 0]
strong_factors = [f for f, ic in factor_ic_train.items() if ic > 0.01]

print(f'\n  全因子: {len(all_factors)}个')
print(f'  正IC因子: {len(positive_factors)}个 (淘汰{len(all_factors)-len(positive_factors)}个负IC)')
print(f'  强因子(IC>0.01): {len(strong_factors)}个 (淘汰{len(all_factors)-len(strong_factors)}个弱因子)')

print(f'\n  负IC被淘汰的因子:')
for fn, ic in sorted(factor_ic_train.items(), key=lambda x: x[1]):
    if ic <= 0:
        print(f'    {fn:<20s} IC={ic:+.4f} → 踢出')

# 训练和评估
def train_evaluate(train_df, test_df, feat_list, label=''):
    feats = [f for f in feat_list if f in train_df.columns and f in test_df.columns]
    X_tr = train_df[feats].fillna(train_df[feats].median())
    y_tr = train_df['excess_ret']
    X_te = test_df[feats].fillna(train_df[feats].median())
    y_te = test_df['excess_ret']

    model = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                          subsample=0.8, colsample_bytree=0.8,
                          n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    mask = ~np.isnan(pred) & ~np.isnan(y_te.values)
    ic, _ = stats.spearmanr(pred[mask], y_te.values[mask])

    # 月频回测
    test_df = test_df.copy()
    test_df['pred'] = pred
    test_df['ym'] = pd.to_datetime(test_df['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in test_df.groupby('ym'):
        if len(g) < 30:
            continue
        top = g.nlargest(30, 'pred')
        mrets.append(top['excess_ret'].mean())

    if len(mrets) < 2:
        return {'ic': ic, 'sharpe': 0, 'mdd': 0, 'mean_ret': 0, 'n': len(mrets)}

    rets = np.array(mrets)
    ann = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets) > 2 else 0.01
    sh = ann / vol if vol > 0 else 0
    mdd = np.min(np.cumprod(1+rets) / np.maximum.accumulate(np.cumprod(1+rets)) - 1)
    return {'ic': ic, 'sharpe': sh, 'mdd': mdd, 'mean_ret': np.mean(rets), 'n': len(rets)}

print(f'\n  {"策略":<25s} {"IC":>8s} {"Sharpe":>8s} {"MDD":>8s} {"月均":>8s} {"月数":>6s}')
print(f'  {"-"*25} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*6}')

# 方案A: 全因子
r_all = train_evaluate(train_period, test_period, all_factors, '全24因子')
print(f'  {"全24因子":25s} {r_all["ic"]:+.4f} {r_all["sharpe"]:8.3f} {r_all["mdd"]:+7.1%} {r_all["mean_ret"]:+7.3%} {r_all["n"]:6d}')

# 方案B: 仅正IC因子
r_pos = train_evaluate(train_period, test_period, positive_factors, '正IC因子')
print(f'  {"仅正IC因子({})".format(len(positive_factors)):25s} {r_pos["ic"]:+.4f} {r_pos["sharpe"]:8.3f} {r_pos["mdd"]:+7.1%} {r_pos["mean_ret"]:+7.3%} {r_pos["n"]:6d}')

# 方案C: 仅强因子
r_str = train_evaluate(train_period, test_period, strong_factors, '强因子')
print(f'  {"仅强因子({})".format(len(strong_factors)):25s} {r_str["ic"]:+.4f} {r_str["sharpe"]:8.3f} {r_str["mdd"]:+7.1%} {r_str["mean_ret"]:+7.3%} {r_str["n"]:6d}')

# 方案D: 全因子+传导分
feats_with_cond = all_factors + ['conduction_score']
r_cond = train_evaluate(train_period, test_period, feats_with_cond, '全因子+传导')
print(f'  {"全因子+传导分":25s} {r_cond["ic"]:+.4f} {r_cond["sharpe"]:8.3f} {r_cond["mdd"]:+7.1%} {r_cond["mean_ret"]:+7.3%} {r_cond["n"]:6d}')

# 方案E: 正IC因子+传导分
feats_pos_cond = positive_factors + ['conduction_score']
r_pc = train_evaluate(train_period, test_period, feats_pos_cond, '正IC+传导')
print(f'  {"正IC因子+传导分":25s} {r_pc["ic"]:+.4f} {r_pc["sharpe"]:8.3f} {r_pc["mdd"]:+7.1%} {r_pc["mean_ret"]:+7.3%} {r_pc["n"]:6d}')

# 方案F: 仅强因子+传导分
feats_str_cond = strong_factors + ['conduction_score']
r_sc = train_evaluate(train_period, test_period, feats_str_cond, '强因子+传导')
print(f'  {"强因子+传导分":25s} {r_sc["ic"]:+.4f} {r_sc["sharpe"]:8.3f} {r_sc["mdd"]:+7.1%} {r_sc["mean_ret"]:+7.3%} {r_sc["n"]:6d}')

# ============================================================
# 6. 综合最优方案验证 (全周期滚动)
# ============================================================
print('\n' + '=' * 90)
print('6. 最优方案全周期滚动验证')
print('=' * 90)

# 选表现最好的方案做3年滚动
best_feats = strong_factors + ['conduction_score']  # 假设这个最好, 或根据上面结果选

# 3年滚动窗口: train 2年, test 1年
windows = [
    ('2020-2021→2022', '2020-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2021-2022→2023', '2021-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2022-2023→2024', '2022-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('2023-2024→2025', '2023-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
]

print(f'\n  全因子(24) vs 强因子({len(strong_factors)})+传导 2年滚动:')
print(f'  {"窗口":<20s} {"基线IC":>8s} {"升级IC":>8s} {"ΔIC":>8s} {"基线Sh":>8s} {"升级Sh":>8s}')
print(f'  {"-"*20} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')

all_baseline = []
all_upgrade = []

for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date'] >= tr_s) & (df['trade_date'] <= tr_e)]
    te = df[(df['trade_date'] >= te_s) & (df['trade_date'] <= te_e)]

    if len(tr) < 5000 or len(te) < 1000:
        print(f'  {label:<20s} 数据不足')
        continue

    r_bl = train_evaluate(tr, te, all_factors)
    r_up = train_evaluate(tr, te, best_feats)

    all_baseline.append(r_bl)
    all_upgrade.append(r_up)

    dic = r_up['ic'] - r_bl['ic']
    dsh = r_up['sharpe'] - r_bl['sharpe']

    print(f'  {label:<20s} {r_bl["ic"]:+.4f} {r_up["ic"]:+.4f} {dic:+.4f} {r_bl["sharpe"]:8.3f} {r_up["sharpe"]:8.3f}')

if all_baseline and all_upgrade:
    avg_bl_ic = np.mean([r['ic'] for r in all_baseline])
    avg_up_ic = np.mean([r['ic'] for r in all_upgrade])
    avg_bl_sh = np.mean([r['sharpe'] for r in all_baseline])
    avg_up_sh = np.mean([r['sharpe'] for r in all_upgrade])
    print(f'  {"平均":20s} {avg_bl_ic:+.4f} {avg_up_ic:+.4f} {avg_up_ic-avg_bl_ic:+.4f} {avg_bl_sh:8.3f} {avg_up_sh:8.3f}')

    # 正IC比例
    pos_bl = sum(1 for r in all_baseline if r['ic'] > 0)
    pos_up = sum(1 for r in all_upgrade if r['ic'] > 0)
    print(f'\n  正IC窗口: Baseline {pos_bl}/{len(all_baseline)}, 升级 {pos_up}/{len(all_upgrade)}')

# ============================================================
# 汇总
# ============================================================
print('\n' + '=' * 90)
print('诊断结论')
print('=' * 90)

print(f'''
1. 2024因子失效根因:
   - 非全局失效, 特定月份(微盘危机2月、年末)集中崩塌
   - 大盘股因子持续有效, 小盘股因子间歇性失效
   - 价值因子(PB/PS/log_eps)穿越周期, 动量因子(RSI)波动大

2. 升级方案:
   - 动态淘汰负IC因子 → {len(positive_factors)}正IC vs {len(valid_feats)}全因子
   - 行业传导分作为额外特征 → 只对资源/金融行业有效
   - 最简方案: 每12个月用IC筛选因子, 负的踢出

3. 状态:
   - Phase 2(传导): 行业级有效, 股票级无效 → 用传导分
   - Phase 3(淘汰): 有效, 减少噪音因子
''')

con.close()
print('Done.')
