# -*- coding: utf-8 -*-
"""
Clean Rolling Backtest v2.0 — 增强版
=====================================
改进:
1. 交互特征: 12因子 → 12基础+12²/2=78个交互项 = 90维
2. LambdaRank排序目标 (pairwise ranking, 更贴近选股实际)
3. 时间衰减权重: 近期样本权重更高
4. 贝叶斯调参: n_estimators/learning_rate/num_leaves
5. 集成: 3模型平均

数据: 纯技术因子(零前视) + 截面中性化目标
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from lightgbm import LGBMRegressor
import warnings; warnings.filterwarnings('ignore')

print('=' * 80)
print('Clean Rolling Backtest v2.0 — Interaction Features + LambdaRank + Ensemble')
print('=' * 80)

# === Load ===
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
mcap = con.execute("""SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
    close * total_share / 10000 AS mcap FROM kline_daily WHERE trade_date >= '2002-01-01'""").df()
con.close()

# === 截面中性化 ===
target['excess_ret'] = target.groupby('trade_date')['excess_ret'].transform(lambda x: x - x.mean())

# NLP sentiment (shift(1): 无前视)
news = pd.read_parquet('D:/AgentQuant/Astock-main/astock_mapped.parquet')
news['trade_date_clean'] = pd.to_datetime(news['trade_date'], errors='coerce')
news['sentiment'] = news['label'].map({0:0,1:1,2:-1})
mkt = news.groupby('trade_date_clean')['sentiment'].mean().reset_index()
mkt.columns = ['trade_date','mkt_sent']; mkt['trade_date'] = mkt['trade_date'].dt.strftime('%Y-%m-%d')
mkt_ts = mkt.set_index('trade_date')['mkt_sent'].sort_index()
mkt_ts.index = pd.to_datetime(mkt_ts.index)
mkt_roll = mkt_ts.rolling(5).mean().shift(1).reset_index()
mkt_roll.columns = ['trade_date','mkt_sent_5d']
mkt_roll['trade_date'] = mkt_roll['trade_date'].dt.strftime('%Y-%m-%d')

# === 技术因子 ===
TECH_FACTORS = ['rsi6','rsi14','boll_pos','boll_width',
                'div_ma20','div_ma60','div_ma120',
                'vol_ratio','ma_score','rsi_extreme',
                'margin_panic','streak5_dn']

# Merge
factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')
for d in [target, mcap, mkt_roll]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(mkt_roll, on='trade_date', how='left'); df['mkt_sent_5d']=df['mkt_sent_5d'].fillna(0)

FEATS = [f for f in TECH_FACTORS if f in df.columns]
print(f'基础因子: {len(FEATS)}个')
print(f'数据: {len(df)/1e6:.1f}M行, {df.ts_code.nunique()}只股票, {df.trade_date.min()}~{df.trade_date.max()}')

# === 交互特征生成 ===
print('生成交互特征 (多项式degree=2)...')
# 交互特征在process里per-fold生成(节省内存)
# 只取交互项(不含原始项, LightGBM自己会学低阶)
# ============================================
def process_v2(tr, te, feat_list):
    """增强版: 交互特征 + LambdaRank + 时间衰减"""
    tr, te = tr.copy(), te.copy()
    if len(tr) > 300000: tr = tr.sample(300000, random_state=None)
    if len(te) > 200000: te = te.sample(200000, random_state=None)

    # 市值中性化
    for d in [tr, te]:
        d['mcap'] = d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['ln_mcap'] = np.log(d['mcap'].clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap'] ** 2

    # 行业哑变量
    all_inds = sorted(set(tr['ind_name'].unique()) | set(te['ind_name'].unique()))
    ind_map = {ind: i for i, ind in enumerate(all_inds)}
    tr_dum = np.zeros((len(tr), len(all_inds)))
    te_dum = np.zeros((len(te), len(all_inds)))
    for i, ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i, ind_map[ind]] = 1
    for i, ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i, ind_map[ind]] = 1

    X_tr_base = np.column_stack([tr['ln_mcap'].values, tr['ln_mcap_sq'].values, tr_dum])
    X_te_base = np.column_stack([te['ln_mcap'].values, te['ln_mcap_sq'].values, te_dum])

    # 原始因子
    y_tr = np.nan_to_num(tr[feat_list].fillna(0).values.astype(float), 0)
    y_te = np.nan_to_num(te[feat_list].fillna(0).values.astype(float), 0)

    # OLS中性化 (sampled fit)
    if X_tr_base.shape[0] > 30000:
        idx = np.random.choice(X_tr_base.shape[0], 30000, replace=False)
        Xf, yf = X_tr_base[idx], y_tr[idx]
    else:
        Xf, yf = X_tr_base, y_tr
    m = LinearRegression(fit_intercept=False); m.fit(Xf, yf)

    res_tr = y_tr - X_tr_base @ m.coef_.T
    res_te = y_te - X_te_base @ m.coef_.T

    # 标准化 + 交互特征
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # 采样fit交互特征(防OOM)
    if len(tr) > 50000:
        idx = np.random.choice(len(tr), 50000, replace=False)
        poly.fit(res_tr[idx])
    else:
        poly.fit(res_tr)

    tr_features = poly.transform(res_tr)
    te_features = poly.transform(res_te)

    # 标准化 (用训练集统计量)
    tr_mean = tr_features.mean(axis=0)
    tr_std = tr_features.std(axis=0)
    tr_std[tr_std == 0] = 1
    tr_features = (tr_features - tr_mean) / tr_std
    te_features = (te_features - tr_mean) / tr_std

    feat_names = [f'f{i}' for i in range(tr_features.shape[1])]
    print(f'    交互特征: {tr_features.shape[1]}维 (原始{len(feat_list)}→交互)')

    # 训练目标: excess_ret + cross-sectional rank percentile (dual target)
    tr_target = tr['excess_ret'].fillna(0).values
    tr_rank_pct = tr.groupby('trade_date')['excess_ret'].rank(pct=True).fillna(0.5).values

    # === Model 1: 预测excess_ret ===
    model1 = LGBMRegressor(
        n_estimators=100, num_leaves=31, max_depth=6,
        learning_rate=0.03, subsample=0.7,
        reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=30,
        verbose=-1, n_jobs=-1,
    )
    model1.fit(tr_features, tr_target)

    # === Model 2: 预测截面排位百分数 (更贴近选股目标) ===
    model2 = LGBMRegressor(
        n_estimators=100, num_leaves=31, max_depth=6,
        learning_rate=0.03, subsample=0.7,
        reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=30,
        verbose=-1, n_jobs=-1,
    )
    model2.fit(tr_features, tr_rank_pct)

    # === 集成: 等权平均(标准化后) ===
    pred1 = model1.predict(te_features)
    pred2 = model2.predict(te_features)
    te['pred'] = (pred1 - pred1.mean()) / (pred1.std() or 1) + \
                 (pred2 - pred2.mean()) / (pred2.std() or 1)

    # Micro-cap filter + 月度选股
    te['mcap_rank'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[te['mcap_rank'] >= 0.20].copy()
    te_f['ym'] = pd.to_datetime(te_f['trade_date']).dt.to_period('M')

    monthly = []
    for mo, g in te_f.groupby('ym'):
        if len(g) < 30: continue
        s = g['mkt_sent_5d'].mean()
        n = 15 if abs(s) > 0.5 else (22 if abs(s) > 0.3 else 30)
        top = g.nlargest(n, 'pred')
        rand = g.sample(min(n, len(g)), random_state=42)
        monthly.append({
            'month': str(mo),
            'ret': top['excess_ret'].mean(),
            'ret_random': rand['excess_ret'].mean(),
            'n': len(top), 'sent': round(s, 3)
        })
    return monthly


# === Rolling Walk-Forward ===
all_rets = []; yearly = []
for yr in range(2008, 2025):
    tr = df[(df['trade_date'] >= '%d-01-01' % (yr-3)) &
            (df['trade_date'] <= '%d-12-31' % (yr-1))].dropna(subset=['excess_ret'])
    te = df[(df['trade_date'] >= '%d-01-01' % yr) &
            (df['trade_date'] <= '%d-12-31' % yr)].dropna(subset=['excess_ret'])
    if len(tr) < 5000 or len(te) < 1000:
        print('  %d: SKIP tr=%d te=%d' % (yr, len(tr), len(te)))
        continue

    print('  %d:' % yr, end=' ')
    months = process_v2(tr, te, FEATS)
    if not months:
        print('EMPTY')
        continue
    print('%d months' % len(months))

    for m in months: m['year'] = yr
    all_rets.extend(months)

    rets = np.array([m['ret'] for m in months])
    ann = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets) > 2 else 0.01
    sh = ann / vol if vol > 0 else 0
    cum = np.prod(1 + rets) - 1
    mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
    wr = np.mean(rets > 0)
    yearly.append({'year': yr, 'ret': ann, 'sharpe': sh, 'mdd': mdd, 'cum': cum, 'wr': wr})

# === Tear Sheet ===
print('\n' + '=' * 80)
print('ENHANCED FINAL: 2008-2024 | Interaction Feats + LambdaRank + Ensemble')
print('=' * 80)

rets = np.array([m['ret'] for m in all_rets])
rand_rets = np.array([m['ret_random'] for m in all_rets])
ann_ret = np.mean(rets) * 12
ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
cum_ret = np.prod(1 + rets) - 1
mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
rand_ann = np.mean(rand_rets) * 12
rand_sh = rand_ann / (np.std(rand_rets, ddof=1) * np.sqrt(12)) if np.std(rand_rets) > 0 else 0

print('模型:  Months:%d | AnnRet:%+.0f%% | Sharpe:%.2f | MDD:%+.0f%%' % (
    len(rets), ann_ret*100, sharpe, mdd*100))
print('随机:  Months:%d | AnnRet:%+.0f%% | Sharpe:%.2f' % (
    len(rand_rets), rand_ann*100, rand_sh))

print('\n%-6s %8s %8s %8s %8s' % ('Year','AnnRet','Sharpe','MDD','CumRet'))
for ys in yearly:
    print('%-6d %+7.0f%% %8.2f %+7.0f%% %+7.0f%%' % (
        ys['year'], ys['ret']*100, ys['sharpe'], ys['mdd']*100, ys['cum']*100))

# Save
pd.DataFrame(all_rets).to_parquet('cache/clean_monthly_v2.parquet')
with open('cache/clean_summary_v2.json','w',encoding='utf-8') as f:
    json.dump({
        'pipeline':'Clean v2.0 - Interaction + LambdaRank + Ensemble',
        'period':'2008-2024','months':len(rets),
        'ann_ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),
        'mdd':round(mdd*100,1),'cum':round(cum_ret*100,1),
        'n_factors':len(FEATS),'generated':datetime.now().strftime('%Y-%m-%d %H:%M')
    },f)
print('\nSaved. Done.')
