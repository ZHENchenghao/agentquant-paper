# -*- coding: utf-8 -*-
"""
Clean Rolling Backtest v3.0 — 回归本质
=======================================
v1.0的简洁 + v2.0的双目标 + 更好正则化
- 12纯技术因子 (零前视)
- 双目标: excess_ret + 截面排位百分数 → LightGBM分别预测 → 等权集成
- 更大训练样本(500K) + L1/L2正则
- 截面中性化目标
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
print('=' * 80)
print('Clean Rolling Backtest v3.0 — Dual Target + Regularized')
print('=' * 80)

# === Load (same as v1.0) ===
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

# 截面中性化
target['excess_ret'] = target.groupby('trade_date')['excess_ret'].transform(lambda x: x - x.mean())

# NLP
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

TECH_FACTORS = ['rsi6','rsi14','boll_pos','boll_width',
                'div_ma20','div_ma60','div_ma120',
                'vol_ratio','ma_score','rsi_extreme',
                'margin_panic','streak5_dn']

factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')
for d in [target, mcap, mkt_roll]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(mkt_roll, on='trade_date', how='left'); df['mkt_sent_5d']=df['mkt_sent_5d'].fillna(0)

FEATS = [f for f in TECH_FACTORS if f in df.columns]
print(f'因子: {len(FEATS)}个 | 数据: {len(df)/1e6:.1f}M行 | {df.trade_date.min()}~{df.trade_date.max()}')


def process_v3(tr, te, feat_list):
    """双目标LightGBM: excess_ret + 截面排位"""
    tr, te = tr.copy(), te.copy()
    if len(tr) > 500000: tr = tr.sample(500000, random_state=None)
    if len(te) > 300000: te = te.sample(300000, random_state=None)

    for d in [tr, te]:
        d['mcap'] = d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['ln_mcap'] = np.log(d['mcap'].clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap'] ** 2

    all_inds = sorted(set(tr['ind_name'].unique()) | set(te['ind_name'].unique()))
    ind_map = {ind: i for i, ind in enumerate(all_inds)}
    tr_dum = np.zeros((len(tr), len(all_inds)))
    te_dum = np.zeros((len(te), len(all_inds)))
    for i, ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i, ind_map[ind]] = 1
    for i, ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i, ind_map[ind]] = 1

    X_tr = np.column_stack([tr['ln_mcap'].values, tr['ln_mcap_sq'].values, tr_dum])
    X_te = np.column_stack([te['ln_mcap'].values, te['ln_mcap_sq'].values, te_dum])
    y_tr_raw = np.nan_to_num(tr[feat_list].fillna(0).values.astype(float), 0)
    y_te_raw = np.nan_to_num(te[feat_list].fillna(0).values.astype(float), 0)

    # OLS中性化
    if X_tr.shape[0] > 50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        Xf, yf = X_tr[idx], y_tr_raw[idx]
    else:
        Xf, yf = X_tr, y_tr_raw
    m = LinearRegression(fit_intercept=False); m.fit(Xf, yf)

    res_tr = y_tr_raw - X_tr @ m.coef_.T
    res_te = y_te_raw - X_te @ m.coef_.T

    neu_names = []
    for i, col in enumerate(feat_list):
        name = col + '_n'
        tr[name] = res_tr[:, i]
        te[name] = res_te[:, i]
        mu, std = tr[name].mean(), tr[name].std()
        if std > 0:
            tr[name] = (tr[name] - mu) / std
            te[name] = (te[name] - mu) / std
        neu_names.append(name)

    flist = [f for f in neu_names if f in tr.columns]
    X_tr_feat = tr[flist].fillna(0).values.astype(float)
    X_te_feat = te[flist].fillna(0).values.astype(float)

    # === 双目标 ===
    # 目标1: 标准化后的excess_ret (回归)
    y1 = tr['excess_ret'].fillna(0).values
    y1 = (y1 - y1.mean()) / (y1.std() or 1)

    # 目标2: 截面排位百分数 (更贴近选股)
    y2 = tr.groupby('trade_date')['excess_ret'].rank(pct=True).fillna(0.5).values

    # Model 1: 预测超额收益
    m1 = LGBMRegressor(n_estimators=150, num_leaves=31, max_depth=6,
                        learning_rate=0.03, subsample=0.8,
                        reg_alpha=0.2, reg_lambda=0.2,
                        min_child_samples=50, verbose=-1, n_jobs=-1)
    m1.fit(X_tr_feat, y1)

    # Model 2: 预测截面排位
    m2 = LGBMRegressor(n_estimators=150, num_leaves=31, max_depth=6,
                        learning_rate=0.03, subsample=0.8,
                        reg_alpha=0.2, reg_lambda=0.2,
                        min_child_samples=50, verbose=-1, n_jobs=-1)
    m2.fit(X_tr_feat, y2)

    # 集成
    p1 = m1.predict(X_te_feat)
    p2 = m2.predict(X_te_feat)
    te['pred'] = (p1 - p1.mean()) / (p1.std() or 1) + \
                 (p2 - p2.mean()) / (p2.std() or 1)

    # Micro-cap + 月度选股
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


# === Rolling ===
all_rets = []; yearly = []
for yr in range(2008, 2025):
    tr = df[(df['trade_date'] >= '%d-01-01' % (yr-3)) &
            (df['trade_date'] <= '%d-12-31' % (yr-1))].dropna(subset=['excess_ret'])
    te = df[(df['trade_date'] >= '%d-01-01' % yr) &
            (df['trade_date'] <= '%d-12-31' % yr)].dropna(subset=['excess_ret'])
    if len(tr) < 5000 or len(te) < 1000: continue

    months = process_v3(tr, te, FEATS)
    if not months: continue
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
print('v3.0 FINAL: 2008-2024 (17yr) | Dual Target + Strong Regularization')
print('=' * 80)

rets = np.array([m['ret'] for m in all_rets])
rand_rets = np.array([m['ret_random'] for m in all_rets])
ann_ret = np.mean(rets) * 12
ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
rand_ann = np.mean(rand_rets) * 12

print('模型:  Months:%d | AnnRet:%+.0f%% | Sharpe:%.2f | MDD:%+.0f%%' % (
    len(rets), ann_ret*100, sharpe, mdd*100))
print('随机:  AnnRet:%+.0f%%' % (rand_ann*100))

# 综合: 取最后5年(2020-2024)评估稳定性
late_rets = np.array([m['ret'] for m in all_rets if m['year'] >= 2020])
late_ann = np.mean(late_rets) * 12
late_sh = late_ann / (np.std(late_rets, ddof=1) * np.sqrt(12)) if np.std(late_rets) > 0 else 0
print('近5年(20-24): AnnRet:%+.0f%% Sharpe:%.2f' % (late_ann*100, late_sh))

print('\n%-6s %8s %8s %8s %8s' % ('Year','AnnRet','Sharpe','MDD','CumRet'))
for ys in yearly:
    print('%-6d %+7.0f%% %8.2f %+7.0f%% %+7.0f%%' % (
        ys['year'], ys['ret']*100, ys['sharpe'], ys['mdd']*100, ys['cum']*100))

elapsed = time.time() - t0
print(f'\n总耗时: {elapsed:.0f}s')

# Save
pd.DataFrame(all_rets).to_parquet('cache/clean_monthly_v3.parquet')
v1 = json.load(open('cache/clean_summary_v1.json'))
print(f'\nv1.0(12F单目标): AnnRet={v1["ann_ret"]:.0f}% Sharpe={v1["sharpe"]:.2f} MDD={v1["mdd"]:.0f}%')
print(f'v3.0(12F双目标): AnnRet={ann_ret*100:.0f}% Sharpe={sharpe:.2f} MDD={mdd*100:.0f}%')

with open('cache/clean_summary_v3.json','w',encoding='utf-8') as f:
    json.dump({'pipeline':'Clean v3.0','period':'2008-2024','months':len(rets),
               'ann_ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),
               'mdd':round(mdd*100,1),'n_factors':len(FEATS),
               'late5y_ann':round(late_ann*100,1),'late5y_sh':round(late_sh,2),
               'generated':datetime.now().strftime('%Y-%m-%d %H:%M')},f)
print('\nSaved.')
