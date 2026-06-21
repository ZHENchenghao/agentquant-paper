# -*- coding: utf-8 -*-
"""
Clean Rolling Backtest v1.0 — 零数据泄露版
============================================
只使用纯技术因子(RSI/均线/布林/波动率), 零前视偏差
财务因子(PE/PB/ROE等)全部砍掉 — 存在1-4月公告滞后导致的look-ahead
Walk-forward: 每年前3年训练, 当年测试 (2008-2024)
Industry neutralization + LightGBM + NLP timing overlay
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings; warnings.filterwarnings('ignore')

print('=' * 80)
print('Clean Rolling Backtest v1.0 — Technical Factors Only (Zero Look-Ahead)')
print('=' * 80)

# === Load ===
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()
# 目标: 20日超额收益 (vs 全A等权, 用截面中性化消除小盘系统性偏差)
# 每天减当天全市场均值 → 每个股票的excess_ret中心化为0
target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()

# === 截面中性化: 每天减全市场均值, 消除小盘系统性正偏 ===
print('截面中性化: excess_ret - daily_mean')
target['cs_neutral'] = target.groupby('trade_date')['excess_ret'].transform(lambda x: x - x.mean())
target['excess_ret'] = target['cs_neutral']
target.drop(columns=['cs_neutral'], inplace=True)
print(f'  中性化后 excess_ret 均值: {target.excess_ret.mean():.6f} (应≈0)')
mcap = con.execute("""SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
    close * total_share / 10000 AS mcap FROM kline_daily WHERE trade_date >= '2002-01-01'""").df()
con.close()

# NLP sentiment (正确用了shift(1): 无前视)
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

# === 纯技术因子 (零前视偏差) ===
TECH_FACTORS = ['rsi6','rsi14','boll_pos','boll_width',
                'div_ma20','div_ma60','div_ma120',
                'vol_ratio','ma_score','rsi_extreme',
                'margin_panic','streak5_dn']

print(f'技术因子: {len(TECH_FACTORS)}个 (RSI/布林/均线偏离/波动率/量比)')
print(f'已删除: PE/PB/PS/ROE/margins/log_eps/log_mcap (财务数据公告滞后1-4月→look-ahead)')

# Merge
factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')
for d in [target, mcap, mkt_roll]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
df = df.merge(mkt_roll, on='trade_date', how='left'); df['mkt_sent_5d']=df['mkt_sent_5d'].fillna(0)

# 只用技术因子
FEATS = [f for f in TECH_FACTORS if f in df.columns]
print(f'可用技术因子: {len(FEATS)}/{len(TECH_FACTORS)}')
print(f'数据: {len(df)/1e6:.1f}M行, {df["ts_code"].nunique()}只股票, {df["trade_date"].min()}~{df["trade_date"].max()}')
print()

# === Neutralize + Train ===
def process(tr, te, feat_list):
    tr, te = tr.copy(), te.copy()
    # 样本上限防止OOM
    if len(tr) > 500000: tr = tr.sample(500000, random_state=42)
    if len(te) > 300000: te = te.sample(300000, random_state=42)

    for d in [tr, te]:
        d['mcap'] = d['mcap'].fillna(
            d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['ln_mcap'] = np.log(d['mcap'].clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap'] ** 2

    # Industry dummies (aligned train/test)
    all_inds = sorted(set(tr['ind_name'].unique()) | set(te['ind_name'].unique()))
    ind_map = {ind: i for i, ind in enumerate(all_inds)}
    tr_dum = np.zeros((len(tr), len(all_inds)))
    te_dum = np.zeros((len(te), len(all_inds)))
    for i, ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i, ind_map[ind]] = 1
    for i, ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i, ind_map[ind]] = 1

    X_tr = np.nan_to_num(np.column_stack([
        tr['ln_mcap'].values, tr['ln_mcap_sq'].values, tr_dum
    ]).astype(float), 0)
    X_te = np.nan_to_num(np.column_stack([
        te['ln_mcap'].values, te['ln_mcap_sq'].values, te_dum
    ]).astype(float), 0)

    y_tr = np.nan_to_num(tr[feat_list].fillna(tr[feat_list].median()).fillna(0).values.astype(float), 0)
    y_te = np.nan_to_num(te[feat_list].fillna(te[feat_list].median()).fillna(0).values.astype(float), 0)

    # Quadratic OLS neutralization (sampled fit)
    if X_tr.shape[0] > 50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        Xf, yf = X_tr[idx], y_tr[idx]
    else:
        Xf, yf = X_tr, y_tr
    m = LinearRegression(fit_intercept=False); m.fit(Xf, yf)

    # Residuals + standardize
    res_tr = y_tr - X_tr @ m.coef_.T
    res_te = y_te - X_te @ m.coef_.T
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

    # LightGBM — 降低复杂度防止过拟合
    flist = [f for f in neu_names if f in tr.columns]
    model = LGBMRegressor(
        learning_rate=0.03, num_leaves=31, max_depth=6,
        subsample=0.7, colsample_bytree=0.7,
        n_estimators=100, reg_alpha=0.1, reg_lambda=0.1,
        min_child_samples=50,
        verbose=-1, n_jobs=-1
    )
    model.fit(tr[flist].fillna(tr[flist].median()), tr['excess_ret'].fillna(0))
    te['pred'] = model.predict(te[flist].fillna(tr[flist].median()))

    # Micro-cap filter
    te['mcap_rank'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[te['mcap_rank'] >= 0.20].copy()

    # NLP timing + 月度选股
    te_f['ym'] = pd.to_datetime(te_f['trade_date']).dt.to_period('M')
    monthly = []
    n_months_empty = 0
    for mo, g in te_f.groupby('ym'):
        if len(g) < 30:
            n_months_empty += 1
            continue
        if len(g) < 30: continue
        s = g['mkt_sent_5d'].mean()

        # === 模型选股 ===
        n = 15 if abs(s) > 0.5 else (22 if abs(s) > 0.3 else 30)
        top = g.nlargest(n, 'pred')

        # === 随机选股 (基准) ===
        rand = g.sample(min(n, len(g)), random_state=42)

        monthly.append({
            'month': str(mo),
            'ret': top['excess_ret'].mean(),
            'ret_random': rand['excess_ret'].mean(),
            'n': len(top), 'sent': round(s, 3)
        })
    if n_months_empty > 0:
        n_groups = len(list(te_f.groupby('ym')))
        print('    DEBUG: te_f=%d rows, %d/%d months <30 stocks' % (len(te_f), n_months_empty, n_groups))
    return monthly

# === Rolling Walk-Forward (2008-2024) ===
all_rets = []; yearly = []
for yr in range(2008, 2025):
    tr = df[(df['trade_date'] >= '%d-01-01' % (yr-3)) &
            (df['trade_date'] <= '%d-12-31' % (yr-1))].dropna(subset=['excess_ret'])
    te = df[(df['trade_date'] >= '%d-01-01' % yr) &
            (df['trade_date'] <= '%d-12-31' % yr)].dropna(subset=['excess_ret'])
    if len(tr) < 5000 or len(te) < 1000:
        print('  %d: SKIP tr=%d te=%d' % (yr, len(tr), len(te)))
        continue

    # 核心诊断: 确认train/test没有时间重叠
    tr_min = tr['trade_date'].min()
    tr_max = tr['trade_date'].max()
    te_min = te['trade_date'].min()
    te_max = te['trade_date'].max()
    if yr <= 2010:
        print('  %d: train=[%s,%s] test=[%s,%s]' % (yr, tr_min[:7], tr_max[:7], te_min[:7], te_max[:7]))

    months = process(tr, te, FEATS)
    if not months:
        print('  %d: process返回空 (te_raw=%d te_mcap_filtered=?)' % (yr, len(te)))
        continue
    for m in months: m['year'] = yr
    all_rets.extend(months)

    rets = np.array([m['ret'] for m in months])
    ann = np.mean(rets) * 12
    vol = np.std(rets, ddof=1) * np.sqrt(12) if len(rets) > 2 else 0.01
    sh = ann / vol if vol > 0 else 0
    cum = np.prod(1 + rets) - 1
    mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
    wr = np.mean(rets > 0)
    nm = np.mean([m['n'] for m in months])
    yearly.append({'year': yr, 'ret': ann, 'sharpe': sh, 'mdd': mdd,
                   'cum': cum, 'wr': wr, 'n': nm, 'months': len(months)})
    print('  %d: Ann=%+.0f%% Sharpe=%.2f MDD=%.0f%% WR=%.0f%% N=%.0f' % (
        yr, ann*100, sh, mdd*100, wr*100, nm))

# === Tear Sheet ===
print('\n' + '=' * 80)
print('CLEAN FINAL: 2008-2024 (17yr) | 12 Tech Factors + OLS Neu + LightGBM + NLP Timing')
print('=' * 80)

rets = np.array([m['ret'] for m in all_rets])
rand_rets = np.array([m['ret_random'] for m in all_rets])

ann_ret = np.mean(rets) * 12
ann_vol = np.std(rets, ddof=1) * np.sqrt(12)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
cum_ret = np.prod(1 + rets) - 1
mdd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)) - 1)
calmar = ann_ret / abs(mdd) if mdd != 0 else 0
wr = np.mean(rets > 0)

rand_ann = np.mean(rand_rets) * 12
rand_sh = rand_ann / (np.std(rand_rets, ddof=1) * np.sqrt(12)) if np.std(rand_rets) > 0 else 0
rand_cum = np.prod(1 + rand_rets) - 1

print('模型:  Months:%d | AnnRet:%+.1f%% | Sharpe:%.3f | MDD:%+.1f%% | Cum:%+.1f%%' % (
    len(rets), ann_ret*100, sharpe, mdd*100, cum_ret*100))
print('随机:  Months:%d | AnnRet:%+.1f%% | Sharpe:%.3f | Cum:%+.1f%%' % (
    len(rand_rets), rand_ann*100, rand_sh, rand_cum*100))
print('超额:  AnnRet:%+.1f%% (模型-随机)' % ((ann_ret-rand_ann)*100))

print('\n%-6s %8s %8s %8s %8s %6s' % ('Year','AnnRet','Sharpe','MDD','CumRet','N'))
for ys in yearly:
    print('%-6d %+7.0f%% %8.2f %+7.0f%% %+7.0f%% %5.0f' % (
        ys['year'], ys['ret']*100, ys['sharpe'], ys['mdd']*100, ys['cum']*100, ys['n']))

# 对比泄漏版
print('\n' + '=' * 80)
print('对比: 泄漏版 vs 清洁版')
print('=' * 80)
try:
    old = json.load(open('cache/production_summary.json'))
    print('泄漏版(24F含财务):     AnnRet=%+.0f%% Sharpe=%.3f MDD=%+.0f%% Cum=%+.0f%%' % (
        old['ann_ret'], old['sharpe'], old['mdd'], old['cum']))
except: pass
print('清洁版(12F纯技术):     AnnRet=%+.1f%% Sharpe=%.3f MDD=%+.1f%% Cum=%+.1f%%' % (
    ann_ret*100, sharpe, mdd*100, cum_ret*100))

# Save
pd.DataFrame(all_rets).to_parquet('cache/clean_monthly_v1.parquet')
with open('cache/clean_summary_v1.json','w',encoding='utf-8') as f:
    json.dump({
        'pipeline':'Clean v1.0 - Tech Factors Only',
        'period':'2008-2024','months':len(rets),
        'ann_ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),
        'mdd':round(mdd*100,1),'cum':round(cum_ret*100,1),
        'factors':FEATS,'n_factors':len(FEATS),
        'generated':datetime.now().strftime('%Y-%m-%d %H:%M')
    },f)
print('\nSaved. Done.')
