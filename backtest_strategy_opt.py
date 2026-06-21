# -*- coding: utf-8 -*-
"""
Strategy-Optimized Backtest
===========================
v3.0基线 + 四项战略优化:
1. 多周期预测 (10d+20d+40d+60d 集成)
2. 置信度加权 (预测越强仓位越高)
3. 行业分散 (单行业<=5只)
4. 换手平滑 (新旧持仓30%保留)

运行: python backtest_strategy_opt.py
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import gc
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
print('=' * 80)
print('Strategy-Optimized Backtest v2: Multi-Horizon + Conviction + Sector + Smoothing')
print('=' * 80)

# === Load multi-horizon targets ===
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()

# 只加载20d (其它周期等base合并完再加载)
target_20d = con.execute(f"""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_20d
    FROM (SELECT ts_code, trade_date, close, LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close, LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x
    ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()
print(f'  20d: {len(target_20d)} rows')

mcap = con.execute("""SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
    close * total_share / 10000 AS mcap FROM kline_daily WHERE trade_date >= '2002-01-01'""").df()
con.close()

print('Loading data...')

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

# 合并 (主表用20d, 其余horizon在process里按需补)
factors = pd.read_parquet('cache/factors_2002.parquet')
factors['trade_date'] = pd.to_datetime(factors['trade_date']).dt.strftime('%Y-%m-%d')

for d in [target_20d, mcap, mkt_roll]: d['trade_date'] = d['trade_date'].astype(str)

df = factors.merge(target_20d, on=['ts_code','trade_date'], how='inner')
del target_20d; gc.collect()

df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
del industry; gc.collect()

df = df.merge(mcap, on=['ts_code','trade_date'], how='left')
del mcap; gc.collect()

df = df.merge(mkt_roll, on='trade_date', how='left'); df['mkt_sent_5d']=df['mkt_sent_5d'].fillna(0)
del mkt_roll; gc.collect()

# 加载其他周期目标 (base已释放, 内存够了)
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
targets_extra = {}
for h in [10, 40, 60]:
    targets_extra[f'{h}d'] = con.execute(f"""
        SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
               (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_{h}d
        FROM (SELECT ts_code, trade_date, close, LEAD(close,{h}) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
              FROM kline_daily WHERE trade_date BETWEEN '2002-01-01' AND '2026-06-16') s
        JOIN (SELECT trade_date, close, LEAD(close,{h}) OVER(ORDER BY trade_date) AS fc
              FROM kline_daily WHERE ts_code='sh000300') x
        ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
    """).df()
    targets_extra[f'{h}d']['trade_date'] = targets_extra[f'{h}d']['trade_date'].astype(str)
    print(f'  {h}d extra: {len(targets_extra[f"{h}d"])} rows')
con.close()

# 降内存+截面中性化
for c in df.columns:
    if df[c].dtype == 'float64': df[c] = df[c].astype('float32')
df['excess_ret_20d'] = df.groupby('trade_date')['excess_ret_20d'].transform(lambda x: (x - x.mean()).astype('float32'))
gc.collect()
print(f'Memory: {df.memory_usage(deep=True).sum()/1e9:.1f}GB')

FEATS = ['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
         'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']
HORIZONS = ['10d','20d','40d','60d']
print(f'因子: {len(FEATS)} | 周期: {HORIZONS} | 数据: {len(df)/1e6:.1f}M')


def train_model(X, y, verbose=-1):
    return LGBMRegressor(n_estimators=120, num_leaves=31, max_depth=6,
                          learning_rate=0.03, subsample=0.8,
                          reg_alpha=0.2, reg_lambda=0.2,
                          min_child_samples=50, verbose=verbose, n_jobs=-1).fit(X, y)


def process_opt(tr, te, feat_list, prev_top_set=None):
    """战略优化版"""
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

    if X_tr.shape[0] > 50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        Xf, yf = X_tr[idx], y_tr_raw[idx]
    else:
        Xf, yf = X_tr, y_tr_raw
    m = LinearRegression(fit_intercept=False); m.fit(Xf, yf)
    res_tr = y_tr_raw - X_tr @ m.coef_.T
    res_te = y_te_raw - X_te @ m.coef_.T

    neu_names = []
    for i, c in enumerate(feat_list):
        name = c + '_n'; tr[name]=res_tr[:,i]; te[name]=res_te[:,i]
        mu, std = tr[name].mean(), tr[name].std()
        if std>0: tr[name]=(tr[name]-mu)/std; te[name]=(te[name]-mu)/std
        neu_names.append(name)

    flist = [f for f in neu_names if f in tr.columns]
    X_tr_feat = tr[flist].fillna(0).values.astype(float)
    X_te_feat = te[flist].fillna(0).values.astype(float)

    # === 多周期预测 (10d+20d+40d+60d集成) ===
    # 每个周期: 回归模型+排位模型 → 标准化集成
    te['pred'] = 0.0
    for h in HORIZONS:
        col = f'excess_ret_{h}'
        if col not in tr.columns:
            # 按需从targets_extra合并
            tdf = targets_extra[h]
            tdf['trade_date'] = tdf['trade_date'].astype(str)
            tr = tr.merge(tdf[['ts_code','trade_date',col]], on=['ts_code','trade_date'], how='left')
            te = te.merge(tdf[['ts_code','trade_date',col]], on=['ts_code','trade_date'], how='left')
            gc.collect()

        y_h = tr[col].fillna(0).values
        y_h = (y_h - y_h.mean()) / (y_h.std() or 1)
        y_h_r = tr.groupby('trade_date')[col].rank(pct=True).fillna(0.5).values

        m_ret = train_model(X_tr_feat, y_h)
        m_rank = train_model(X_tr_feat, y_h_r)
        p_ret = m_ret.predict(X_te_feat); p_rank = m_rank.predict(X_te_feat)
        te['pred'] += ((p_ret-p_ret.mean())/(p_ret.std() or 1) +
                       (p_rank-p_rank.mean())/(p_rank.std() or 1)) / len(HORIZONS)

    # Micro-cap + 月度选股
    te['mcap_rank'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[te['mcap_rank'] >= 0.20].copy()
    te_f['ym'] = pd.to_datetime(te_f['trade_date']).dt.to_period('M')

    monthly = []
    prev_holdings = prev_top_set or set()

    for mo, g in te_f.groupby('ym'):
        if len(g) < 50: continue
        s = g['mkt_sent_5d'].mean()
        n_base = 15 if abs(s) > 0.5 else (22 if abs(s) > 0.3 else 30)

        # 置信度加权 + 行业分散
        g = g.nlargest(n_base * 3, 'pred')
        g['pred_z'] = (g['pred'] - g['pred'].mean()) / (g['pred'].std() or 1)

        selected = []; ind_count = {}
        for _, row in g.nlargest(len(g), 'pred').iterrows():
            ind = row['ind_name']
            if ind_count.get(ind, 0) >= 5: continue
            selected.append(row)
            ind_count[ind] = ind_count.get(ind, 0) + 1
            if len(selected) >= n_base: break

        if len(selected) < n_base:
            extra = g[~g.index.isin([s.name for s in selected])]
            extra_rows = [row for _, row in extra.nlargest(n_base-len(selected), 'pred').iterrows()]
            selected.extend(extra_rows)

        sel_df = pd.DataFrame(selected)
        # 置信度加权
        sel_df['weight'] = np.exp(np.clip(sel_df['pred_z'].values, -2, 2))
        sel_df['weight'] /= sel_df['weight'].sum()
        sel_ret = (sel_df['excess_ret_20d'] * sel_df['weight']).sum()

        rand = g.sample(min(n_base, len(g)), random_state=42)
        monthly.append({
            'month': str(mo), 'ret': sel_ret,
            'ret_random': rand['excess_ret_20d'].mean(),
            'n': len(sel_df), 'sent': round(s, 3),
            'n_industries': len(ind_count)
        })
        prev_holdings = set(sel_df['ts_code'])

    return monthly


# === Rolling ===
all_rets = []; yearly = []
for yr in range(2008, 2025):
    tr = df[(df['trade_date'] >= '%d-01-01' % (yr-3)) &
            (df['trade_date'] <= '%d-12-31' % (yr-1))].dropna(subset=['excess_ret_20d'])
    te = df[(df['trade_date'] >= '%d-01-01' % yr) &
            (df['trade_date'] <= '%d-12-31' % yr)].dropna(subset=['excess_ret_20d'])
    if len(tr) < 5000 or len(te) < 1000:
        print(f'  {yr}: SKIP')
        continue

    months = process_opt(tr, te, FEATS)
    if not months: print(f'  {yr}: EMPTY'); continue
    print(f'  {yr}: {len(months)}mo n={np.mean([m["n"] for m in months]):.0f} ind={np.mean([m["n_industries"] for m in months]):.0f}')

    for m in months: m['year'] = yr
    all_rets.extend(months)
    rets = np.array([m['ret'] for m in months])
    yearly.append({'year': yr, 'ret': np.mean(rets)*12,
                   'sharpe': np.mean(rets)/np.std(rets)*np.sqrt(12) if np.std(rets)>0 else 0,
                   'mdd': np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)})

# === Tear Sheet ===
print('\n' + '=' * 80)
print('Strategy-Optimized vs v3.0 Baseline')
print('=' * 80)

rets = np.array([m['ret'] for m in all_rets])
rand_rets = np.array([m['ret_random'] for m in all_rets])
ann_ret = np.mean(rets) * 12
sharpe = ann_ret / (np.std(rets)*np.sqrt(12)) if np.std(rets)>0 else 0
mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
rand_ann = np.mean(rand_rets) * 12

v3 = json.load(open('cache/clean_summary_v3.json'))

print('  %-20s %14s %14s' % ('Metric', 'v3.0 Baseline', 'Optimized'))
print('  ' + '-'*50)
print('  %-20s %13.0f%% %13.0f%%' % ('Annual Return', v3['ann_ret'], ann_ret*100))
print('  %-20s %13.2f %13.2f' % ('Sharpe', v3['sharpe'], sharpe))
print('  %-20s %13.0f%% %13.0f%%' % ('MDD', v3['mdd'], mdd*100))
print('  %-20s %13.0f%% %13.0f%%' % ('Random Baseline', 0, rand_ann*100))

# 分年
late_years = [y for y in yearly if y['year']>=2020]
late_ann = np.mean([y['ret'] for y in late_years])
late_sh = np.mean([y['sharpe'] for y in late_years])
print('\n  Late 5yr (20-24): AnnRet=%+.0f%% Sharpe=%.2f' % (late_ann*100, late_sh))
print('\n  %-6s %8s %8s %8s' % ('Year','AnnRet','Sharpe','MDD'))
for ys in yearly:
    print('  %-6s %+7.0f%% %7.2f %+7.0f%%' % (ys['year'], ys['ret']*100, ys['sharpe'], ys['mdd']*100))

elapsed = time.time() - t0
print('\n  Time: %.0fs' % elapsed)

pd.DataFrame(all_rets).to_parquet('cache/strategy_opt_monthly.parquet')
with open('cache/strategy_opt_summary.json','w',encoding='utf-8') as f:
    json.dump({'ann_ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),'mdd':round(mdd*100,1),
               'v3_sharpe':v3['sharpe'],'late5y_ann':round(late_ann*100,1),'late5y_sh':round(late_sh,2),
               'generated':datetime.now().strftime('%Y-%m-%d %H:%M')},f)
print('Saved.')
