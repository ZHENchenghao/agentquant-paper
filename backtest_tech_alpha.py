# -*- coding: utf-8 -*-
"""
Tech-Alpha Engine v2 — 降方差版
=================================
进攻: 5硬科技行业 + 3周期集成(5d/10d/20d加权递减)
      + 高确信度门槛(信号弱→空仓) + 贝叶斯收缩(不确定→等权)

防御: MA200双条件择时 → 国债40%+黄金30%+纳指30%

分年加载 + 全审计修复
"""
import sys, io, os, json, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
import duckdb, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import warnings; warnings.filterwarnings('ignore')

t0 = time.time()
print('=' * 80)
print('Tech-Alpha Engine — 硬科技进攻 + MA200防御')
print('=' * 80)

# === 硬科技资产池 ===
TECH_INDUSTRIES = ['电子', '计算机', '通信', '国防军工', '机械设备']

# === 静态资源 ===
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

industry = con.execute("""SELECT ts_code, ind_name FROM (
    SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
    FROM stock_industry_map) WHERE rn = 1""").df()
# 只保留硬科技行业
industry = industry[industry['ind_name'].isin(TECH_INDUSTRIES)]
tech_codes = set(industry['ts_code'])
print(f'Tech universe: {len(TECH_INDUSTRIES)} industries, {len(tech_codes)} stocks (filtered from industry map)')

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

FEATS = ['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
         'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

STAMP, COMM, SLIP = 0.0005, 0.0002, 0.0015
MONTHLY_COST = 0.75 * 2 * (STAMP + COMM + SLIP)
print(f'Fee: {MONTHLY_COST*100:.2f}%/mo | Tech stocks: ~{len(tech_codes)} codes | Audit fixes: ON')


# === 防御资产 ===
def load_defense_assets():
    """国债+黄金+纳指日收益"""
    import akshare as ak
    # 国债 (10Y yield → bond return proxy)
    bond = ak.bond_zh_us_rate()
    cols = list(bond.columns)
    bond = bond.rename(columns={cols[0]:'trade_date', cols[3]:'yield_10y'})
    bond['trade_date'] = pd.to_datetime(bond['trade_date'])
    bond = bond[['trade_date','yield_10y']].dropna().sort_values('trade_date')
    bond['ret'] = -8.0 * bond['yield_10y'].diff() / 100
    bond = bond.set_index('trade_date')[['ret']]

    # 黄金
    gold = ak.futures_foreign_hist(symbol='XAU')
    gold['trade_date'] = pd.to_datetime(gold['date'])
    gold = gold.dropna(subset=['close']).sort_values('trade_date')
    gold['ret'] = gold['close'].pct_change()
    gold = gold.set_index('trade_date')[['ret']]

    # 纳指
    nasdaq = ak.index_us_stock_sina(symbol='.IXIC')
    nasdaq['trade_date'] = pd.to_datetime(nasdaq['date'])
    nasdaq = nasdaq.dropna(subset=['close']).sort_values('trade_date')
    nasdaq['ret'] = nasdaq['close'].pct_change()
    nasdaq = nasdaq.set_index('trade_date')[['ret']]

    # 合并防御组合: 国债40%+黄金30%+纳指30%
    defense = pd.DataFrame(index=bond.index)
    defense['bond'] = bond['ret']
    defense['gold'] = gold['ret'].reindex(defense.index).fillna(0)
    defense['nasdaq'] = nasdaq['ret'].reindex(defense.index).fillna(0)
    defense = defense.dropna()
    defense['port'] = defense['bond']*0.4 + defense['gold']*0.3 + defense['nasdaq']*0.3
    return defense[['port']]

print('Loading defense assets (bond+gold+nasdaq)...')
defense_rets = load_defense_assets()
print(f'  Defense: {len(defense_rets)} days ({defense_rets.index[0].date()}~{defense_rets.index[-1].date()})')


# === MA200 择时 ===
def compute_regime(hs300_price):
    close = hs300_price.dropna().sort_index()
    ma200 = close.rolling(200, min_periods=200).mean()
    slope = ma200.pct_change(5)
    raw = (close > ma200) & (slope > -0.01)
    # 3日缓冲
    initial = raw.iloc[:3].mean() >= 0.5
    regime = pd.Series('BULL' if initial else 'BEAR', index=raw.index)
    bull_streak = 0; bear_streak = 0; current = regime.iloc[0]
    for i in range(len(raw)):
        if raw.iloc[i]:
            bull_streak += 1; bear_streak = 0
            if bull_streak >= 3 and current != 'BULL': current = 'BULL'
        else:
            bear_streak += 1; bull_streak = 0
            if bear_streak >= 3 and current != 'BEAR': current = 'BEAR'
        regime.iloc[i] = current
    return regime


def load_fold(conn, yr):
    """分年加载硬科技股票数据"""
    tr_start = f'{yr-3}-01-01'; tr_end = f'{yr-1}-12-31'
    te_start = f'{yr}-01-01'; te_end = f'{yr}-12-31'
    all_start = tr_start; all_end = te_end

    factors = pd.read_parquet('cache/factors_2002.parquet')
    factors['trade_date'] = pd.to_datetime(factors['trade_date'])
    factors = factors[(factors['trade_date'] >= all_start) & (factors['trade_date'] <= all_end)]
    factors['trade_date'] = factors['trade_date'].dt.strftime('%Y-%m-%d')
    # 只保留硬科技标的
    factors = factors[factors['ts_code'].isin(tech_codes)]

    # 多周期目标: 5d, 10d, 20d
    target_dfs = []
    for h in [10, 20]:
        tdf = conn.execute(f"""
            SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
                   (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_{h}d,
                   s.close/LAG(s.close) OVER(PARTITION BY s.ts_code ORDER BY s.trade_date)-1.0 AS ret_1d,
                   s.vol AS volume
            FROM (SELECT ts_code, trade_date, close,LEAD(close,{h}) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc,
                         vol FROM kline_daily WHERE trade_date BETWEEN '{all_start}' AND '{all_end}') s
            JOIN (SELECT trade_date, close, LEAD(close,{h}) OVER(ORDER BY trade_date) AS fc
                  FROM kline_daily WHERE ts_code='sh000300' AND trade_date BETWEEN '{all_start}' AND '{all_end}') x
            ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
        """).df()
        tdf['trade_date'] = tdf['trade_date'].astype(str)
        tdf = tdf[tdf['ts_code'].isin(tech_codes)]
        if h == 20:
            target = tdf
        else:
            target_dfs.append(tdf[['ts_code','trade_date',f'excess_ret_{h}d']])
    target['trade_date'] = target['trade_date'].astype(str)
    target = target[target['ts_code'].isin(tech_codes)]

    mcap = conn.execute(f"""
        SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
               close*total_share/10000 AS mcap
        FROM kline_daily WHERE trade_date BETWEEN '{all_start}' AND '{all_end}'
    """).df()
    mcap['trade_date'] = mcap['trade_date'].astype(str)

    fold_df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
    del target; gc.collect()
    # 合并额外周期目标
    for tdf in target_dfs:
        fold_df = fold_df.merge(tdf, on=['ts_code','trade_date'], how='left')
    fold_df = fold_df.merge(industry, on='ts_code', how='left')
    fold_df = fold_df.merge(mcap, on=['ts_code','trade_date'], how='left')
    del mcap; gc.collect()
    fold_df = fold_df.merge(mkt_roll, on='trade_date', how='left'); fold_df['mkt_sent_5d']=fold_df['mkt_sent_5d'].fillna(0)

    for c in fold_df.columns:
        if fold_df[c].dtype == 'float64': fold_df[c] = fold_df[c].astype('float32')

    tr = fold_df[(fold_df['trade_date']>=tr_start)&(fold_df['trade_date']<=tr_end)].dropna(subset=['excess_ret_20d']).copy()
    te = fold_df[(fold_df['trade_date']>=te_start)&(fold_df['trade_date']<=te_end)].dropna(subset=['excess_ret_20d']).copy()
    del fold_df; gc.collect()
    return tr, te


def train_model(X, y):
    return LGBMRegressor(n_estimators=120, num_leaves=31, max_depth=6,
                          learning_rate=0.03, subsample=0.8,
                          reg_alpha=0.2, reg_lambda=0.2,
                          min_child_samples=50, verbose=-1, n_jobs=-1).fit(X, y)


def process_tech(tr, te, feat_list):
    """硬科技ML管线: 全审计修复"""
    tr, te = tr.copy(), te.copy()
    tr['excess_ret_20d'] = tr.groupby('trade_date')['excess_ret_20d'].transform(lambda x: x - x.mean())
    te['excess_ret_20d'] = te.groupby('trade_date')['excess_ret_20d'].transform(lambda x: x - x.mean())

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
    res_tr = y_tr_raw - X_tr @ m.coef_.T; res_te = y_te_raw - X_te @ m.coef_.T

    neu_names = []
    for i, c in enumerate(feat_list):
        name = c + '_n'; tr[name]=res_tr[:,i]; te[name]=res_te[:,i]
        mu, std = tr[name].mean(), tr[name].std()
        if std>0: tr[name]=(tr[name]-mu)/std; te[name]=(te[name]-mu)/std
        neu_names.append(name)

    flist = [f for f in neu_names if f in tr.columns]
    X_tr_feat = tr[flist].fillna(0).values.astype(float)
    X_te_feat = te[flist].fillna(0).values.astype(float)

    # === 降方差#1: 多周期衰退集成 (5d×0.5 + 10d×0.3 + 20d×0.2) ===
    HORIZON_WEIGHTS = [(10,0.5), (20,0.5)]  # 5d太噪, 只用10d+20d
    te['pred'] = 0.0
    fi = {}
    n_models = 0

    for h, hw in HORIZON_WEIGHTS:
        col = f'excess_ret_{h}d'
        if col not in tr.columns: continue

        y_h = tr[col].fillna(0).values
        y_h = (y_h - y_h.mean()) / (y_h.std() or 1)
        y_h_r = tr.groupby('trade_date')[col].rank(pct=True).fillna(0.5).values

        m_ret = train_model(X_tr_feat, y_h)
        m_rank = train_model(X_tr_feat, y_h_r)
        p_r = m_ret.predict(X_te_feat); p_k = m_rank.predict(X_te_feat)
        te['pred'] += hw * ((p_r-p_r.mean())/(p_r.std() or 1) + (p_k-p_k.mean())/(p_k.std() or 1))

        for fk, fv in zip(flist, m_ret.feature_importances_):
            orig = fk.replace('_n',''); fi[orig] = fi.get(orig,0) + float(fv)*hw
        n_models += 2

    total = sum(fi.values()) or 1
    for f in fi: fi[f] /= total

    # === 降方差#2: 高确信度门槛 ===
    te['pred_std'] = te.groupby('trade_date')['pred'].transform('std')
    pred_std_median = te['pred_std'].median()
    te['confident'] = te['pred_std'] >= pred_std_median * 0.5

    # === 降方差#3: 贝叶斯收缩 (温和版) ===
    pred_z = (te['pred'] - te['pred'].mean()) / (te['pred'].std() or 1)
    extreme = np.abs(pred_z) > 3.0  # 只收缩3σ外的极端值
    te.loc[extreme, 'pred'] = te.loc[extreme, 'pred'].values * 0.8

    tr_sent_abs = tr['mkt_sent_5d'].abs()
    sent_high = tr_sent_abs.quantile(0.67) if len(tr_sent_abs)>10 else 0.5
    sent_mid = tr_sent_abs.quantile(0.33) if len(tr_sent_abs)>10 else 0.3

    te_clean = te[(te['ret_1d'].fillna(0) < 0.095) & (te['ret_1d'].fillna(0) > -0.098) &
                  (te['volume'].fillna(0) > 0)].copy()
    te_f = te_clean[te_clean.groupby('trade_date')['mcap'].rank(pct=True) >= 0.20].copy()
    te_f['ym'] = pd.to_datetime(te_f['trade_date']).dt.to_period('M')

    monthly = []
    for mo, g in te_f.groupby('ym'):
        if len(g) < 20: continue  # 科技池小, 放宽到20
        s = g['mkt_sent_5d'].mean()
        # 高确信度门控: 不确定→减仓
        confident_pct = g['confident'].mean()
        n_base = min(15 if abs(s) > sent_high else (20 if abs(s) > sent_mid else 25), len(g)//2)
        n_base = max(5, int(n_base * min(1.0, 0.5 + confident_pct)))  # 确信度50%→不缩, <50%渐进缩

        g = g.nlargest(n_base * 3, 'pred')
        g['pred_z'] = (g['pred'] - g['pred'].mean()) / (g['pred'].std() or 1)

        selected = []; ind_count = {}
        for _, row in g.nlargest(len(g), 'pred').iterrows():
            ind = row['ind_name']
            if ind_count.get(ind, 0) >= 8: continue  # 5行业, 上限放宽到8
            selected.append(row); ind_count[ind] = ind_count.get(ind, 0) + 1
            if len(selected) >= n_base: break
        if len(selected) < n_base:
            extra = g[~g.index.isin([s.name for s in selected])]
            selected.extend([row for _, row in extra.nlargest(n_base-len(selected), 'pred').iterrows()])

        sel_df = pd.DataFrame(selected)
        sel_df['weight'] = np.exp(np.clip(sel_df['pred_z'].values, -2, 2))
        sel_df['weight'] /= sel_df['weight'].sum()
        gross_ret = (sel_df['excess_ret_20d'] * sel_df['weight']).sum()
        net_ret = gross_ret - MONTHLY_COST

        all_day = te_f[te_f['ym'] == mo]
        rand_all = all_day.sample(min(n_base, len(all_day)), random_state=None) if len(all_day)>=n_base else all_day

        monthly.append({
            'month': str(mo), 'ret': net_ret, 'ret_gross': gross_ret,
            'ret_random': rand_all['excess_ret_20d'].mean(), 'n': len(sel_df)
        })

    return monthly, fi


# === 主循环 ===

# 加载沪深300用于择时
hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date BETWEEN '2005-06-01' AND '2026-06-19'
    ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300_price = hs300.set_index('trade_date')['close']
regime = compute_regime(hs300_price)
print(f'Regime: BULL {((regime=="BULL").mean()*100):.0f}% BEAR {((regime=="BEAR").mean()*100):.0f}%')

all_rets = []; yearly = []; yearly_fi = {}
for yr in range(2008, 2025):
    tr, te = load_fold(con, yr)
    if len(tr) < 3000 or len(te) < 500:
        print(f'  {yr}: SKIP tr={len(tr)} te={len(te)}')
        continue

    months, fi = process_tech(tr, te, FEATS)
    if not months: print(f'  {yr}: EMPTY'); continue

    # Apply regime + defense
    yr_regime = regime[regime.index.year == yr]
    bull_pct = (yr_regime == 'BULL').mean() if len(yr_regime) > 0 else 0.5

    # 计算该年defense月度收益
    defense_in_year = defense_rets[defense_rets.index.year == yr]
    defense_monthly = None
    if len(defense_in_year) > 0:
        defense_monthly = defense_in_year['port'].resample('M').apply(lambda x: (1+x).prod()-1)

    for m in months:
        m['year'] = yr
        mo_str = m['month']
        try:
            mo_end = pd.Period(mo_str).end_time
            regime_dates = regime.index[regime.index <= mo_end]
            is_bull = regime.loc[regime_dates[-1]] == 'BULL' if len(regime_dates)>0 else True
        except:
            is_bull = True

        if not is_bull and defense_monthly is not None:
            # Bear: 防御组合当月收益
            try:
                def_ret = defense_monthly.loc[mo_str]
                m['ret'] = float(def_ret) if not pd.isna(def_ret) else 0
            except:
                m['ret'] = 0
            m['ret_gross'] = m['ret']

    all_rets.extend(months)
    rets = np.array([m['ret'] for m in months])
    print(f'  {yr}: {len(months)}mo net={np.mean(rets)*12*100:+.0f}% regime-BULL={bull_pct:.0%}')

    yearly.append({'year': yr, 'ret': np.mean(rets)*12,
                   'sharpe': np.mean(rets)/np.std(rets)*np.sqrt(12) if np.std(rets)>0 else 0,
                   'mdd': np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)})
    yearly_fi[yr] = fi

con.close()

# === Tear Sheet ===
print('\n' + '=' * 80)
print('TECH-ALPHA ENGINE — Final Report')
print('=' * 80)

rets = np.array([m['ret'] for m in all_rets])
rand_rets = np.array([m.get('ret_random', 0) for m in all_rets])
ann_ret = np.mean(rets) * 12
sharpe = ann_ret / (np.std(rets)*np.sqrt(12)) if np.std(rets)>0 else 0
mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
rand_ann = np.mean(rand_rets) * 12

print(f'\n  Universe: {len(TECH_INDUSTRIES)} tech industries (~{len(tech_codes)} stocks)')
print(f'  Months: {len(rets)} | AnnRet: {ann_ret*100:+.0f}% | Sharpe: {sharpe:.2f} | MDD: {mdd*100:.0f}%')
print(f'  Random baseline: {rand_ann*100:+.0f}%')

print(f'\n  {"Year":<6} {"AnnRet":>8} {"Sharpe":>8} {"MDD":>8}')
for ys in yearly:
    print(f'  {ys["year"]:<6} {ys["ret"]*100:>+7.0f}% {ys["sharpe"]:>7.2f} {ys["mdd"]*100:>+7.0f}%')

late = [y for y in yearly if y['year']>=2020]
late_ann = np.mean([y['ret'] for y in late]) if late else 0
late_sh = np.mean([y['sharpe'] for y in late]) if late else 0
print(f'\n  Late 5yr (20-24): AnnRet={late_ann*100:+.0f}% Sharpe={late_sh:.2f}')

# Factor importance
print(f'\n  Factor Importance (17yr avg):')
all_fi = {}
for yr, fi in yearly_fi.items():
    for f, v in fi.items():
        all_fi[f] = all_fi.get(f, 0) + v
n_yr = max(len(yearly_fi), 1)
for f in all_fi: all_fi[f] /= n_yr
for f, v in sorted(all_fi.items(), key=lambda x: x[1], reverse=True):
    bar = '█' * max(1, int(v * 300))
    print(f'  {f:20s} {v:.4f} {bar}')

elapsed = time.time() - t0
print(f'\n  Time: {elapsed:.0f}s')

pd.DataFrame(all_rets).to_parquet('cache/tech_alpha_monthly.parquet')
with open('cache/tech_alpha_summary.json','w') as f:
    json.dump({'ann_ret':round(ann_ret*100,1),'sharpe':round(sharpe,3),'mdd':round(mdd*100,1),
               'late5y_ann':round(late_ann*100,1),'late5y_sh':round(late_sh,2),
               'universe':TECH_INDUSTRIES,'n_stocks':len(tech_codes),
               'generated':datetime.now().strftime('%Y-%m-%d %H:%M')},f)
print('Saved.')
