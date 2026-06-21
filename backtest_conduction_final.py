# -*- coding: utf-8 -*-
"""
Conduction features -> LightGBM backtest.
Each stock gets a daily conduction score based on all validated macro->basket links.
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
print('Conduction Features -> Factor Model Backtest')
print('=' * 80)

# ============================================================
# Validated conduction baskets
# ============================================================
BASKETS = {
    # STRONG (acc>62%)
    '黄金矿业': ['sh601899','sh600547','sh600489','sz002155','sh600988','sz000975'],
    '铜矿': ['sh600362','sz000630','sh601168','sz002203','sh603799'],
    '铝业': ['sh601600','sh000807','sz002532','sh603993'],
    # VALID (58-62%)
    '石油开采': ['sh601857','sh600028','sh600938','sh601808','sh600583'],
    '石油炼化': ['sh600346','sh000301','sh600688','sz002493'],
    # WEAK but good t (55-58%, t>4)
    '光模块CPO': ['sz300308','sz300502','sz300394','sz002281','sz300570'],
    'PCB载板': ['sz002463','sz002916','sz002938','sz300476','sz002384'],
    '半导体设备': ['sh688012','sz002371','sh688082','sh688120'],
    'AI服务器': ['sh601138','sz000977','sh603019','sz000938'],
}

# Macro -> basket links
LINKS = [
    ('gold', '黄金矿业', '+', 2),
    ('copper', '铜矿', '+', 1),
    ('copper', '铝业', '+', 1),
    ('wti', '石油开采', '+', 1),
    ('wti', '石油炼化', '+', 1),
    ('sox', '光模块CPO', '+', 0),
    ('sox', 'PCB载板', '+', 3),
    ('sox', '半导体设备', '+', 1),
    ('sox', 'AI服务器', '+', 3),
]

# ============================================================
# Load data
# ============================================================
con = duckdb.connect(DB, read_only=True)

# Factor + target
factors = pd.read_parquet(CACHE)
factors['trade_date'] = factors['trade_date'].astype(str)

target = con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1) - (x.fc/x.close-1) AS excess_ret
    FROM (SELECT ts_code, trade_date, close,
          LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date BETWEEN '2016-01-01' AND '2026-06-16') s
    JOIN (SELECT trade_date, close,
          LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300') x ON s.trade_date=x.trade_date
    WHERE s.fc IS NOT NULL
""").df()
target['trade_date'] = target['trade_date'].astype(str)

# Macro data
macro = con.execute("""
    SELECT trade_date, wti, copper, gold FROM macro_indicators
    WHERE trade_date >= '2016-01-01' ORDER BY trade_date
""").df()
for c in ['wti', 'copper', 'gold']:
    macro[c] = macro[c].ffill().bfill()

# SOX
sox = con.execute("""
    SELECT trade_date, close AS sox FROM global_index_daily
    WHERE index_code = '.SOX' AND trade_date >= '2019-01-01' ORDER BY trade_date
""").df()

# Stock returns for basket calc
all_basket_stocks = set()
for s in BASKETS.values():
    all_basket_stocks.update(s)

stock_ret = con.execute("""
    SELECT ts_code, trade_date,
           close / LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date) - 1 AS ret
    FROM kline_daily WHERE trade_date >= '2016-01-01'
""").df().dropna(subset=['ret'])
stock_ret = stock_ret[stock_ret['ts_code'].isin(all_basket_stocks)]

con.close()

# ============================================================
# Build conduction scores per stock per day
# ============================================================
print('\nBuilding conduction scores...')

# Pre-compute basket daily returns
basket_rets = {}
for name, stocks in BASKETS.items():
    br = stock_ret[stock_ret['ts_code'].isin(stocks)].groupby('trade_date')['ret'].mean()
    basket_rets[name] = br

# Compute conduction feature for each stock:
# score = sum over links: direction * basket_return * (stock_in_basket ? 1 : 0)
# For simplicity: assign basket-level score to each stock in that basket.

# Daily basket scores based on macro triggers
macro_daily = macro.copy()
macro_daily['trade_date'] = macro_daily['trade_date'].astype(str)

sox_daily = sox.copy()
sox_daily['trade_date'] = sox_daily['trade_date'].astype(str)
sox_daily['sox'] = sox_daily['sox'].astype(float)

# Build daily macro change
for col in ['wti', 'copper', 'gold']:
    macro_daily[col + '_chg'] = macro_daily[col].pct_change()
sox_daily['sox_chg'] = sox_daily['sox'].pct_change()

# Merge with factor data
df = factors.merge(target, on=['ts_code', 'trade_date'], how='inner')
df['trade_date'] = df['trade_date'].astype(str)

# Add conduction features
df['cond_score'] = 0.0
df['cond_count'] = 0

for macro_col, basket_name, direction, lag in LINKS:
    # Get macro changes
    if macro_col == 'sox':
        md = sox_daily[['trade_date', 'sox_chg']].copy()
        md.columns = ['trade_date', 'chg']
    else:
        col_name = macro_col + '_chg'
        md = macro_daily[['trade_date', col_name]].dropna()
        md.columns = ['trade_date', 'chg']

    # Apply lag: today's macro change predicts basket return in lag days
    md['signal_date'] = pd.to_datetime(md['trade_date'])
    md['effective_date'] = md['signal_date'] + pd.Timedelta(days=lag)
    md['trade_date'] = md['effective_date'].dt.strftime('%Y-%m-%d')

    # Get basket returns on effective dates
    if basket_name in basket_rets:
        br = basket_rets[basket_name].reset_index()
        br.columns = ['trade_date', 'basket_ret']
        br['trade_date'] = br['trade_date'].astype(str)
        md['trade_date'] = md['trade_date'].astype(str)

        # Merge macro signal with basket return
        ms = md.merge(br, on='trade_date', how='inner')

        # Signal: macro_chg direction * basket_ret direction
        if direction == '+':
            ms['signal'] = np.sign(ms['chg']) * ms['basket_ret']
        else:
            ms['signal'] = -np.sign(ms['chg']) * ms['basket_ret']

        # Assign to stocks in basket
        basket_stocks = BASKETS[basket_name]
        for ts in basket_stocks:
            if ts not in df['ts_code'].values:
                continue
            stock_signal = ms[['trade_date', 'signal']].copy()
            stock_signal['ts_code'] = ts

            # Merge into df
            mask = df['ts_code'] == ts
            df_temp = df.loc[mask, ['ts_code', 'trade_date']].merge(
                stock_signal, on=['ts_code', 'trade_date'], how='left')
            df.loc[mask, 'cond_score'] += df_temp['signal'].fillna(0).values
            df.loc[mask, 'cond_count'] += (~df_temp['signal'].isna()).astype(int).values

print('  Conduction coverage: %.1f%% of rows' % (100 * (df['cond_count'] > 0).sum() / len(df)))
print('  Conduction score range: %.4f ~ %.4f' % (df['cond_score'].min(), df['cond_score'].max()))

# ============================================================
# LightGBM backtest
# ============================================================
print('\nBacktest: 24 factors vs 24+conduction...')

exclude = ['ts_code', 'trade_date', 'close', 'factor_group', '_k', 'report_date',
           'excess_ret']
base_feats = [c for c in df.columns if c not in exclude
              and df[c].dtype in ('float64', 'float32', 'int64', 'int32')
              and c not in ('cond_score', 'cond_count')]

upgrade_feats = base_feats + ['cond_score']

print('  Base: %d features, +Conduction: %d' % (len(base_feats), len(upgrade_feats)))

def backtest(train_df, test_df, feat_list):
    feats = [f for f in feat_list if f in train_df.columns]
    X_tr = train_df[feats].fillna(train_df[feats].median())
    y_tr = train_df['excess_ret'].fillna(0)
    X_te = test_df[feats].fillna(train_df[feats].median())

    m = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                       subsample=0.8, colsample_bytree=0.8,
                       n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    mask = ~np.isnan(pred) & ~np.isnan(test_df['excess_ret'].values)
    ic, _ = stats.spearmanr(pred[mask], test_df['excess_ret'].values[mask])

    te2 = test_df.copy()
    te2['pred'] = pred
    te2['ym'] = pd.to_datetime(te2['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in te2.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        mrets.append(top['excess_ret'].mean())

    if len(mrets) < 3: return {'ic': ic, 'sh': 0, 'mdd': 0}
    rets = np.array(mrets)
    ann = np.mean(rets)*12
    vol = np.std(rets, ddof=1)*np.sqrt(12) if len(rets)>2 else 0.01
    sh = ann/vol if vol>0 else 0
    mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    return {'ic': ic, 'sh': sh, 'mdd': mdd, 'mr': np.mean(rets)}

windows = [
    ('2020->2021', '2017-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
    ('2021->2022', '2018-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
    ('2022->2023', '2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
    ('2023->2024', '2020-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
]

print('  %-14s | %8s %8s %8s | %8s %8s | %s' % (
    'Window', 'BL_IC', 'CD_IC', 'dIC', 'BL_Sh', 'CD_Sh', 'Win?'))
print('  ' + '-' * 75)

all_res = []
for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date']>=tr_s) & (df['trade_date']<=tr_e)]
    te = df[(df['trade_date']>=te_s) & (df['trade_date']<=te_e)]
    if len(tr) < 5000: continue

    bl = backtest(tr, te, base_feats)
    cd = backtest(tr, te, upgrade_feats)
    all_res.append((label, bl, cd))

    dic = cd['ic'] - bl['ic']
    dsh = cd['sh'] - bl['sh']
    winner = 'COND' if cd['ic'] > bl['ic'] else 'BASE'
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f | %s' % (
        label, bl['ic'], cd['ic'], dic, bl['sh'], cd['sh'], winner))

if all_res:
    avg_bl_ic = np.mean([r[1]['ic'] for r in all_res])
    avg_cd_ic = np.mean([r[2]['ic'] for r in all_res])
    avg_bl_sh = np.mean([r[1]['sh'] for r in all_res])
    avg_cd_sh = np.mean([r[2]['sh'] for r in all_res])
    wins = sum(1 for r in all_res if r[2]['ic'] > r[1]['ic'])
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f | Wins: %d/%d' % (
        'AVERAGE', avg_bl_ic, avg_cd_ic, avg_cd_ic-avg_bl_ic,
        avg_bl_sh, avg_cd_sh, wins, len(all_res)))

# ============================================================
# Correct usage: conduction as SELECTION FILTER, not as feature
# ============================================================
print('\n' + '=' * 80)
print('Conduction as SELECTION FILTER (not feature)')
print('=' * 80)

def backtest_filter(train_df, test_df, feat_list, conduction_filter=False):
    feats = [f for f in feat_list if f in train_df.columns]
    X_tr = train_df[feats].fillna(train_df[feats].median())
    y_tr = train_df['excess_ret'].fillna(0)
    X_te = test_df[feats].fillna(train_df[feats].median())

    m = LGBMRegressor(learning_rate=0.05, num_leaves=63, max_depth=10,
                       subsample=0.8, colsample_bytree=0.8,
                       n_estimators=200, verbose=-1, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    ic_mask = ~np.isnan(pred) & ~np.isnan(test_df['excess_ret'].values)
    ic, _ = stats.spearmanr(pred[ic_mask], test_df['excess_ret'].values[ic_mask])

    te2 = test_df.copy()
    te2['pred'] = pred

    # Apply conduction filter
    if conduction_filter:
        # Check if ANY macro signal is active on each date
        # If active, only select from the relevant basket
        for trade_date in te2['trade_date'].unique():
            day_mask = te2['trade_date'] == trade_date
            active_basket_stocks = set()

            for macro_col, basket_name, direction, lag in LINKS:
                # Check if macro triggered today
                if macro_col == 'sox':
                    sox_day = sox_daily[sox_daily['trade_date'] == trade_date]
                    if len(sox_day) > 0:
                        chg = sox_day['sox_chg'].values[0]
                        if not pd.isna(chg) and abs(chg) * 100 > 1.5:
                            active_basket_stocks.update(BASKETS.get(basket_name, []))
                else:
                    col_name = macro_col + '_chg'
                    mac_day = macro_daily[macro_daily['trade_date'] == trade_date]
                    if len(mac_day) > 0:
                        chg = mac_day[col_name].values[0]
                        thresh = 1.5 if macro_col == 'gold' else (2.0 if macro_col == 'copper' else 3.0)
                        if not pd.isna(chg) and abs(chg) * 100 > thresh:
                            active_basket_stocks.update(BASKETS.get(basket_name, []))

            if active_basket_stocks:
                # Boost predictions for active basket stocks (push them into top 30)
                basket_mask = te2['ts_code'].isin(active_basket_stocks)
                if basket_mask.sum() > 5:
                    te2.loc[day_mask & basket_mask, 'pred'] += te2.loc[day_mask, 'pred'].std() * 0.5

    te2['ym'] = pd.to_datetime(te2['trade_date']).dt.to_period('M')
    mrets = []
    for mo, g in te2.groupby('ym'):
        if len(g) < 30: continue
        top = g.nlargest(30, 'pred')
        mrets.append(top['excess_ret'].mean())

    if len(mrets) < 3: return {'ic': ic, 'sh': 0, 'mdd': 0}
    rets = np.array(mrets)
    ann = np.mean(rets)*12
    vol = np.std(rets, ddof=1)*np.sqrt(12) if len(rets)>2 else 0.01
    sh = ann/vol if vol>0 else 0
    mdd = np.min(np.cumprod(1+rets)/np.maximum.accumulate(np.cumprod(1+rets))-1)
    return {'ic': ic, 'sh': sh, 'mdd': mdd}

print('  %-14s | %8s %8s %8s | %8s %8s | %s' % (
    'Window', 'BL_IC', 'FL_IC', 'dIC', 'BL_Sh', 'FL_Sh', 'Win?'))
print('  ' + '-' * 75)

fl_results = []
for label, tr_s, tr_e, te_s, te_e in windows:
    tr = df[(df['trade_date']>=tr_s) & (df['trade_date']<=tr_e)]
    te = df[(df['trade_date']>=te_s) & (df['trade_date']<=te_e)]
    if len(tr) < 5000: continue

    bl = backtest(tr, te, base_feats)
    fl = backtest_filter(tr, te, base_feats, conduction_filter=True)
    fl_results.append((label, bl, fl))

    dic = fl['ic'] - bl['ic']
    dsh = fl['sh'] - bl['sh']
    winner = 'FILTER' if fl['ic'] > bl['ic'] else 'BASE'
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f | %s' % (
        label, bl['ic'], fl['ic'], dic, bl['sh'], fl['sh'], winner))

if fl_results:
    avg_bl = np.mean([r[1]['ic'] for r in fl_results])
    avg_fl = np.mean([r[2]['ic'] for r in fl_results])
    avg_bl_sh = np.mean([r[1]['sh'] for r in fl_results])
    avg_fl_sh = np.mean([r[2]['sh'] for r in fl_results])
    wins = sum(1 for r in fl_results if r[2]['ic'] > r[1]['ic'])
    print('  %-14s | %+.4f %+.4f %+.4f | %8.3f %8.3f | Wins: %d/%d' % (
        'AVERAGE', avg_bl, avg_fl, avg_fl-avg_bl, avg_bl_sh, avg_fl_sh, wins, len(fl_results)))

print('\nDone.')
