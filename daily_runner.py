# -*- coding: utf-8 -*-
"""
QuantLab 每日纸交引擎 v2.0
=============================
三策略并行: ML选股 + ETF轮动 + 小众战法Top30
交易日检测 → 选股 → 更新portfolio → Git存档
"""
import sys, io, os, json, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import akshare as ak
import duckdb, pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
PAPER_DIR = 'D:/AgentQuant/our'
TODAY = date.today()

# ============================================================
# 1. 交易日检测
# ============================================================
def is_trading_day(d):
    try:
        cal = ak.tool_trade_date_hist_sina()
        return d.isoformat() in set(cal['trade_date'].astype(str).values)
    except:
        return d.weekday() < 5

def last_trading_day(d):
    cal = ak.tool_trade_date_hist_sina()
    cal_dates = sorted(cal['trade_date'].astype(str).values)
    for td in reversed(cal_dates):
        if td <= d.isoformat(): return td
    return d.isoformat()

LTD = last_trading_day(TODAY)
IS_TRADING = is_trading_day(TODAY)
print(f'Daily Runner v2.0 — {TODAY.isoformat()}')
print(f'  最近交易日: {LTD} | 今日交易: {IS_TRADING}')

# ============================================================
# 2. ML选股 (backtest_final_production.py 管线)
# ============================================================
def run_stock_paper():
    print('\n[ML选股] 启动...')
    train_start = f'{TODAY.year-5}-01-01'; train_end = LTD
    con = duckdb.connect(DB, read_only=True)

    factors = pd.read_parquet(f'{PAPER_DIR}/cache/factors_2002.parquet')
    factors['trade_date'] = pd.to_datetime(factors['trade_date'])
    factors = factors[(factors['trade_date'] >= train_start) & (factors['trade_date'] <= train_end)]
    factors['trade_date'] = factors['trade_date'].dt.strftime('%Y-%m-%d')

    target = con.execute(f"""
        SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
               (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_20d
        FROM (SELECT ts_code, trade_date, close,
              LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
              FROM kline_daily WHERE trade_date BETWEEN '{train_start}' AND '{LTD}') s
        JOIN (SELECT trade_date, close,
              LEAD(close,20) OVER(ORDER BY trade_date) AS fc
              FROM kline_daily WHERE ts_code='sh000300'
                AND trade_date BETWEEN '{train_start}' AND '{LTD}') x
        ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
    """).df(); target['trade_date'] = target['trade_date'].astype(str)

    mcap = con.execute(f"""
        SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date, close*total_share/10000 AS mcap
        FROM kline_daily WHERE trade_date BETWEEN '{train_start}' AND '{LTD}'
    """).df(); mcap['trade_date'] = mcap['trade_date'].astype(str)

    industry = con.execute("""SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn = 1""").df()
    con.close()

    df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
    df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
    df = df.merge(mcap, on=['ts_code','trade_date'], how='left')

    FEATS = ['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
             'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

    from sklearn.linear_model import LinearRegression
    from lightgbm import LGBMRegressor

    tr = df[df['trade_date'] < LTD].dropna(subset=['excess_ret_20d']).tail(500000).copy()
    te = df[df['trade_date'] == LTD].dropna(subset=['excess_ret_20d']).copy()
    if len(tr) < 10000 or len(te) < 100:
        print(f'  数据不足: tr={len(tr)} te={len(te)}')
        return []

    for d in [tr, te]:
        d['excess_ret_20d'] = d.groupby('trade_date')['excess_ret_20d'].transform(lambda x: x - x.mean())
        d['mcap'] = d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['ln_mcap'] = np.log(d['mcap'].clip(lower=1e6))
        d['ln_mcap_sq'] = d['ln_mcap'] ** 2

    all_inds = sorted(set(tr['ind_name'].unique()) | set(te['ind_name'].unique()))
    ind_map = {ind: i for i, ind in enumerate(all_inds)}
    tr_dum = np.zeros((len(tr), len(all_inds))); te_dum = np.zeros((len(te), len(all_inds)))
    for i, ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i, ind_map[ind]] = 1
    for i, ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i, ind_map[ind]] = 1

    X_tr = np.column_stack([tr['ln_mcap'].values, tr['ln_mcap_sq'].values, tr_dum])
    X_te = np.column_stack([te['ln_mcap'].values, te['ln_mcap_sq'].values, te_dum])
    y_tr_raw = np.nan_to_num(tr[FEATS].fillna(0).values.astype(float), 0)
    y_te_raw = np.nan_to_num(te[FEATS].fillna(0).values.astype(float), 0)

    idx = np.random.choice(X_tr.shape[0], min(50000, X_tr.shape[0]), replace=False)
    Xf, yf = X_tr[idx], y_tr_raw[idx]
    m = LinearRegression(fit_intercept=False); m.fit(Xf, yf)
    res_tr = y_tr_raw - X_tr @ m.coef_.T; res_te = y_te_raw - X_te @ m.coef_.T

    neu_names = []
    for i, c in enumerate(FEATS):
        name = c + '_n'; tr[name]=res_tr[:,i]; te[name]=res_te[:,i]
        mu, std = tr[name].mean(), tr[name].std()
        if std>0: tr[name]=(tr[name]-mu)/std; te[name]=(te[name]-mu)/std
        neu_names.append(name)

    flist = [f for f in neu_names if f in tr.columns]
    X_tr_f = tr[flist].fillna(0).values.astype(float)
    X_te_f = te[flist].fillna(0).values.astype(float)

    y1 = tr['excess_ret_20d'].fillna(0).values
    y2 = tr.groupby('trade_date')['excess_ret_20d'].rank(pct=True).fillna(0.5).values

    m1 = LGBMRegressor(n_estimators=120, num_leaves=31, max_depth=6, learning_rate=0.03,
                        subsample=0.8, reg_alpha=0.2, reg_lambda=0.2,
                        min_child_samples=50, verbose=-1, n_jobs=-1).fit(X_tr_f, y1)
    m2 = LGBMRegressor(n_estimators=120, num_leaves=31, max_depth=6, learning_rate=0.03,
                        subsample=0.8, reg_alpha=0.2, reg_lambda=0.2,
                        min_child_samples=50, verbose=-1, n_jobs=-1).fit(X_tr_f, y2)

    te['pred'] = (m1.predict(X_te_f)-y1.mean())/(y1.std() or 1) + \
                 (m2.predict(X_te_f)-0.5)/0.3

    te['mom_20d'] = te.groupby('ts_code')['excess_ret_20d'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    te['mom_60d'] = te.groupby('ts_code')['excess_ret_20d'].transform(lambda x: x.rolling(60, min_periods=5).mean())
    te.loc[(te['mom_20d'].fillna(0)-te['mom_60d'].fillna(0))<0, 'pred'] *= 0.8

    te['mcap_r'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[te['mcap_r'] >= 0.20]
    te_f = te_f.nlargest(20, 'pred')

    picks = []
    for _, row in te_f.iterrows():
        picks.append({'ts_code': row['ts_code'], 'pred': round(float(row['pred']), 3),
                       'mcap': round(float(row['mcap']), 0), 'ind_name': row['ind_name']})
    print(f'  选出 {len(picks)} 只')
    return picks[:15]

# ============================================================
# 3. ETF轮动
# ============================================================
def run_etf_paper():
    print('\n[ETF轮动] 启动...')
    con = duckdb.connect(DB, read_only=True)
    hs300 = con.execute("""
        SELECT trade_date, close FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date >= '2025-01-01' ORDER BY trade_date
    """).df(); con.close()

    if len(hs300) < 200: return []
    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
    px = hs300.set_index('trade_date')['close']
    ma200 = px.rolling(200).mean().iloc[-1]
    latest_px = px.iloc[-1]
    is_bull = latest_px > ma200
    print(f'  CSI300: {latest_px:.0f} vs MA200: {ma200:.0f} → {"BULL" if is_bull else "BEAR"}')
    return ['进攻:电子ETF','通信ETF','计算机ETF','国防ETF','机械ETF'] if is_bull \
           else ['防御:国债ETF(511010)','黄金ETF(518880)','纳指ETF(513100)']

# ============================================================
# 4. 小众战法 Top30 (NEW v2.0)
# ============================================================
XIAOZHONG_FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
XIAOZHONG_PAIRS = [
    ('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')
]
# 固定选对: 基于2021-2025 Walk-Forward训练 (IR排序前4)
XIAOZHONG_TOP4 = [('amihud','turnover_rev'),('amihud','max_rev'),('amihud','sr5'),('turnover_rev','sr5')]

def dd_smart_gate():
    """DD_SMART门禁: 沪深300为唯一基准"""
    con = duckdb.connect(DB, read_only=True)
    hs300 = con.execute("""
        SELECT trade_date, close,
               AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma50,
               MAX(close) OVER(ORDER BY trade_date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW) AS high_2y,
               MIN(close) OVER(ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS low_1y
        FROM kline_daily WHERE ts_code='sh000300'
        ORDER BY trade_date DESC LIMIT 1
    """).fetchone()
    con.close()

    if not hs300: return 1.0, 'unknown'
    close, ma50, high_2y, low_1y = hs300
    dd_2y = close/high_2y - 1 if high_2y else 0
    recovery = close/low_1y - 1 if low_1y else 0
    above_ma50 = close > ma50 if ma50 else False

    if dd_2y < -0.20: return 0.2, f'深熊(dd={dd_2y*100:.0f}%)'
    elif dd_2y < -0.15: return 0.4, f'回撤(dd={dd_2y*100:.0f}%)'
    elif above_ma50: return 1.0, f'满仓(MA50之上,dd={dd_2y*100:.0f}%)'
    elif recovery > 0.10: return 0.7, f'回场中(rec={recovery*100:.0f}%)'
    else: return 0.5, f'观望'

def run_xiaozhong_paper():
    """小众战法: 6因子乘法 × Top30"""
    print('\n[小众战法 Top30] 启动...')

    gate_pos, gate_msg = dd_smart_gate()
    print(f'  DD_SMART: {gate_msg} → 仓位{gate_pos*100:.0f}%')

    fn = pd.read_parquet(f'{PAPER_DIR}/cache/factors_orig6f_2002.parquet')
    fn['trade_date'] = pd.to_datetime(fn['trade_date'])
    day = fn[fn['trade_date'] == LTD].copy()

    if len(day) < 200:
        print(f'  因子数据不足: {len(day)}只')
        return [], gate_pos

    con = duckdb.connect(DB, read_only=True)
    kline_today = con.execute(f"""
        SELECT ts_code, close, COALESCE(amount, GREATEST(vol*close,1.0)) AS amount_proxy,
               close/pre_close-1 AS change_pct,
               close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
        FROM kline_daily WHERE trade_date = '{LTD}'
    """).df(); con.close()

    day = day.merge(kline_today.rename(columns={'close':'price','amount_proxy':'mcap_proxy'}), on='ts_code', how='inner')
    print(f'  因子+价格: {len(day)}只')

    # 因子排名+乘法
    all_f = list(set([x for p in XIAOZHONG_TOP4 for x in p]))
    for f in all_f:
        if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

    day['score'] = 0
    for fa,fb in XIAOZHONG_TOP4:
        if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
            day['score'] += day[f'{fa}_r']*day[f'{fb}_r']

    # 风控
    day['mcap_r'] = day['mcap_proxy'].rank(pct=True)
    day['lim_chk'] = day['change_pct'].fillna(day['ret_1d'])
    day = day[day['mcap_r'] >= 0.20]
    day = day[day['lim_chk'].notna() & (day['lim_chk'] < 0.095)]
    day = day[day['price'] > 0]; day = day[day['mcap_proxy'] > 0]

    if len(day) < 60: return [], gate_pos

    top = day.nlargest(30, 'score')
    picks = []
    for _, row in top.iterrows():
        picks.append({
            'ts_code': row['ts_code'],
            'score': round(float(row['score']), 4),
            'price': round(float(row['price']), 2),
            'mcap_proxy': round(float(row['mcap_proxy']), 0)
        })
    print(f'  选出 {len(picks)} 只 | 门禁: {gate_pos*100:.0f}%')
    return picks, gate_pos

# ============================================================
# 5. 主逻辑
# ============================================================
if not IS_TRADING:
    print(f'\n{TODAY} 非交易日, 跳过')
    sys.exit(0)

stock_picks = run_stock_paper()
etf_picks = run_etf_paper()
xz_picks, xz_gate = run_xiaozhong_paper()

report = {
    'date': LTD,
    'generated': datetime.now().isoformat(),
    'strategies': {
        'ml_stock': {
            'name': 'ML选股 vFinal',
            'picks': stock_picks,
            'count': len(stock_picks)
        },
        'etf_rotation': {
            'name': 'ETF轮动 v2.4',
            'picks': etf_picks,
            'count': len(etf_picks)
        },
        'xiaozhong_top30': {
            'name': '小众战法 Top30',
            'gate_position': xz_gate,
            'gate_msg': dd_smart_gate()[1],
            'pairs': [f'{a[:4]}x{b[:4]}' for a,b in XIAOZHONG_TOP4],
            'picks': xz_picks,
            'count': len(xz_picks)
        }
    }
}

with open(f'{PAPER_DIR}/paper_daily_signal.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'\n{"="*50}')
print(f'纸交信号已保存')
print(f'  ML选股:    {len(stock_picks)}只')
print(f'  ETF轮动:   {len(etf_picks)}项')
print(f'  小众战法:  {len(xz_picks)}只 (仓位{xz_gate*100:.0f}%)')
print(f'  信号文件:  paper_daily_signal.json')
