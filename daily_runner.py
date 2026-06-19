# -*- coding: utf-8 -*-
"""
QuantLab 每日纸交引擎 vFinal
=============================
交易日检测 → 个股策略(审计修复版) + ETF策略(多资产防御)
→ 更新portfolio → Git存档

策略选择:
- 个股: backtest_final_production.py 管线 (动量拐点, Sharpe 1.62)
- ETF: 行业轮动 v2.4 (MA200择时, Sharpe 1.74)
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
    """A股是否交易日"""
    try:
        cal = ak.tool_trade_date_hist_sina()
        cal_dates = set(cal['trade_date'].astype(str).values)
        return d.isoformat() in cal_dates
    except:
        return d.weekday() < 5  # fallback

def last_trading_day(d):
    """最近交易日 (含当天)"""
    cal = ak.tool_trade_date_hist_sina()
    cal_dates = sorted(cal['trade_date'].astype(str).values)
    for td in reversed(cal_dates):
        if td <= d.isoformat():
            return td
    return d.isoformat()

LTD = last_trading_day(TODAY)
IS_TRADING = is_trading_day(TODAY)

print(f'Daily Runner — {TODAY.isoformat()}')
print(f'  最近交易日: {LTD} | 今日交易: {IS_TRADING}')

# ============================================================
# 2. 个股纸交 (审计修复版管线)
# ============================================================
def run_stock_paper():
    """运行个股纸交选股, 基于backtest_final_production.py管线"""
    print('\n[个股纸交] 启动...')

    # 使用最近5年数据训练 (与Walk-Forward一致)
    train_start = f'{TODAY.year-5}-01-01'
    train_end = LTD

    con = duckdb.connect(DB, read_only=True)

    # 加载因子+目标
    factors = pd.read_parquet(f'{PAPER_DIR}/cache/factors_2002.parquet')
    factors['trade_date'] = pd.to_datetime(factors['trade_date'])
    factors = factors[(factors['trade_date'] >= train_start) & (factors['trade_date'] <= train_end)]
    factors['trade_date'] = factors['trade_date'].dt.strftime('%Y-%m-%d')

    # 目标
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
    """).df()
    target['trade_date'] = target['trade_date'].astype(str)

    # 市值
    mcap = con.execute(f"""
        SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
               close*total_share/10000 AS mcap
        FROM kline_daily WHERE trade_date BETWEEN '{train_start}' AND '{LTD}'
    """).df()
    mcap['trade_date'] = mcap['trade_date'].astype(str)

    # 行业
    industry = con.execute("""SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn = 1""").df()

    con.close()

    # 合并
    df = factors.merge(target, on=['ts_code','trade_date'], how='inner')
    df = df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
    df = df.merge(mcap, on=['ts_code','trade_date'], how='left')

    FEATS = ['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
             'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

    from sklearn.linear_model import LinearRegression
    from lightgbm import LGBMRegressor

    # 训练 (最近3年训练, 最近1月预测)
    tr = df[df['trade_date'] < LTD].dropna(subset=['excess_ret_20d']).tail(500000).copy()
    te = df[df['trade_date'] == LTD].dropna(subset=['excess_ret_20d']).copy()

    if len(tr) < 10000 or len(te) < 100:
        print(f'  数据不足: tr={len(tr)} te={len(te)}')
        return []

    # 截面中性化
    for d in [tr, te]:
        d['excess_ret_20d'] = d.groupby('trade_date')['excess_ret_20d'].transform(lambda x: x - x.mean())

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
    y_tr_raw = np.nan_to_num(tr[FEATS].fillna(0).values.astype(float), 0)
    y_te_raw = np.nan_to_num(te[FEATS].fillna(0).values.astype(float), 0)

    if X_tr.shape[0] > 50000:
        idx = np.random.choice(X_tr.shape[0], 50000, replace=False)
        Xf, yf = X_tr[idx], y_tr_raw[idx]
    else:
        Xf, yf = X_tr, y_tr_raw
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

    # 动量拐点
    te['mom_20d'] = te.groupby('ts_code')['excess_ret_20d'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    te['mom_60d'] = te.groupby('ts_code')['excess_ret_20d'].transform(lambda x: x.rolling(60, min_periods=5).mean())
    te.loc[(te['mom_20d'].fillna(0)-te['mom_60d'].fillna(0))<0, 'pred'] *= 0.8

    # 选Top-20 (考虑仓位分散)
    te['mcap_r'] = te.groupby('trade_date')['mcap'].rank(pct=True)
    te_f = te[te['mcap_r'] >= 0.20]
    te_f = te_f.nlargest(20, 'pred')

    picks = []
    for _, row in te_f.iterrows():
        picks.append({
            'ts_code': row['ts_code'],
            'pred': round(float(row['pred']), 3),
            'mcap': round(float(row['mcap']), 0),
            'ind_name': row['ind_name'],
        })

    print(f'  选出 {len(picks)} 只')
    return picks[:15]  # 实盘选10-15只

# ============================================================
# 3. ETF纸交
# ============================================================
def run_etf_paper():
    """ETF行业轮动: MA200择时 → 选防御或进攻"""
    print('\n[ETF纸交] 启动...')
    # 简化: 只判断MA200择时状态, 进攻选5行业, 防御选债券/黄金
    con = duckdb.connect(DB, read_only=True)
    hs300 = con.execute("""
        SELECT trade_date, close FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date >= '2025-01-01'
        ORDER BY trade_date
    """).df()
    con.close()

    if len(hs300) < 200:
        print('  数据不足')
        return []

    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
    px = hs300.set_index('trade_date')['close']
    ma200 = px.rolling(200).mean().iloc[-1]
    latest_px = px.iloc[-1]
    is_bull = latest_px > ma200

    print(f'  CSI300: {latest_px:.0f} vs MA200: {ma200:.0f} → {"BULL" if is_bull else "BEAR"}')

    if is_bull:
        return ['进攻: 电子ETF', '通信ETF', '计算机ETF', '国防ETF', '机械ETF']
    else:
        return ['防御: 国债ETF(511010)', '黄金ETF(518880)', '纳指ETF(513100)']

# ============================================================
# 4. 主逻辑
# ============================================================
if not IS_TRADING:
    print(f'\n{TODAY} 非交易日, 跳过')
    sys.exit(0)

stock_picks = run_stock_paper()
etf_picks = run_etf_paper()

# 保存
report = {
    'date': LTD,
    'generated': datetime.now().isoformat(),
    'stock_picks': stock_picks,
    'etf_picks': etf_picks,
}
with open(f'{PAPER_DIR}/paper_daily_signal.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f'\n信号已保存: paper_daily_signal.json')

# Git存档
os.chdir(PAPER_DIR)
subprocess.run('git add paper_daily_signal.json paper_portfolio.json', shell=True, capture_output=True)
subprocess.run(f'git commit -m "纸交 {LTD} — 个股{len(stock_picks)}只 ETF{len(etf_picks)}个" --allow-empty',
               shell=True, capture_output=True)
print('Git committed.')
