# -*- coding: utf-8 -*-
"""
ML选股 · 严格Walk-Forward修复版
================================
修复:
  1. 目标: 月度调仓收益(非20日重叠窗口)
  2. Purge: 训练/测试间+1月隔离(Lopez de Prado)
  3. OLS中性化: 仅在训练集拟合
  4. Walk-Forward: 5年训练→1年OOS, 滚动
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
PURGE_MONTHS = 1  # Lopez de Prado: 训练/测试间隔1月

print("=" * 60)
print("ML选股 · 严格Walk-Forward修复版")
print("=" * 60)

# ============ 加载 ============
print("[1] 加载...")
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, vol,
           COALESCE(amount, GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000, GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date >= '2002-01-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2002-01-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
con.close()

# 月度调仓日
dates = sorted(kline['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print("月度: %d个" % len(monthly_dates))

# ============ 构建月度因子+目标数据集 ============
print("[2] 构建月度数据集...")

def compute_ta_factors(df_stock):
    """从日K线计算12个TA因子, 只用到当日及之前的数据"""
    close = df_stock['close'].values
    vol = df_stock['vol'].values
    high = df_stock['high'].values
    low = df_stock['low'].values

    n = len(close)
    if n < 120: return None

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0); loss = -np.minimum(delta, 0)
    avg_gain_6 = pd.Series(gain).rolling(6).mean().values
    avg_loss_6 = pd.Series(loss).rolling(6).mean().values
    rs6 = avg_gain_6 / np.maximum(avg_loss_6, 1e-6)
    rsi6 = 100 - 100/(1+rs6)
    avg_gain_14 = pd.Series(gain).rolling(14).mean().values
    avg_loss_14 = pd.Series(loss).rolling(14).mean().values
    rs14 = avg_gain_14 / np.maximum(avg_loss_14, 1e-6)
    rsi14 = 100 - 100/(1+rs14)

    # MA
    ma20 = pd.Series(close).rolling(20).mean().values
    ma60 = pd.Series(close).rolling(60).mean().values
    ma120 = pd.Series(close).rolling(120).mean().values

    # Bollinger
    std20 = pd.Series(close).rolling(20).std().values
    boll_pos = np.clip((close - (ma20 - 2*std20)) / np.maximum(4*std20, 1e-6), 0, 1)
    boll_width = (4*std20) / np.maximum(ma20, 1e-6)

    # Divergence
    div_ma20 = close / np.maximum(ma20, 1e-6) - 1
    div_ma60 = close / np.maximum(ma60, 1e-6) - 1
    div_ma120 = close / np.maximum(ma120, 1e-6) - 1

    # Volume
    vol_ma20 = pd.Series(vol).rolling(20).mean().values
    vol_ratio = vol / np.maximum(vol_ma20, 1)

    # MA score
    ma_score = ((close > ma20).astype(int) + (close > ma60).astype(int) + (close > ma120).astype(int)) / 3

    # RSI extreme
    rsi_extreme = np.abs(rsi14 - 50) / 50

    # Margin panic (volatility of recent losses)
    margin_panic = pd.Series(-delta).rolling(5).std().values

    # Streak
    streak5_dn = pd.Series(delta < 0).rolling(5).sum().values

    return {
        'rsi6': rsi6[-1], 'rsi14': rsi14[-1], 'boll_pos': boll_pos[-1], 'boll_width': boll_width[-1],
        'div_ma20': div_ma20[-1], 'div_ma60': div_ma60[-1], 'div_ma120': div_ma120[-1],
        'vol_ratio': vol_ratio[-1], 'ma_score': ma_score[-1], 'rsi_extreme': rsi_extreme[-1],
        'margin_panic': margin_panic[-1], 'streak5_dn': streak5_dn[-1]
    }

# 月度因子数据构建
monthly_data = []
for i, rd in enumerate(monthly_dates):
    # 当天数据
    day_kline = kline[kline['trade_date'] == rd]
    if len(day_kline) < 100: continue

    factors_for_date = []
    for code, grp in kline[kline['trade_date'] <= rd].groupby('ts_code'):
        if code not in day_kline['ts_code'].values: continue
        stock_data = grp.sort_values('trade_date').tail(150)  # 最近150天
        if len(stock_data) < 60: continue
        factors = compute_ta_factors(stock_data)
        if factors is None: continue
        factors['ts_code'] = code
        factors['trade_date'] = rd
        factors['close'] = float(stock_data['close'].iloc[-1])
        factors['mcap'] = float(stock_data['mcap'].iloc[-1]) if 'mcap' in stock_data.columns else 1e6
        factors['ret_1d'] = float(stock_data['ret_1d'].iloc[-1]) if 'ret_1d' in stock_data.columns else 0
        factors_for_date.append(factors)

    if i % 50 == 0:
        print("   %d/%d (%s) %d stocks" % (i, len(monthly_dates), rd.date(), len(factors_for_date)))

    monthly_data.extend(factors_for_date)

factors_df = pd.DataFrame(monthly_data)
print("月度因子: %d行, %d个月" % (len(factors_df), factors_df['trade_date'].nunique()))

# ============ 构建月度目标 ============
print("[3] 构建月度目标(下一调仓日收益)...")
# 价格映射: 当月close → 下月open
price_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    price_map[cur] = m

# 合并目标
targets = []
for rd in monthly_dates:
    if rd not in price_map: continue
    px = price_map[rd]
    targets.append(px[['fwd_ret']].reset_index().assign(trade_date=rd))

target_df = pd.concat(targets)
target_df['trade_date'] = pd.to_datetime(target_df['trade_date'])
print("目标: %d行" % len(target_df))

del kline; gc.collect()

# ============ 合并因子+目标 ============
full_df = factors_df.merge(target_df, on=['ts_code','trade_date'], how='inner')
full_df = full_df.dropna(subset=['fwd_ret'])
print("合并: %d行" % len(full_df))

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS + 1  # +1 for purge

FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']

print("\n[4] Walk-Forward (%d-%d)..." % (FIRST_TEST_YR, YEARS[-1]))

all_results = []
yearly_perf = []

for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    # 训练期: test_yr-TRAIN_YEARS 到 test_yr-1(不含), 再加1月purge
    train_end = pd.Timestamp('%d-01-01' % test_yr) - pd.DateOffset(months=PURGE_MONTHS+1)
    train_start = pd.Timestamp('%d-01-01' % (test_yr - TRAIN_YEARS))
    test_start = pd.Timestamp('%d-01-01' % test_yr)
    test_end = pd.Timestamp('%d-12-31' % test_yr)

    tr = full_df[(full_df['trade_date'] >= train_start) & (full_df['trade_date'] < train_end)].copy()
    te = full_df[(full_df['trade_date'] >= test_start) & (full_df['trade_date'] <= test_end)].copy()

    if len(tr) < 5000 or len(te) < 500: continue

    # OLS中性化: 仅在训练集拟合
    for d in [tr, te]:
        d['mcap_fill'] = d['mcap'].fillna(d['mcap'].median()) if 'mcap' in d.columns else 1e6
        d['ln_mcap'] = np.log(d['mcap_fill'].clip(lower=1e6))

    X_ols_tr = np.column_stack([tr['ln_mcap'].values])
    X_ols_te = np.column_stack([te['ln_mcap'].values])
    ols = LinearRegression()

    neu_feats = []
    for f in FEATS:
        if f not in tr.columns: continue
        y_tr = tr[f].fillna(0).values
        y_te = te[f].fillna(0).values
        ols.fit(X_ols_tr, y_tr)
        tr[f+'_n'] = y_tr - ols.predict(X_ols_tr)
        te[f+'_n'] = y_te - ols.predict(X_ols_te)
        # 用训练集的mu/std标准化
        mu, std = tr[f+'_n'].mean(), tr[f+'_n'].std()
        if std > 0:
            tr[f+'_n'] = (tr[f+'_n'] - mu) / std
            te[f+'_n'] = (te[f+'_n'] - mu) / std
        neu_feats.append(f+'_n')

    # LightGBM
    X_tr = tr[neu_feats].fillna(0).values.astype(float)
    y_tr = tr['fwd_ret'].fillna(0).values
    X_te = te[neu_feats].fillna(0).values.astype(float)

    # 样本限制防过拟合
    if len(X_tr) > 100000:
        idx = np.random.choice(len(X_tr), 100000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    # 训练集内再分valid做early stopping
    split = int(len(X_tr) * 0.8)
    X_t, X_v = X_tr[:split], X_tr[split:]
    y_t, y_v = y_tr[:split], y_tr[split:]

    model = LGBMRegressor(
        n_estimators=500, num_leaves=15, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
        min_child_samples=100, verbose=-1, n_jobs=-1,
        early_stopping_rounds=30
    )
    model.fit(X_t, y_t, eval_set=[(X_v, y_v)])
    best_iter = model.best_iteration_ if model.best_iteration_ else 100

    # 重训(全量训练集, best_iter)
    model2 = LGBMRegressor(
        n_estimators=best_iter, num_leaves=15, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
        min_child_samples=100, verbose=-1, n_jobs=-1
    )
    model2.fit(X_tr, y_tr)
    te['pred'] = model2.predict(X_te)

    # 选股
    te['mcap_r'] = te.groupby('trade_date')['mcap_fill'].rank(pct=True)
    te_f = te[te['mcap_r'] >= MCAP_FLOOR]
    te_f = te_f[te_f['ret_1d'].fillna(0) < LIMIT_UP]

    fold_rets = []
    for rd in sorted(te_f['trade_date'].unique()):
        day_te = te_f[te_f['trade_date'] == rd]
        if len(day_te) < 50: continue
        top = day_te.nlargest(TOP_N, 'pred')
        if len(top) < 5: continue
        month_ret = top['fwd_ret'].mean() - COST
        fold_rets.append(month_ret)
        all_results.append({'date': str(rd)[:7], 'ret': month_ret, 'yr': rd.year, 'n': len(top)})

    if fold_rets:
        r = np.array(fold_rets)
        ann = np.mean(r)*12; vol = np.std(r)*np.sqrt(12)
        sh = ann/vol if vol>0 else 0; mdd = np.min(np.cumprod(1+r)/np.maximum.accumulate(np.cumprod(1+r))-1)
        yearly_perf.append({'yr': test_yr, 'ann': ann, 'sharpe': sh, 'mdd': mdd, 'months': len(r)})
        print("  %d: ann=%+.1f%% sh=%+.2f mdd=%.1f%% best_iter=%d" % (test_yr, ann*100, sh, mdd*100, best_iter))

# ============ 总结 ============
r_all = np.array([x['ret'] for x in all_results])
ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
sh = ann/vol if vol>0 else 0; mdd = np.min(np.cumprod(1+r_all)/np.maximum.accumulate(np.cumprod(1+r_all))-1)
win = (r_all>0).mean()*100; calmar = ann/abs(mdd) if mdd!=0 else 0
total = np.prod(1+r_all)-1

print("\n" + "="*60)
print("ML修复版 终验")
print("="*60)
print("年化: %+.1f%% | Sharpe: %+.2f | MDD: %.1f%% | Calmar: %+.2f" % (ann*100, sh, mdd*100, calmar))
print("胜率: %.0f%% | 累计: %+.0f%% | 月数: %d" % (win, total*100, len(r_all)))

print("\n年      收益   Sharpe    MDD")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    dr = [x['ret'] for x in all_results if x['yr']==yr]
    if len(dr)>=6:
        rr = np.array(dr); a = np.mean(rr)*12; v = np.std(rr)*np.sqrt(12)
        s = a/v if v>0 else 0; m = np.min(np.cumprod(1+rr)/np.maximum.accumulate(np.cumprod(1+rr))-1)
        print("%d %+7.1f%% %+6.2f %+6.1f%%" % (yr, (np.prod(1+rr)-1)*100, s, m*100))

print("\n耗时: %.0fs" % (time.time()-t0))
