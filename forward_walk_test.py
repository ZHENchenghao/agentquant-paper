# -*- coding: utf-8 -*-
"""
前向验证: 随机10个历史起点，模拟纸交60天。
每个起点只用到当天为止的数据，之后60天纯跟踪。
"""
import sys, io, os, json, time
sys.path.insert(0, 'D:/AgentQuant/our')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import duckdb, numpy as np, pandas as pd
from datetime import date, timedelta
from lightgbm import LGBMRegressor
from quant_backtest_engine import RiskNeutralizer

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
CACHE = 'D:/AgentQuant/our/cache/factors_all.parquet'
N_TESTS = 10
TOP_N = 30
HORIZON = 60  # 跟踪60个交易日

def get_db():
    for i in range(5):
        try:
            c = duckdb.connect(DB, read_only=True); c.execute('SELECT 1'); return c
        except: time.sleep(min(2**i,10))
    return duckdb.connect(DB, read_only=True)

def sql(c, q, label=''):
    for a in range(3):
        try: return c.execute(q).df()
        except Exception as e:
            if a==2: print(f'  ⚠ [{label}] {str(e)[:80]}'); return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

# ── 加载历史因子+目标 ──
print('加载数据...', end=' ', flush=True)
data = pd.read_parquet(CACHE)
c = get_db()
target = sql(c, """
    WITH sf AS (
        SELECT ts_code, trade_date, close,
               LEAD(close,60) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fwd_close
        FROM kline_daily WHERE trade_date BETWEEN '2015-01-01' AND '2026-06-16'
    ),
    xf AS (
        SELECT trade_date, close,
               LEAD(close,60) OVER(ORDER BY trade_date) AS fwd_close
        FROM kline_daily WHERE ts_code='sh000300' AND trade_date BETWEEN '2015-01-01' AND '2026-06-16'
    )
    SELECT s.ts_code, s.trade_date,
           (s.fwd_close/s.close-1)-(x.fwd_close/x.close-1) AS excess_ret
    FROM sf s JOIN xf x ON s.trade_date=x.trade_date
    WHERE s.fwd_close IS NOT NULL
""", 'target')
c.close()

data = data.merge(target, on=['ts_code','trade_date'], how='inner')
print(f'{len(data)}行')

# 特征列
feat_cols = [c for c in data.columns if c not in
             ('ts_code','trade_date','excess_ret','fwd_ret','factor_group','report_date','_k','close')]

# ── 随机选10个起点 ──
all_dates = sorted(data['trade_date'].unique())
# 只选有足够历史数据且能跟踪60天的日期
valid_starts = []
for d in all_dates:
    dt = pd.Timestamp(d)
    # 至少3年历史
    if dt < pd.Timestamp('2018-01-01'): continue
    # 不能太近(要能跟踪60天)
    if dt > pd.Timestamp('2026-03-01'): continue
    valid_starts.append(d)

np.random.seed(42)
test_dates = sorted(np.random.choice(valid_starts, N_TESTS, replace=False))
print(f'\n10个测试起点:')
for d in test_dates: print(f'  {d}')

# ── 逐起点模拟 ──
print(f'\n{"─"*75}')
print(f'  {"起点":12s} {"选股数":>6s} {"60日后":12s} {"策略收益":>8s} {"基准收益":>8s} {"超额":>8s} {"胜率":>6s}')
print(f'  {"─"*75}')

results = []
tracking_curves = []

for i, start_date in enumerate(test_dates):
    # 训练: 只用start_date之前的数据
    train_data = data[data['trade_date'] <= start_date].dropna(subset=feat_cols+['excess_ret'])
    if len(train_data) < 5000:
        print(f'  {start_date}  ⚠ 训练样本不足')
        continue

    # 中性化
    neut = RiskNeutralizer()
    neut_factors = [c for c in feat_cols if c != 'log_mcap']
    if 'log_mcap' in train_data.columns:
        train_data['ind_code'] = train_data['ts_code'].astype(str).str[:4]
        for col in neut_factors:
            if col not in train_data.columns: continue
            fv = neut.industry_neutralize(train_data[col].values, train_data['ind_code'].values)
            fv_q, _, _, _ = neut.size_neutralize_quadratic(fv, train_data['log_mcap'].values)
            train_data[col] = fv_q

    X_train = train_data[feat_cols].fillna(train_data[feat_cols].median())
    y_train = train_data['excess_ret']

    model = LGBMRegressor(
        objective='regression', metric='rmse',
        learning_rate=0.05, num_leaves=63, max_depth=10,
        min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        n_estimators=300, verbose=-1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 预测当天所有股票
    test_mask = data['trade_date'] == start_date
    X_test = data.loc[test_mask, feat_cols].copy()
    if len(X_test) < 100: continue

    # 同样中性化
    if 'log_mcap' in data.columns:
        data.loc[test_mask, 'ind_code'] = data.loc[test_mask, 'ts_code'].astype(str).str[:4]
        for col in neut_factors:
            if col not in data.columns: continue
            fv = neut.industry_neutralize(data.loc[test_mask, col].values,
                                          data.loc[test_mask, 'ind_code'].values)
            fv_q, _, _, _ = neut.size_neutralize_quadratic(
                fv, data.loc[test_mask, 'log_mcap'].values)
            data.loc[test_mask, col] = fv_q

    X_test_filled = X_test.fillna(train_data[feat_cols].median())
    # 市值底线
    if 'log_mcap' in data.columns:
        mcap_floor = data.loc[test_mask, 'log_mcap'].quantile(0.20)
        valid_idx = X_test_filled.index[data.loc[test_mask, 'log_mcap'] >= mcap_floor]
        X_test_filled = X_test_filled.loc[valid_idx]
        test_mask_orig = test_mask.copy()
        test_mask = pd.Series(False, index=test_mask_orig.index)
        test_mask.loc[valid_idx] = True

    scores = model.predict(X_test_filled)
    pred_df = data.loc[test_mask, ['ts_code']].copy()
    pred_df['ml_score'] = scores
    top_n = pred_df.drop_duplicates(subset=['ts_code']).nlargest(TOP_N, 'ml_score')

    # ── 追踪60日 ──
    start_dt = pd.Timestamp(start_date)
    c2 = get_db()
    # 找start_date后第60个交易日
    all_trading = sql(c2, f"""
        SELECT trade_date FROM kline_daily WHERE ts_code='sh000300'
        AND trade_date > '{start_date}' ORDER BY trade_date
    """, 'dates')
    if len(all_trading) < HORIZON:
        c2.close(); continue
    end_date = all_trading['trade_date'].iloc[HORIZON-1]

    # 期间每日净值
    track_dates = [start_date] + all_trading['trade_date'].iloc[:HORIZON].tolist()
    selected_codes = top_n['ts_code'].tolist()

    # 获取每日价格
    codes_str = ','.join(["'%s'" % x for x in selected_codes])
    price_sql = """
        SELECT ts_code, trade_date, close FROM kline_daily
        WHERE ts_code IN (%s)
        AND trade_date >= '%s' AND trade_date <= '%s'
        ORDER BY ts_code, trade_date
    """ % (codes_str, start_date, end_date)
    prices = sql(c2, price_sql, 'prices')

    # 获取基准
    bench = sql(c2, f"""
        SELECT trade_date, close FROM kline_daily WHERE ts_code='sh000300'
        AND trade_date >= '{start_date}' AND trade_date <= '{end_date}' ORDER BY trade_date
    """, 'bench')

    # 模拟等权持仓
    curve = []
    if not prices.empty and not bench.empty:
        price_pivot = prices.pivot(index='trade_date', columns='ts_code', values='close')
        # 填充缺失值(停牌→前向填充)
        price_pivot = price_pivot.ffill()

        bench_vals = bench.set_index('trade_date')['close']
        nav = 1.0
        bench_nav = 1.0

        for j, d in enumerate(track_dates):
            if d not in price_pivot.index:
                if j > 0 and curve:
                    curve.append({'day': j, 'nav': curve[-1]['nav'], 'bench': curve[-1]['bench'],
                                  'trade_date': d})
                continue
            day_prices = price_pivot.loc[d]
            valid_px = day_prices[day_prices > 0]
            if len(valid_px) > 0:
                nav = valid_px.mean() / price_pivot.iloc[0].mean()
            if d in bench_vals.index:
                bench_nav = bench_vals[d] / bench_vals.iloc[0]
            curve.append({'day': j, 'nav': nav, 'bench': bench_nav, 'trade_date': d})

    c2.close()

    # 计算指标
    if curve:
        strat_ret = curve[-1]['nav'] / curve[0]['nav'] - 1
        bench_ret = curve[-1]['bench'] / curve[0]['bench'] - 1
        excess = strat_ret - bench_ret

        # 胜率: 有多少天nav在涨
        daily_chg = np.diff([c['nav'] for c in curve])
        win_rate = np.mean(daily_chg > 0)

        results.append({
            'start': start_date, 'end': str(end_date),
            'n_stocks': len(selected_codes),
            'strat_ret': strat_ret, 'bench_ret': bench_ret, 'excess': excess,
            'win_rate': win_rate
        })
        tracking_curves.append(curve)

        sign = '+' if excess > 0 else ''
        print(f'  {str(start_date):12s} {len(selected_codes):>6d}  {str(end_date):12s}  '
              f'{strat_ret:>+7.1%}  {bench_ret:>+7.1%}  {sign}{excess:>7.1%}  {win_rate:>5.0%}')

# ── 汇总 ──
print(f'\n{"═"*75}')
if results:
    wins = sum(1 for r in results if r['excess'] > 0)
    avg_excess = np.mean([r['excess'] for r in results])
    avg_wr = np.mean([r['win_rate'] for r in results])
    avg_strat = np.mean([r['strat_ret'] for r in results])
    print(f'  {len(results)}/{N_TESTS}次有效')
    print(f'  跑赢基准: {wins}/{len(results)}次 ({wins/len(results)*100:.0f}%)')
    print(f'  平均超额: {avg_excess:+.1%}')
    print(f'  平均策略收益: {avg_strat:+.1%}')
    print(f'  平均日胜率: {avg_wr:.0%}')

    if wins >= len(results) * 0.6:
        print(f'\n  ✅ 可靠: {wins}/{len(results)}次跑赢, 不是运气')
    elif wins >= len(results) * 0.4:
        print(f'\n  ⚠ 一般: {wins}/{len(results)}次跑赢, 部分市场有效')
    else:
        print(f'\n  ❌ 不可靠: 仅{wins}/{len(results)}次跑赢, 需排查')
print(f'{"═"*75}')
