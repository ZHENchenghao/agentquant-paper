# -*- coding: utf-8 -*-
"""
圆桌会议 · ML纸交引擎
========================
读取最新数据 → 跑ML模型 → 输出今日买卖清单 → 更新paper_portfolio
基于 backtest_ml_chain.py 的二次项OLS中性化管线。
"""
import sys, io, os, json, time
sys.path.insert(0, 'D:/AgentQuant/our')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import duckdb
import numpy as np
import pandas as pd
from datetime import date, timedelta
from lightgbm import LGBMRegressor
from quant_backtest_engine import RiskNeutralizer, ExecutionSimulator

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
PF = 'D:/AgentQuant/our/paper_portfolio.json'
TOP_N = 30
INIT_CASH = 100000

def get_db():
    for i in range(5):
        try:
            c = duckdb.connect(DB, read_only=True)
            c.execute('SELECT 1'); return c
        except Exception:
            time.sleep(min(2**i, 10))
    return duckdb.connect(DB, read_only=True)

def sql(c, q, label=''):
    for a in range(3):
        try: return c.execute(q).df()
        except Exception as e:
            if a==2: print(f'  ⚠ [{label}] {str(e)[:100]}'); return pd.DataFrame()
            time.sleep(1)
    return pd.DataFrame()

def load_portfolio():
    if os.path.exists(PF):
        with open(PF, 'r', encoding='utf-8') as f:
            pf = json.load(f)
            pf.setdefault('initial_capital', INIT_CASH)
            return pf
    return {'cash': INIT_CASH, 'initial_capital': INIT_CASH, 'positions': {}, 'history': []}

def save_portfolio(pf):
    with open(PF, 'w', encoding='utf-8') as f:
        json.dump(pf, f, ensure_ascii=False, indent=2)

def get_ts_code(raw_code):
    """600519 -> sh600519, 000001 -> sz000001"""
    raw = str(raw_code).zfill(6)
    if raw.startswith(('6','5','9')): return f'sh{raw}'
    if raw.startswith(('0','3','2')): return f'sz{raw}'
    if raw.startswith(('8','4')): return f'bj{raw}'
    return raw

def run():
    t0 = time.time()
    print('='*60)
    print(f'  圆桌会议 · ML纸交引擎')
    print(f'  运行时间: {date.today().isoformat()}')
    print('='*60)

    c = get_db()
    # 回退到最近完整交易日 (>1000只股票)
    today = c.execute("""
        SELECT trade_date FROM kline_daily
        GROUP BY trade_date HAVING COUNT(DISTINCT ts_code) > 1000
        ORDER BY trade_date DESC LIMIT 1
    """).fetchone()[0]
    print(f'  交易日期: {today} (最近完整日)')

    # ── Step 1: 构建当日因子 ──
    print('\n── 构建当日因子 ──')
    td_str = today.isoformat()

    # 估值因子 — 在JOIN前统一ts_code格式为前缀(sh/sz)
    df_I = sql(c, f"""
        WITH pit AS (
            SELECT ts_code, net_profit, eps, revenue, roe, gross_margin, net_margin,
                   ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY report_date DESC) rn
            FROM financial_statements
            WHERE report_date <= '{td_str}' AND report_date >= '{td_str}'::DATE - INTERVAL '540 days'
              AND net_profit>0 AND eps>0 AND roe>0 AND roe<100
        ),
        fin AS (
            SELECT CASE WHEN ts_code LIKE '%.SZ' THEN 'sz'||REPLACE(ts_code,'.SZ','')
                         WHEN ts_code LIKE '%.SH' THEN 'sh'||REPLACE(ts_code,'.SH','')
                         WHEN ts_code LIKE '%.BJ' THEN 'bj'||REPLACE(ts_code,'.BJ','')
                         ELSE ts_code END AS ts_code,
                   net_profit, eps, revenue, roe, gross_margin, net_margin
            FROM pit WHERE rn=1
        ),
        -- kline也统一为前缀格式
        kline_std AS (
            SELECT CASE WHEN ts_code LIKE '%.SH' THEN 'sh'||REPLACE(ts_code,'.SH','')
                         WHEN ts_code LIKE '%.SZ' THEN 'sz'||REPLACE(ts_code,'.SZ','')
                         WHEN ts_code LIKE '%.BJ' THEN 'bj'||REPLACE(ts_code,'.BJ','')
                         ELSE ts_code END AS ts_code,
                   close
            FROM kline_daily WHERE trade_date='{td_str}' AND close>0
        ),
        priced AS (
            SELECT f.*, k.close, k.close * (f.net_profit/NULLIF(f.eps,0)) AS mcap
            FROM fin f JOIN kline_std k ON f.ts_code=k.ts_code
        )
        SELECT ts_code,
               mcap/NULLIF(net_profit,0) AS pe,
               (mcap/NULLIF(net_profit,0))*(roe/100.0) AS pb,
               mcap/NULLIF(revenue,0) AS ps,
               LN(NULLIF(mcap,0)) AS log_mcap,
               roe, gross_margin, net_margin
        FROM priced WHERE mcap>0
    """, 'I')
    print(f'  估值: {len(df_I)}只')

    # 质量因子 — ts_code统一为前缀
    df_B = sql(c, f"""
        WITH pit AS (
            SELECT ts_code, roe, gross_margin, net_margin, eps, net_profit, revenue,
                   ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY report_date DESC) rn
            FROM financial_statements
            WHERE report_date<='{td_str}' AND report_date>='{td_str}'::DATE - INTERVAL '540 days'
              AND net_profit>0 AND eps>0 AND roe>0 AND roe<100
        ),
        fin AS (
            SELECT CASE WHEN ts_code LIKE '%.SZ' THEN 'sz'||REPLACE(ts_code,'.SZ','')
                         WHEN ts_code LIKE '%.SH' THEN 'sh'||REPLACE(ts_code,'.SH','')
                         WHEN ts_code LIKE '%.BJ' THEN 'bj'||REPLACE(ts_code,'.BJ','')
                         ELSE ts_code END AS ts_code,
                   roe, gross_margin, net_margin, eps, net_profit, revenue
            FROM pit WHERE rn=1
        )
        SELECT ts_code, roe, gross_margin, net_margin,
               net_profit/NULLIF(revenue,0) AS profit_margin,
               LN(NULLIF(eps,0)) AS log_eps
        FROM fin
    """, 'B')
    print(f'  质量: {len(df_B)}只')

    # 判断因子(最新信号) — 直接查单个值后广播
    margin_panic = c.execute(f"""
        SELECT CASE WHEN (margin_balance/LAG(margin_balance) OVER(ORDER BY trade_date)-1)*100 < -3
                THEN 1 ELSE 0 END FROM margin_trading
        WHERE trade_date<='{td_str}' ORDER BY trade_date DESC LIMIT 1
    """).fetchone()
    margin_panic = margin_panic[0] if margin_panic else 0

    streak5_dn = c.execute(f"""
        SELECT CASE WHEN close<LAG(close) OVER w AND LAG(close) OVER w<LAG(close,2) OVER w
                      AND LAG(close,2) OVER w<LAG(close,3) OVER w
                      AND LAG(close,3) OVER w<LAG(close,4) OVER w
                THEN 1 ELSE 0 END
        FROM kline_daily WHERE ts_code='sh000300' AND trade_date<='{td_str}'
        WINDOW w AS (ORDER BY trade_date) ORDER BY trade_date DESC LIMIT 1
    """).fetchone()
    streak5_dn = streak5_dn[0] if streak5_dn else 0

    vix_val = c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL ORDER BY trade_date DESC LIMIT 1").fetchone()
    vix_stress = 1 if (vix_val and vix_val[0] and vix_val[0]>25) else (0.5 if (vix_val and vix_val[0] and vix_val[0]>20) else 0)

    nb_bull = c.execute(f"""
        WITH n AS (SELECT SUM(net_flow) daily FROM north_bound_flow WHERE trade_date='{td_str}'),
             i AS (SELECT (close/LAG(close) OVER(ORDER BY trade_date)-1)*100 chg FROM kline_daily WHERE ts_code='sh000300' AND trade_date='{td_str}')
        SELECT CASE WHEN n.daily>30 AND i.chg>0 THEN 1 ELSE 0 END FROM n,i
    """).fetchone()
    nb_bull = nb_bull[0] if nb_bull else 0

    # 广播判断因子到全市场
    df_C = sql(c, f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date='{td_str}' AND close>0 AND vol>0
        AND ts_code NOT LIKE 'sh000%' AND ts_code NOT LIKE 'sz399%'
    """, 'stocks')
    df_C['margin_panic'] = margin_panic
    df_C['streak5_dn'] = streak5_dn
    df_C['vix_stress'] = vix_stress
    df_C['nb_bull'] = nb_bull
    df_C['nb_diverge'] = 0
    print(f'  判断: {len(df_C)}只 (恐慌={margin_panic}, 连跌={streak5_dn}, VIX压力={vix_stress})')

    # 技术因子
    df_H = sql(c, f"""
        SELECT t.ts_code, t.trade_date,
               t.rsi6/100.0 AS rsi6, t.rsi14/100.0 AS rsi14,
               CASE WHEN t.rsi6<30 THEN 1 WHEN t.rsi6>70 THEN -1 ELSE 0 END AS rsi_extreme,
               (k.close-t.boll_lower)/NULLIF(t.boll_upper-t.boll_lower,0) AS boll_pos,
               (t.boll_upper-t.boll_lower)/NULLIF(t.boll_mid,0) AS boll_width,
               k.close/NULLIF(t.ma20,0)-1 AS div_ma20,
               k.close/NULLIF(t.ma60,0)-1 AS div_ma60,
               k.close/NULLIF(t.ma120,0)-1 AS div_ma120,
               t.volume_ratio AS vol_ratio,
               CASE WHEN t.ma5>t.ma20 AND t.ma20>t.ma60 THEN 2
                    WHEN t.ma5>t.ma20 THEN 1
                    WHEN t.ma5<t.ma20 AND t.ma20<t.ma60 THEN -2
                    WHEN t.ma5<t.ma20 THEN -1 ELSE 0 END AS ma_score
        FROM technical_indicators t
        JOIN kline_daily k ON t.ts_code=k.ts_code AND t.trade_date=k.trade_date
        WHERE t.trade_date='{td_str}' AND t.rsi6 IS NOT NULL
    """, 'H')
    print(f'  技术: {len(df_H)}只')

    # ── 合并 ──
    print('\n── 合并因子 ──')
    data = df_I.copy()
    if not df_B.empty:
        overlap = set(data.columns) & set(df_B.columns) - {'ts_code'}
        if overlap: df_B = df_B.drop(columns=list(overlap))
        data = data.merge(df_B, on='ts_code', how='inner')
    for d, label in [(df_C, 'C'), (df_H, 'H')]:
        if d is None or d.empty: continue
        overlap = set(data.columns) & set(d.columns) - {'ts_code'}
        if overlap: d = d.drop(columns=list(overlap))
        data = data.merge(d, on='ts_code', how='left')
    for col in ['margin_panic', 'streak5_dn', 'vix_stress', 'nb_bull', 'nb_diverge',
                'rsi6', 'rsi14', 'rsi_extreme', 'boll_pos', 'boll_width',
                'div_ma20', 'div_ma60', 'div_ma120', 'vol_ratio', 'ma_score']:
        if col in data.columns: data[col] = data[col].fillna(0)
    # 价格: 兼容kline_daily两种ts_code格式 (sh600519 / 600519.SH)
    price_lookup = sql(c, f"SELECT ts_code, close FROM kline_daily WHERE trade_date='{td_str}'", 'price')
    if not price_lookup.empty:
        # 把两种格式都标准化为前缀格式
        price_lookup['ts_code_std'] = price_lookup['ts_code'].apply(lambda x:
            'sh'+x.replace('.SH','') if '.SH' in str(x) else
            ('sz'+x.replace('.SZ','') if '.SZ' in str(x) else
             ('bj'+x.replace('.BJ','') if '.BJ' in str(x) else str(x)))
        )
        price_lookup = price_lookup[['ts_code_std', 'close']].rename(columns={'ts_code_std': 'ts_code'})
        data = data.merge(price_lookup, on='ts_code', how='left')
    data = data.dropna(subset=['close'])
    print(f'  合并: {len(data)}只 (含价格)')

    # 特征列
    feat_cols = [c for c in data.columns if c not in
                 ('ts_code', 'trade_date', 'close', 'factor_group', 'report_date', '_k')]
    print(f'  特征: {len(feat_cols)}个')

    # ── 二次项市值中性化 ──
    print('\n── 中性化 ──')
    if 'log_mcap' in data.columns:
        data['industry_code'] = data['ts_code'].astype(str).str[:4]
        neut = RiskNeutralizer()
        neut_factors = [c for c in feat_cols if c != 'log_mcap']
        for col in neut_factors:
            if col not in data.columns: continue
            fv = neut.industry_neutralize(data[col].values, data['industry_code'].values)
            fv_q, _, _, _ = neut.size_neutralize_quadratic(fv, data['log_mcap'].values)
            data[col] = fv_q
        data.drop(columns=['industry_code'], errors='ignore', inplace=True)
        print(f'  完成')
    else:
        print(f'  ⚠ 无log_mcap')

    # ── ML训练+预测 ──
    print('\n── ML预测 ──')
    # 用历史数据训练
    hist_target = sql(c, f"""
        WITH stock_fwd AS (
            SELECT ts_code, trade_date, close,
                   LEAD(close,60) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fwd_close
            FROM kline_daily WHERE trade_date BETWEEN '2015-01-01' AND '{td_str}'
        ),
        idx_fwd AS (
            SELECT trade_date, close,
                   LEAD(close,60) OVER(ORDER BY trade_date) AS fwd_close
            FROM kline_daily WHERE ts_code='sh000300' AND trade_date BETWEEN '2015-01-01' AND '{td_str}'
        )
        SELECT s.ts_code, s.trade_date,
               (s.fwd_close/s.close-1)-(i.fwd_close/i.close-1) AS excess_ret
        FROM stock_fwd s JOIN idx_fwd i ON s.trade_date=i.trade_date
        WHERE s.fwd_close IS NOT NULL AND i.fwd_close IS NOT NULL
          AND s.trade_date >= '2018-01-01'
    """, 'train_target')

    # 用cache中的历史因子训练
    cache_file = 'D:/AgentQuant/our/cache/factors_all.parquet'
    if os.path.exists(cache_file):
        print('  加载历史因子缓存...')
        hist_data = pd.read_parquet(cache_file)
        hist_data = hist_data.merge(hist_target[['ts_code','trade_date','excess_ret']],
                                     on=['ts_code','trade_date'], how='inner')
        print(f'  历史样本: {len(hist_data)}行')

        # 同样的中性化
        if 'log_mcap' in hist_data.columns:
            hist_data['industry_code'] = hist_data['ts_code'].astype(str).str[:4]
            for col in neut_factors:
                if col not in hist_data.columns: continue
                fv = neut.industry_neutralize(hist_data[col].values, hist_data['industry_code'].values)
                fv_q, _, _, _ = neut.size_neutralize_quadratic(fv, hist_data['log_mcap'].values)
                hist_data[col] = fv_q

        X_hist = hist_data[feat_cols].fillna(hist_data[feat_cols].median())
        y_hist = hist_data['excess_ret']
        train_mask = hist_data['trade_date'] < td_str

        model = LGBMRegressor(
            objective='regression', metric='rmse',
            learning_rate=0.05, num_leaves=63, max_depth=10,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            n_estimators=300, early_stopping_rounds=20,
            verbose=-1, random_state=42, n_jobs=-1
        )

        split_idx = int(train_mask.sum() * 0.8)
        if split_idx > 500:
            X_train, y_train = X_hist[train_mask].iloc[:split_idx], y_hist[train_mask].iloc[:split_idx]
            X_val, y_val = X_hist[train_mask].iloc[split_idx:], y_hist[train_mask].iloc[split_idx:]
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_hist[train_mask], y_hist[train_mask])
        print(f'  模型训练完成')
    else:
        print('  ⚠ 无历史缓存, 需先跑 backtest_ml_chain.py 生成')
        return

    # 预测今日
    X_today = data[feat_cols].fillna(X_hist[feat_cols].median())
    scores = model.predict(X_today)
    data['ml_score'] = scores
    print(f'  预测完成, {len(data)}只股票打分')

    # ── 选股 ──
    print(f'\n── 选股 (Top {TOP_N}) ──')
    # 排除ST/上市不足1年
    st_codes = set(sql(c, f"SELECT DISTINCT ts_code FROM kline_daily WHERE trade_date='{td_str}' AND is_st=TRUE", 'st')['ts_code'].tolist())
    data_clean = data[~data['ts_code'].isin(st_codes)]

    # ── 组合层面小盘硬约束 ──
    # 1. 市值底线: 排除全市场市值最小的20%
    if 'log_mcap' in data_clean.columns:
        mcap_floor = data_clean['log_mcap'].quantile(0.20)
        data_clean = data_clean[data_clean['log_mcap'] >= mcap_floor].copy()
        print(f'  市值底线(log_mcap>{mcap_floor:.1f}), 剩余{len(data_clean)}只')
    # 2. 微盘仓位上限: 选股TopN中, log_mcap低于中位数的不得超过50%
    top_n = data_clean.drop_duplicates(subset=['ts_code']).nlargest(TOP_N * 2, 'ml_score')
    if 'log_mcap' in top_n.columns:
        median_mcap = top_n['log_mcap'].median()
        # 从高到低取, 但微盘(低于中位数)最多占一半
        large = top_n[top_n['log_mcap'] >= median_mcap]
        small = top_n[top_n['log_mcap'] < median_mcap]
        n_large = min(len(large), TOP_N)
        n_small = min(len(small), TOP_N // 2)
        top_n = pd.concat([
            large.nlargest(n_large, 'ml_score'),
            small.nlargest(n_small, 'ml_score')
        ]).drop_duplicates(subset=['ts_code']).nlargest(TOP_N, 'ml_score')
        print(f'  微盘约束: 大盘{n_large}只 + 小盘{n_small}只')
    else:
        top_n = top_n.nlargest(TOP_N, 'ml_score')
    print(f'  候选: {len(data_clean)}, 排除ST: {len(st_codes)}')

    # ── 价格(从data自带) ──
    price_map = dict(zip(data['ts_code'], data['close']))
    buy_codes = top_n['ts_code'].tolist()
    valid_buy = [c for c in buy_codes if c in price_map and price_map[c] > 0]

    # ── NLP情绪仓位控制层 ──
    nlp_n = TOP_N  # 默认满仓
    try:
        # 加载雪球词典打分最近5天新闻
        dict_dir = 'D:/AgentQuant/xueqiu_spider_LQH_LZQ-main/晴报局'
        pos_words, neg_words = set(), set()
        for path, s in [(os.path.join(dict_dir,'正面词典.txt'), pos_words),
                         (os.path.join(dict_dir,'负面词典.txt'), neg_words)]:
            if os.path.exists(path):
                with open(path,'r',encoding='utf-8') as fh:
                    for line in fh:
                        w = line.strip()
                        if w and len(w)>1: s.add(w)

        recent_news = sql(c, f\"\"\"
            SELECT title, content FROM news_articles
            WHERE publish_date >= '{td_str}'::DATE - INTERVAL '5 days'
            ORDER BY publish_date DESC LIMIT 200
        \"\"\", 'nlp_news')

        if len(recent_news) > 10:
            scores = []
            for _, row in recent_news.iterrows():
                text = str(row['title']) + ' ' + str(row.get('content',''))[:500]
                p = sum(1 for w in pos_words if w in text)
                n = sum(1 for w in neg_words if w in text)
                total = p + n
                scores.append((p-n)/max(total,1) if total>0 else 0)

            mkt_sent = np.mean(scores)
            bullish_pct = np.mean([s>0.2 for s in scores])

            if abs(mkt_sent) > 0.5:
                nlp_n = max(15, TOP_N // 2)
                print(f'  [NLP风控] 情绪极端({mkt_sent:+.2f}), 仓位降至{nlp_n}只')
            elif abs(mkt_sent) > 0.3:
                nlp_n = max(22, int(TOP_N * 0.73))
                print(f'  [NLP风控] 情绪偏高({mkt_sent:+.2f}), 仓位降至{nlp_n}只')
            else:
                print(f'  [NLP风控] 情绪正常({mkt_sent:+.2f}), 满仓{TOP_N}只')
        else:
            print(f'  [NLP风控] 新闻不足({len(recent_news)}篇), 默认满仓')
    except Exception as e:
        print(f'  [NLP风控] 异常: {e}, 默认满仓')
    # ── NLP风控层结束 ──

    print(f'  可买: {len(valid_buy)}, NLP调整后买入{nlp_n}只')

    # ── 更新纸交组合 ──
    pf = load_portfolio()
    # 标准化: 所有代码转前缀格式
    def std_code(c):
        c = str(c)
        if '.SH' in c: return 'sh'+c.replace('.SH','')
        if '.SZ' in c: return 'sz'+c.replace('.SZ','')
        if '.BJ' in c: return 'bj'+c.replace('.BJ','')
        return c

    # 标准化现有持仓
    std_positions = {}
    for code, pos in list(pf['positions'].items()):
        std_positions[std_code(code)] = pos
    pf['positions'] = std_positions

    # 构建全市场价格映射表(不只是候选股)
    all_prices = dict(zip(data['ts_code'], data['close']))
    # 补充: 对当前持仓不在factor数据中的, 从kline直接取
    for code in pf['positions']:
        if code not in all_prices:
            px_row = c.execute('SELECT close FROM kline_daily WHERE ts_code=? ORDER BY trade_date DESC LIMIT 1', [code]).fetchone()
            if px_row: all_prices[code] = px_row[0]

    # 卖出不在新选股中的持仓
    for code, pos in list(pf['positions'].items()):
        if code not in valid_buy:
            px = all_prices.get(code, 0)
            if px > 0 and pos.get('shares', 0) > 0:
                sell_cost = 0.001 + 0.00025 + 0.0005
                sell_value = pos['shares'] * px * (1 - sell_cost)
                buy_px = pos.get('buy_price', px)
                pnl_pct = (px / buy_px - 1) * 100 if buy_px > 0 else 0
                pf['cash'] += sell_value
                pf['history'].append({
                    'date': td_str, 'action': 'SELL', 'code': code,
                    'shares': pos['shares'], 'price': px, 'value': sell_value,
                    'buy_price': buy_px, 'pnl_pct': round(pnl_pct, 1),
                    'reason': 'ML调仓: 不在新选股'
                })
                del pf['positions'][code]
            else:
                # 价格缺失，保留持仓（不卖）
                pass

    # 买入新选股(等权) — 跳过已持有的
    new_buys = [c for c in valid_buy if c not in pf['positions']]
    n_buy = min(len(new_buys), nlp_n)
    if n_buy > 0:
        capital_per = pf['cash'] / n_buy
        for ts_code in new_buys[:n_buy]:
            px = price_map.get(ts_code, 0)
            if px <= 0: continue
            buy_cost = 0.00025 + 0.0005
            buy_price = px * (1 + buy_cost)
            shares = int(capital_per / buy_price / 100) * 100
            if shares == 0: continue
            cost = shares * buy_price
            if cost <= pf['cash']:
                pf['cash'] -= cost
                pf['positions'][ts_code] = {'shares': shares, 'buy_price': px, 'avg_cost': px, 'buy_date': td_str}
                pf['history'].append({
                    'date': td_str, 'action': 'BUY', 'code': ts_code,
                    'shares': shares, 'price': px, 'value': cost,
                    'reason': f'ML选股(NLP仓位={nlp_n})'
                })

    save_portfolio(pf)

    # ── 报告 ──
    print(f'\n{\"=\"*60}')
    print(f'  今日ML纸交信号')
    print(f'{\"=\"*60}')
    init_cap = pf.get('initial_capital', INIT_CASH)
    print(f'  初始本金: ¥{init_cap:,.0f}')
    print(f'  现金: ¥{pf[\"cash\"]:,.0f}')
    print(f'  持仓: {len(pf[\"positions\"])}只')

    # 估值
    total_mv = pf['cash']
    win_count = 0
    for code, pos in pf['positions'].items():
        px = price_map.get(code, pos.get('buy_price', 0))
        if px <= 0: px = all_prices.get(code, pos.get('buy_price', 0))
        if px <= 0: continue
        mv = pos['shares'] * px
        total_mv += mv
        bp = pos.get('buy_price', px)
        if bp > 0 and px > bp: win_count += 1

    total_pnl = total_mv - init_cap
    n_positions = len(pf['positions'])
    print(f'  总市值: ¥{total_mv:,.0f}')
    print(f'  总盈亏: {total_pnl:+,.0f} ({total_pnl/init_cap*100:+.1f}%)')
    if n_positions > 0:
        print(f'  持仓盈利: {win_count}/{n_positions} ({win_count/n_positions*100:.0f}%)')

    print(f'  总资产: ¥{total_mv:,.0f}  (初始¥{INIT_CASH:,})')
    print(f'  累计收益: {(total_mv/INIT_CASH-1)*100:+.1f}%')
    print(f'\n  今日买入({len(valid_buy[:n_buy])}只):')
    for i, ts_code in enumerate(valid_buy[:n_buy]):
        px = price_map.get(ts_code, 0)
        label = f'{ts_code} ¥{px:.2f}'
        print(f'  {i+1:2d}. {label:30s}')

    c.close()
    elapsed = time.time() - t0
    print(f'\n  ⏱ {elapsed:.0f}s')
    print(f'{"="*60}')

if __name__ == '__main__':
    try:
        if hasattr(sys.stdout, 'buffer') and not sys.stdout.closed:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except: pass
    run()
