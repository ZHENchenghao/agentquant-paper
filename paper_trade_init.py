# -*- coding: utf-8 -*-
"""
小众战法 · 纸交初始化 · 2026-06-18建仓
========================================
A股交易规则:
  - T+1: 今日买,次日才能卖
  - 最小单位: 100股(1手)
  - 涨跌停: ±10%(主板), ±20%(科创/创业)
  - 印花税: 0.05%(卖出单向)
  - 佣金: 0.02%(买卖双向, 最低5元)
  - 过户费: 0.001%(买卖双向)
  - 滑点预估: 买入+0.1%, 卖出-0.1%
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')

CAPITAL = 100000          # 纸交本金
TOP_N = 30
COST_MONTHLY = 0.0033     # 双边预留: 印花0.05+佣金0.04+过户0.002+滑点≈0.33%
MCAP_FLOOR = 0.20
LIMIT_UP = 0.095

FEATS = ['amihud', 'max_rev', 'price_rev', 'turnover_rev', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')
]

TARGET_DATE = pd.Timestamp('2026-06-18')

print("=" * 70)
print(f"小众战法 · 纸交建仓 · {TARGET_DATE.date()}")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 全量K线 (用于训练期价格映射)
kline_all = con.execute("""
    SELECT ts_code, trade_date, open, close, vol,
           COALESCE(amount, GREATEST(vol*close, 1.0)) AS amount_proxy,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d,
           close/pre_close-1 AS change_pct
    FROM kline_daily WHERE trade_date >= '2020-12-01'
""").df()
kline_all['trade_date'] = pd.to_datetime(kline_all['trade_date'])

# 6/18当天K线
kline_today = kline_all[kline_all['trade_date'] == TARGET_DATE].copy()
print(f"6/18 K线: {len(kline_today)}只股票")

# 前日K线 (用于计算前收+涨停判断)
kline_prev = kline_all[kline_all['trade_date'] == pd.Timestamp('2026-06-17')].copy()

# 前日HS300 (用于门禁)
hs300_hist = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date >= '2005-01-01' ORDER BY trade_date
""").df()
hs300_hist['trade_date'] = pd.to_datetime(hs300_hist['trade_date'])
hs300_hist['ma50'] = hs300_hist['close'].rolling(50).mean()
hs300_hist['high_2y'] = hs300_hist['close'].rolling(504).max()
hs300_hist['low_1y'] = hs300_hist['close'].rolling(252).min()

# 股票基本信息
stock_info = con.execute("SELECT ts_code, name, list_date FROM stock_basic").df()
stock_info = stock_info.set_index('ts_code')

con.close()

# ============ 门禁判断 ============
print("[2] DD_SMART门禁判断...")
hs_row = hs300_hist[hs300_hist['trade_date'] == TARGET_DATE]
if len(hs_row) > 0:
    r = hs_row.iloc[0]
    close = r['close']; ma50 = r['ma50']; high_2y = r['high_2y']; low_1y = r['low_1y']
    dd_2y = close/high_2y - 1
    recovery = close/low_1y - 1
    above_ma50 = close > ma50

    print(f"  沪深300: {close:.1f} | MA50: {ma50:.1f} | 2年高: {high_2y:.1f}")
    print(f"  2年回撤: {dd_2y*100:+.1f}% | 1年低点反弹: {recovery*100:+.1f}% | MA50之上: {above_ma50}")

    if dd_2y > -0.15:
        print(f"  >>> 门禁: 满仓 (回撤{dd_2y*100:.1f}%>-15%)")
        GATE_POSITION = 1.0
    elif dd_2y > -0.20:
        print(f"  >>> 门禁: 4成仓 (回撤{dd_2y*100:.1f}%)")
        GATE_POSITION = 0.4
    else:
        print(f"  >>> 门禁: 2成仓 (回撤{dd_2y*100:.1f}%<-20%)")
        GATE_POSITION = 0.2
else:
    GATE_POSITION = 1.0

# ============ 选股 ============
print(f"[3] 6因子选股 (训练窗: 2021-2025)...")

# 训练期选对
train_data = fn[(fn['trade_date'] >= '2021-01-01') & (fn['trade_date'] < '2026-01-01')]
print(f"  训练数据: {len(train_data):,}行")

# 简化: 用2021-2025全期IR选对
dates_train = sorted(train_data['trade_date'].unique())
monthly_train = []
for ym, g in pd.Series(dates_train).groupby([d.strftime('%Y-%m') for d in dates_train]):
    monthly_train.append(g.iloc[0])
monthly_train = sorted(monthly_train)

# 构建训练期价格映射
rd_map_train = {}
for i in range(len(monthly_train)-1):
    cur = monthly_train[i]; nxt = monthly_train[i+1]
    cp = kline_all[kline_all['trade_date']==cur][['ts_code','close','amount_proxy']].rename(columns={'amount_proxy':'mcap'}).set_index('ts_code')
    np_ = kline_all[kline_all['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'no'}).set_index('ts_code')
    if cp is not None and np_ is not None and len(cp)>0 and len(np_)>0:
        m = cp.join(np_, how='inner'); m['fwd_ret'] = m['no']/m['close']-1
        rd_map_train[cur] = m

# 选IR最高4对
pair_ir = {}
for (fa, fb) in ALL_PAIRS:
    spreads = []
    for rd in monthly_train:
        if rd not in rd_map_train: continue
        day = train_data[train_data['trade_date']==rd].copy()
        px = rd_map_train[rd]
        valid = set(px.index); day = day[day['ts_code'].isin(valid)]
        if len(day) < 100: continue
        for f in [fa,fb]:
            if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)
        if f'{fa}_r' not in day.columns or f'{fb}_r' not in day.columns: continue
        day['score'] = day[f'{fa}_r']*day[f'{fb}_r']
        day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
        vd = day.dropna(subset=['score','fwd_ret'])
        if len(vd) < 50: continue
        nq = int(len(vd)*0.2)
        spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
    if len(spreads)>=12:
        mu=np.mean(spreads); std=np.std(spreads)
        pair_ir[(fa,fb)]=mu/std if std>0 else 0

sorted_pairs = sorted(pair_ir.items(), key=lambda x:x[1], reverse=True)
top4 = [p for p,ir in sorted_pairs[:4]]
print(f"  选对: {' | '.join([f'{a[:4]}x{b[:4]}(IR{ir:.2f})' for (a,b),ir in sorted_pairs[:4]])}")

# ============ 6/18当天选股 ============
print("[4] 6/18选股...")
day = fn[fn['trade_date'] == TARGET_DATE].copy()
print(f"  当日因子数据: {len(day)}只")

# 与价格合并 (用rename避免列名冲突, amount_proxy作市值代理)
kt = kline_today[['ts_code','close','amount_proxy','change_pct','ret_1d']].rename(
    columns={'close':'price_today','amount_proxy':'mcap_today'})
day = day.merge(kt, on='ts_code', how='inner')
print(f"  有价格数据: {len(day)}只")

# 计算因子排名
all_f = list(set([x for p in top4 for x in p]))
for f in all_f:
    if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

# 乘法得分
day['score'] = 0
for fa,fb in top4:
    if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
        day['score'] += day[f'{fa}_r']*day[f'{fb}_r']

# 风控过滤
day['mcap_r'] = day['mcap_today'].rank(pct=True)
day = day[day['mcap_r'] >= MCAP_FLOOR]           # 市值后20%剔除
# 涨停过滤: 用change_pct, fallback到ret_1d
day['lim_chk'] = day['change_pct'].fillna(day['ret_1d'])
day = day[day['lim_chk'].notna() & (day['lim_chk'] < LIMIT_UP)]
day = day[day['price_today'] > 0]
day = day[day['mcap_today'] > 0]
print(f"  风控过滤后: {len(day)}只")

# Top30
top = day.nlargest(TOP_N, 'score')
print(f"  选出: {len(top)}只")

# ============ 计算仓位 ============
print("[5] 计算仓位+手续费...")

# A股费率
STAMP_TAX = 0.0005    # 印花税(卖出)
COMMISSION = 0.0002   # 佣金
TRANSFER = 0.00001    # 过户费
SLIPPAGE_BUY = 0.001  # 买入滑点

total_capital = CAPITAL * GATE_POSITION  # 实际投入
cash_per_stock = total_capital / TOP_N

# 获取股票名称
top = top.copy()
top['name'] = top['ts_code'].map(lambda x: stock_info.loc[x,'name'] if x in stock_info.index else '?')

positions = []
total_cost = 0

for _, row in top.iterrows():
    code = row['ts_code']
    price = row['price_today']
    name = row['name']
    score = row['score']
    mcap_val = row['mcap_today']

    # 整手计算 (100股=1手)
    target_value = cash_per_stock
    shares_raw = target_value / (price * (1 + SLIPPAGE_BUY))
    lots = int(shares_raw / 100)  # 整手数
    if lots < 1: lots = 1        # 至少1手
    shares = lots * 100

    buy_price = price * (1 + SLIPPAGE_BUY)
    trade_value = shares * buy_price

    # 买入费用
    buy_commission = max(5, trade_value * COMMISSION)  # 最低5元
    buy_transfer = trade_value * TRANSFER
    buy_cost = trade_value + buy_commission + buy_transfer

    total_cost += buy_cost

    positions.append({
        'code': code,
        'name': name,
        'shares': shares,
        'price': round(price, 2),
        'buy_price': round(buy_price, 2),
        'trade_value': round(trade_value, 2),
        'cost_with_fee': round(buy_cost, 2),
        'score': round(score, 4),
        'mcap_yi': round(mcap_val, 1),
        'sector': '科创板' if code.startswith('688') else ('创业板' if code.startswith('300') or code.startswith('301') else '主板')
    })

# 输出
print(f"\n{'='*70}")
print(f"小众战法 · 纸交持仓 ({TARGET_DATE.date()})")
print(f"{'='*70}")
print(f"本金: ¥{CAPITAL:,} | 门禁仓位: {GATE_POSITION*100:.0f}% | 实际投入: ¥{total_capital:,.0f}")
print(f"单票目标: ¥{cash_per_stock:,.0f} | 选股: {len(positions)}只")
print(f"{'='*70}")
print(f"{'代码':<12s} {'名称':<10s} {'板块':<6s} {'股数':>6s} {'买入价':>8s} {'含费成本':>10s} {'得分':>8s} {'市值(亿)':>8s}")
print("-"*85)

for p in positions:
    print(f"{p['code']:<12s} {p['name']:<10s} {p['sector']:<6s} {p['shares']:>6d} {p['buy_price']:>8.2f} ¥{p['cost_with_fee']:>9.2f} {p['score']:>8.4f} {p['mcap_yi']:>8.1f}")

total_fee = total_cost - sum(p['shares'] * p['price'] for p in positions)
print("-"*85)
print(f"{'合计':<12s} {'':10s} {'':6s} {'':>6s} {'':>8s} ¥{total_cost:>9.2f}")
print(f"  其中手续费: ¥{total_fee:.2f} (佣金{total_cost*COMMISSION:.2f}+过户{total_cost*TRANSFER:.2f}+滑点{sum(p['shares']*p['price'] for p in positions)*SLIPPAGE_BUY:.2f})")
print(f"  剩余现金: ¥{CAPITAL - total_cost:,.2f}")
print(f"  名义持仓: ¥{sum(p['shares']*p['price'] for p in positions):,.2f}")

# ============ 保存纸交账户 ============
account = {
    "strategy": "小众战法_Top30",
    "version": "v1.0",
    "initial_capital": CAPITAL,
    "start_date": "2026-06-18",
    "gate_position": GATE_POSITION,
    "pairs": [f"{a[:4]}x{b[:4]}" for a,b in top4],
    "cash": round(CAPITAL - total_cost, 2),
    "positions": {},
    "history": [],
    "rules": {
        "stamp_tax": "0.05% (卖出单向)",
        "commission": "0.02% (最低5元,买卖双向)",
        "transfer_fee": "0.001% (买卖双向)",
        "slippage_buy": "0.1% (买入预估)",
        "slippage_sell": "0.1% (卖出预估)",
        "t_plus_1": True,
        "limit_up_down": "主板±10%, 科创/创业±20%",
        "lot_size": 100,
        "rebalance": "月度(每月首个交易日)",
        "cost_reserve": "0.33%/月 (双边预留)"
    }
}

for p in positions:
    account["positions"][p['code']] = {
        "name": p['name'],
        "shares": p['shares'],
        "buy_price": p['price'],
        "buy_price_with_slippage": p['buy_price'],
        "cost_with_fee": p['cost_with_fee'],
        "buy_date": "2026-06-18",
        "score": p['score'],
        "sector": p['sector']
    }
    account["history"].append({
        "date": "2026-06-18",
        "action": "BUY",
        "code": p['code'],
        "name": p['name'],
        "shares": p['shares'],
        "price": p['price'],
        "cost": p['cost_with_fee'],
        "score": p['score']
    })

OUT = 'D:/AgentQuant/our/paper_portfolio_xiaozhong.json'
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(account, f, ensure_ascii=False, indent=2)
print(f"\n纸交账户已保存: {OUT}")

# 对比总览
print(f"\n{'='*70}")
print("三账户总览")
print(f"{'='*70}")
print(f"  ML选股:    22只A股  (existing paper_portfolio.json)")
print(f"  ETF轮动:    5只ETF  (existing paper_portfolio.json)")
print(f"  小众战法:   {len(positions)}只A股  (paper_portfolio_xiaozhong.json)")
print(f"  总账户数:   3")
