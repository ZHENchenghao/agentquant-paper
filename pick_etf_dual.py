# -*- coding: utf-8 -*-
"""ETF轮动 v3 · 双模切换 实盘信号
====================================
判断: HS300 > MA200 且 MA200上升 → BULL(动量Top5)
      否则 → BEAR(低波Top5)
回撤>12% → 强制空仓
"""
import duckdb, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

TODAY = '2026-06-18'  # 最新交易日
CAPITAL = 100000; TOP_N = 5; COST = 0.003
FLOOR = 0.10; EXIT_DD = -0.12

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 1. 牛熊判断
r = con.execute("""
    SELECT close, AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma200
    FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date DESC LIMIT 1
""").fetchone()
close, ma200 = r
# MA200斜率(3月前MA200)
r2 = con.execute("""
    SELECT AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) AS ma200
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date <= DATE '2026-03-18'
    ORDER BY trade_date DESC LIMIT 1
""").fetchone()
ma200_3m_ago = r2[0] if r2 else ma200
slope = ma200 - ma200_3m_ago
is_bull = close > ma200 and slope > 0

# DD门禁
h2y = con.execute("""
    SELECT MAX(close) FROM kline_daily WHERE ts_code='sh000300'
    AND trade_date BETWEEN DATE '2024-06-18' AND DATE '2026-06-18'
""").fetchone()[0]
dd = close/h2y - 1

if dd < EXIT_DD:
    gate_pos = FLOOR; gate_msg = 'CRASH'
else:
    gate_pos = 1.0; gate_msg = 'FULL'

# 2. ETF数据
etf_all = con.execute("""
    SELECT ts_code, trade_date, close, volume, name FROM etf_daily
    ORDER BY ts_code, trade_date
""").df()
etf_all['trade_date'] = pd.to_datetime(etf_all['trade_date'])

# 过滤: >3年历史 + 流动性
etf_cnt = etf_all.groupby('ts_code')['trade_date'].nunique()
valid = etf_cnt[etf_cnt > 750].index
etf = etf_all[etf_all['ts_code'].isin(valid)].sort_values(['ts_code','trade_date'])
etf['ret'] = etf.groupby('ts_code')['close'].pct_change()
etf['vol_20d'] = etf.groupby('ts_code')['volume'].transform(lambda x: x.rolling(20).mean())

# 3. 当月信号
# 用最近20个交易日计算
etf_signal = etf.groupby('ts_code').agg(
    close=('close', 'last'),
    ret_1m=('ret', lambda x: (1+x.tail(20)).prod()-1 if len(x)>=15 else np.nan),
    vol_1m=('ret', lambda x: x.tail(20).std()*np.sqrt(252) if len(x)>=15 else np.nan),
    avg_vol=('volume', lambda x: x.tail(60).mean()),
    name=('name', 'first'),
).reset_index()
etf_signal = etf_signal.dropna(subset=['ret_1m', 'vol_1m', 'avg_vol'])
etf_signal = etf_signal[etf_signal['avg_vol'] > 20000000]

if is_bull:
    etf_signal['score'] = etf_signal['ret_1m'].rank(pct=True)  # 动量
    mode = 'BULL-动量'
else:
    etf_signal['score'] = etf_signal['vol_1m'].rank(pct=True, ascending=False)  # 低波
    mode = 'BEAR-低波'

top = etf_signal.nlargest(TOP_N, 'score')

con.close()

print(f"ETF轮动 v3 · {TODAY}")
print(f"HS300={close:.0f} MA200={ma200:.0f} slope={slope:+.0f} DD={dd*100:+.1f}%")
print(f"判态: {mode} | 门禁: {gate_msg}({gate_pos*100:.0f}%)")
print(f"\n{'代码':<10s} {'名称':<12s} {'价格':>7s} {'1月动量':>8s} {'波动':>7s} {'得分':>6s}")
print('-'*55)

positions = []
cash_per = CAPITAL * gate_pos / TOP_N

for _, row in top.iterrows():
    price = row['close']; code = row['ts_code']; name = row['name']
    mom = row['ret_1m']*100; vol = row['vol_1m']*100
    shares = max(1, int(cash_per / (price*1.001) / 100)) * 100
    cost = shares * price * 1.001
    s = row['score']
    print(f'{code:<10s} {name:<12s} {price:>6.2f} {mom:>+7.1f}% {vol:>6.1f}% {s:>5.3f}')
    positions.append({'code':code,'name':name,'shares':shares,'price':float(price),'cost':cost})

invested = sum(p['cost'] for p in positions)
print(f'\n投入: {invested:.0f} | 现金: {CAPITAL-invested:.0f} | {mode}')

# 保存
acct = {'strategy':'ETF_dual_v3','date':TODAY,'capital':CAPITAL,
    'mode':mode,'gate':gate_msg,'hs300':float(close),'ma200':float(ma200),
    'invested':round(invested,2),'cash':round(CAPITAL-invested,2),'positions':positions}
with open('paper_portfolio_etf.json','w',encoding='utf-8') as f:
    json.dump(acct, f, ensure_ascii=False, indent=2)
print('已保存: paper_portfolio_etf.json')
