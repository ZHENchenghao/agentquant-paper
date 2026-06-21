# -*- coding: utf-8 -*-
"""
小众战法 每日纸交引擎 vFinal
==============================
三道防线: ST过滤 + 暴跌止损 + DD_SMART门禁
7因子: price_rev,turnover_rev,amihud,max_rev,sr5,vp_corr,ind_mom
ETF: 双模切换(BULL动量/BEAR低波)
"""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
import duckdb, pandas as pd, numpy as np
import warnings; warnings.filterwarnings('ignore')

DB = 'D:/FreeFinanceData/data/duckdb/finance.db'
DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = date.today()
CAPITAL = 100000
TOP_N = 30
COST = 0.0033
MCAP_FLOOR = 0.20
LIMIT_UP = 0.095
EXIT_THRESH = -0.12
REENTRY_THRESH = 0.10
FLOOR = 0.10
CRASH_STOP = -0.30
FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr','ind_mom']
# 最优4对(含行业动量, 3年WF训练)
TOP4 = [('turnover_rev','ind_mom'),('price_rev','ind_mom'),('price_rev','turnover_rev'),('max_rev','ind_mom')]

def normalize_code(code):
    code = str(code).strip()
    if '.' in code:
        parts = code.split('.')
        return parts[1].lower() + parts[0]
    return code.lower()

# ═══════════════════════════════════════════════
# 1. 交易日检测 + 数据加载
# ═══════════════════════════════════════════════
con = duckdb.connect(DB, read_only=True)
latest_kline = str(con.execute("SELECT MAX(trade_date) FROM kline_daily").fetchone()[0])
LTD = latest_kline  # 用最新K线日期
print(f'Daily Runner — {TODAY.isoformat()} | 最新K线: {LTD}')

# 如果K线不是今天/昨天, 可能周末, 用最新数据
TARGET = LTD

# ═══════════════════════════════════════════════
# 2. DD_SMART v2 门禁
# ═══════════════════════════════════════════════
r = con.execute(f"""
    SELECT close, AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma50,
           MAX(close) OVER(ORDER BY trade_date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW) AS high_2y,
           MIN(close) OVER(ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS low_1y
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date <= DATE '{TARGET}'
    ORDER BY trade_date DESC LIMIT 1
""").fetchone()
close, ma50, high_2y, low_1y = r
dd_2y = close/high_2y - 1
recovery_1y = close/low_1y - 1 if low_2y and low_2y > 0 else 0

if dd_2y >= EXIT_THRESH:
    gate_pos, gate_msg = 1.0, 'FULL'
elif dd_2y >= EXIT_THRESH - 0.05:
    gate_pos, gate_msg = FLOOR * 2, 'REDUCE'
else:
    gate_pos, gate_msg = FLOOR, 'CRASH'

print(f'DD_SMART: HS300={close:.0f} DD={dd_2y*100:.1f}% -> {gate_msg}({gate_pos*100:.0f}%)')

# ═══════════════════════════════════════════════
# 3. 因子数据 + 行业动量
# ═══════════════════════════════════════════════
fn = pd.read_parquet(f'{DIR}/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
day = fn[fn['trade_date'] == TARGET].copy()

# 价格+市值
kt = con.execute(f"""
    SELECT ts_code, close, COALESCE(amount, GREATEST(vol*close,1.0)) AS amount_proxy,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date = DATE '{TARGET}'
""").df()

# 行业动量
ind_map = con.execute('SELECT ts_code, ind_name FROM stock_industry').df().rename(columns={'ind_name':'industry'})
ind_idx = con.execute(f"""
    SELECT industry, close, trade_date FROM proxy_industry_daily
    ORDER BY industry, trade_date
""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M')
ind_m = ind_idx.groupby(['industry','month'])['close'].last().reset_index()
ind_m['month'] = ind_m['month'].dt.to_timestamp()
ind_m['ind_ret_1m'] = ind_m.groupby('industry')['close'].pct_change()
latest_m = ind_m['month'].max()
latest_ind = ind_m[ind_m['month'] == latest_m][['industry','ind_ret_1m']].dropna()

# ST名单
st_set = set(con.execute("SELECT ts_code FROM stock_basic WHERE is_st=true").fetchdf()['ts_code'].values)
delisted_set = set(con.execute("SELECT ts_code FROM stock_basic WHERE delist_date IS NOT NULL").fetchdf()['ts_code'].values)
names_df = con.execute('SELECT ts_code, name FROM stock_basic').df()
con.close()

# 统一ts_code格式
def norm(c):
    return normalize_code(c)

ind_map['ts_code_fmt'] = ind_map['ts_code'].apply(norm)
st_set_norm = set(norm(c) for c in st_set)
delisted_norm = set(norm(c) for c in delisted_set)
names_df['ts_code_norm'] = names_df['ts_code'].apply(norm)
names = names_df.set_index('ts_code_norm')['name']

kt['ts_code_norm'] = kt['ts_code'].apply(norm)
day['ts_code_norm'] = day['ts_code'].apply(norm)

# 合并
day = day.merge(kt[['ts_code_norm','close','amount_proxy','ret_1d']].rename(columns={'close':'price','amount_proxy':'mcap_proxy'}), on='ts_code_norm', how='inner')
ind_merge = ind_map[['ts_code_fmt','industry']].drop_duplicates(subset='ts_code_fmt')
day = day.merge(ind_merge, left_on='ts_code_norm', right_on='ts_code_fmt', how='left')
day = day.merge(latest_ind, on='industry', how='left')
day['ind_ret_1m'] = day['ind_ret_1m'].fillna(0)
day['ind_mom'] = day['ind_ret_1m'].rank(pct=True)
day['ts_code'] = day['ts_code_norm']  # 统一用norm格式

print(f'因子: {len(day)}只')

# ═══════════════════════════════════════════════
# 4. 风控过滤
# ═══════════════════════════════════════════════
n_before = len(day)
day = day.dropna(subset=['industry'])
day = day[~day['ts_code'].isin(st_set_norm)]
day = day[~day['ts_code'].isin(delisted_norm)]
print(f'ST/退市过滤: {n_before}->{len(day)}只')

# 因子排名+打分
all_f = list(set([x for p in TOP4 for x in p]))
for f in all_f:
    if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)

day['score'] = 0
for fa, fb in TOP4:
    if fa+'_r' in day.columns and fb+'_r' in day.columns:
        day['score'] += day[fa+'_r'] * day[fb+'_r']

# MCAP+涨跌停过滤
day['mcap_r'] = day['mcap_proxy'].rank(pct=True)
day = day[day['mcap_r'] >= MCAP_FLOOR]
day = day[day['ret_1d'].notna() & (day['ret_1d'] < LIMIT_UP)]
day = day[day['price'] > 0]
day = day[day['mcap_proxy'] > 0]
print(f'风控后: {len(day)}只')

# ═══════════════════════════════════════════════
# 5. 选股建仓
# ═══════════════════════════════════════════════
top = day.nlargest(TOP_N, 'score')
SLIPPAGE = 0.001; COMMISSION = 0.0002; TRANSFER = 0.00001
cash_per = CAPITAL * gate_pos / TOP_N
total = 0; positions = []

print(f'\n=== 小众战法 Top{TOP_N} ({TARGET}) ===')
for _, row in top.iterrows():
    code = row['ts_code']; price = row['price']
    name = names.get(code, code)
    lots = max(1, int(cash_per / (price * (1+SLIPPAGE)) / 100))
    shares = lots * 100
    bp = price * (1+SLIPPAGE)
    tv = shares * bp
    cost = tv + max(5, tv*COMMISSION) + tv*TRANSFER
    total += cost
    positions.append({'code':code,'name':name,'shares':shares,'price':float(price),
                      'buy_price':float(bp),'cost':float(cost),'score':float(row['score'])})

print(f'合计: {len(positions)}只 | 投入: {total:.2f} | 现金: {CAPITAL-total:.2f}')

# ═══════════════════════════════════════════════
# 6. 暴跌止损检查(持仓)
# ═══════════════════════════════════════════════
pf_file = f'{DIR}/paper_portfolio_xiaozhong.json'
old_pf = None
if os.path.exists(pf_file):
    try:
        with open(pf_file, 'r') as f:
            old_pf = json.load(f)
        old_positions = old_pf.get('positions', [])
        if old_positions:
            con2 = duckdb.connect(DB, read_only=True)
            stopped = []
            for pos in old_positions:
                code = pos['code']; entry = pos['buy_price']
                r2 = con2.execute(f"SELECT close FROM kline_daily WHERE ts_code='{code}' AND trade_date<=DATE '{TARGET}' ORDER BY trade_date DESC LIMIT 1").fetchone()
                if r2:
                    pnl = float(r2[0])/entry - 1
                    if pnl < CRASH_STOP:
                        stopped.append({'code':code,'name':pos.get('name','?'),'pnl':pnl})
                        print(f'  !!止损: {code} {pos.get("name","?")} {pnl*100:+.1f}%')
            con2.close()
            if stopped:
                old_pf['stop_loss'] = {'date':TARGET,'count':len(stopped),'codes':[s['code'] for s in stopped]}
    except Exception as e:
        print(f'止损检查异常: {e}')

# ═══════════════════════════════════════════════
# 7. 保存portfolio
# ═══════════════════════════════════════════════
acct = {
    'strategy': '小众战法_vFinal_7F',
    'date': TARGET,
    'capital': CAPITAL,
    'gate': {'position': gate_pos, 'hs300': float(close), 'dd_2y': float(dd_2y)*100},
    'pairs': ['x'.join([a[:4] for a in p]) for p in TOP4],
    'cash': round(CAPITAL-total, 2),
    'invested': round(total, 2),
    'positions': positions,
    'stop_loss': old_pf.get('stop_loss') if old_pf else None
}
with open(pf_file, 'w', encoding='utf-8') as f:
    json.dump(acct, f, ensure_ascii=False, indent=2)
print(f'已保存: {pf_file}')

# ═══════════════════════════════════════════════
# 8. ETF双模信号
# ═══════════════════════════════════════════════
print(f'\n=== ETF双模 ===')
try:
    exec(open(f'{DIR}/pick_etf_dual.py').read().replace("TODAY = '2026-06-18'", f"TODAY = '{TARGET}'"))
except Exception as e:
    print(f'ETF信号异常: {e}')

print('\nDone.')
