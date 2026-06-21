# -*- coding: utf-8 -*-
"""小众战法 Top30 选股信号生成 (DD_SMART v2)"""
import duckdb, pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

TARGET = '2026-06-18'; CAPITAL = 100000; TOP_N = 30; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr','ind_mom','concept_mom']
# 最优4对(含行业+概念动量, WF验证: 概念动量提升+12.7%年化)
TOP4 = [('turnover_rev','concept_mom'),('price_rev','concept_mom'),('max_rev','concept_mom'),('price_rev','turnover_rev')]

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
r = con.execute("""
    SELECT close, AVG(close) OVER(ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma50,
           MAX(close) OVER(ORDER BY trade_date ROWS BETWEEN 503 PRECEDING AND CURRENT ROW) AS high_2y,
           MIN(close) OVER(ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS low_1y
    FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date DESC LIMIT 1
""").fetchone()
close, ma50, high_2y, low_1y = r
dd_2y = close/high_2y - 1; recovery = close/low_1y - 1; above_ma50 = close > ma50

# DD_SMART v2
if dd_2y >= EXIT_THRESH: gate_pos, gate_msg = 1.0, 'FULL'
elif dd_2y >= EXIT_THRESH - 0.05: gate_pos, gate_msg = FLOOR * 2, 'REDUCE'
else: gate_pos, gate_msg = FLOOR, 'CRASH'

print('DD_SMART v2: HS300=%.0f MA50=%.0f DD=%.1f%% REC=%.1f%% -> %s(%.0f%%)' % (close,ma50,dd_2y*100,recovery*100,gate_msg,gate_pos*100))

# 因子数据
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
day = fn[fn['trade_date'] == TARGET].copy()
print('因子: %d只' % len(day))

# 价格数据
kt = con.execute("""
    SELECT ts_code, close, COALESCE(amount, GREATEST(vol*close,1.0)) AS amount_proxy,
           close/pre_close-1 AS change_pct,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date = '%s'
""" % TARGET).df()

# ts_code格式统一
def normalize_code(code):
    code = str(code).strip()
    if '.' in code:
        parts = code.split('.')
        return parts[1].lower() + parts[0]
    return code.lower()

# 股票名称 + ST标记 + 行业映射
stocks_info = con.execute('SELECT ts_code, name, is_st, delist_date FROM stock_basic').df()
# 统一格式
stocks_info['ts_code_norm'] = stocks_info['ts_code'].apply(normalize_code)
names = stocks_info[['ts_code_norm','name']].set_index('ts_code_norm')
names.index.name = 'ts_code'
st_set = set(stocks_info[stocks_info['is_st']==True]['ts_code_norm'].values)
delisted_set = set(stocks_info[stocks_info['delist_date'].notna()]['ts_code_norm'].values)

# 行业动量 - 统一ts_code格式
ind_map = con.execute('SELECT ts_code, ind_name FROM stock_industry').df().rename(columns={'ind_name':'industry'})
ind_map['ts_code_fmt'] = ind_map['ts_code'].apply(normalize_code)
ind_idx = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    ORDER BY industry, trade_date
""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
# 找最近的完整月份
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M')
ind_monthly = ind_idx.groupby(['industry','month'])['close'].last().reset_index()
ind_monthly['month'] = ind_monthly['month'].dt.to_timestamp()
ind_monthly['ind_ret_1m'] = ind_monthly.groupby('industry')['close'].pct_change()
# 取最新月份
latest_m = ind_monthly['month'].max()
latest_ind = ind_monthly[ind_monthly['month'] == latest_m][['industry','ind_ret_1m']].dropna()
print(f'行业数据: {len(latest_ind)}个, 最新月: {latest_m.date()}')

# 🆕 概念动量 (WF验证最强新因子: 年化+12.7%提升)
print('加载概念动量...')
cm = pd.read_parquet('D:/AgentQuant/our/cache/concept_monthly.parquet')
cm['month'] = pd.to_datetime(cm['month'])
tm = pd.read_parquet('D:/AgentQuant/our/cache/ts/ths_members_300.parquet')
# 筛纯主题概念(去风格)
wide_pat = ['全A','沪深','科创','创业','主板','中小','ST','新股','次新','指数','综合','加权','等权',
            '减持','大盘','小盘','中盘','均衡','动量','盈利','价值','成长','除金融','除科创']
tm['ncode'] = tm['con_code'].apply(normalize_code)
tm_filt = tm[~tm['concept_name'].str.contains('|'.join(wide_pat))]
concept_sizes = tm_filt.groupby('concept_code')['ncode'].nunique()
top_concepts = concept_sizes[concept_sizes>=20].head(60).index.tolist()
tm_clean = tm_filt[tm_filt['concept_code'].isin(top_concepts)]
# 股票→概念映射
stock_conc = tm_clean.groupby('ncode')['concept_code'].apply(list).to_dict()
# 最新月概念动量rank
latest_cm_m = cm['month'].max()
cm_latest = cm[cm['month'] == latest_cm_m].set_index('concept')['concept_mom']
print(f'  概念动量: {len(top_concepts)}概念, {len(stock_conc)}只股票, 最新月: {latest_cm_m.date()}')

con.close()

# 统一ts_code格式
day['ts_code_norm'] = day['ts_code'].apply(normalize_code)
kt['ts_code_norm'] = kt['ts_code'].apply(normalize_code)

day = day.merge(kt.rename(columns={'close':'price','amount_proxy':'mcap_proxy'}), on='ts_code_norm', how='inner')
# 行业映射
ind_merge = ind_map[['ts_code_fmt','industry']].drop_duplicates(subset='ts_code_fmt')
day = day.merge(ind_merge, left_on='ts_code_norm', right_on='ts_code_fmt', how='left')
# 用标准化后的code做后续操作
day['ts_code'] = day['ts_code_norm']
# 删除冗余
for c in ['ts_code_norm','ts_code_fmt','ts_code_x','ts_code_y']:
    if c in day.columns: day = day.drop(columns=[c])
day = day.merge(latest_ind, on='industry', how='left')
# 没有行业数据的股票给中性分
day['ind_ret_1m'] = day['ind_ret_1m'].fillna(0)
day['ind_mom'] = day['ind_ret_1m'].rank(pct=True)

# 🆕 概念动量: 每只股票取所属概念的平均排名
concept_scores = []
for nc in day['ts_code']:
    if nc in stock_conc:
        cons = stock_conc[nc]
        sc = [cm_latest.get(c, np.nan) for c in cons if c in cm_latest.index]
        concept_scores.append(np.nanmean(sc) if sc else 0.5)  # 无概念→中性
    else:
        concept_scores.append(0.5)
day['concept_mom'] = concept_scores
n_with = sum(1 for s in concept_scores if s != 0.5)
print(f'  概念动量覆盖: {n_with}/{len(day)} ({n_with/len(day)*100:.0f}%)')

# 只保留有行业映射的个股
day = day.dropna(subset=['industry'])
print('因子+价格+行业: %d只' % len(day))

# 🆕 ST硬过滤
n_before = len(day)
day = day[~day['ts_code'].isin(st_set)]
day = day[~day['ts_code'].isin(delisted_set)]
print('ST/退市过滤: %d→%d只 (-%d)' % (n_before, len(day), n_before-len(day)))

# 因子排名
all_f = list(set([x for p in TOP4 for x in p]))
for f in all_f:
    if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)

# 乘法得分
day['score'] = 0
for fa, fb in TOP4:
    if fa+'_r' in day.columns and fb+'_r' in day.columns:
        day['score'] += day[fa+'_r'] * day[fb+'_r']

# 风控
day['mcap_r'] = day['mcap_proxy'].rank(pct=True)
day['lim_chk'] = day['change_pct'].fillna(day['ret_1d'])
day = day[day['mcap_r'] >= MCAP_FLOOR]
day = day[day['lim_chk'].notna() & (day['lim_chk'] < LIMIT_UP)]
day = day[day['price'] > 0]
day = day[day['mcap_proxy'] > 0]
print('风控后: %d只' % len(day))

# Top30
top = day.nlargest(TOP_N, 'score')
print('选出: %d只 (得分%.3f~%.3f)' % (len(top), top['score'].min(), top['score'].max()))

# 建仓
SLIPPAGE = 0.001; COMMISSION = 0.0002; TRANSFER = 0.00001
cash_per = CAPITAL * gate_pos / TOP_N
total = 0; positions = []

print('\n=== 小众战法 Top30 持仓 (%s) ===' % TARGET)
print('%-14s %-10s %s %6s %8s %8s' % ('代码','名称','板块','股数','买入价','含费'))

for _, row in top.iterrows():
    code = row['ts_code']; price = row['price']
    name = names.loc[code,'name'] if code in names.index else code
    lots = max(1, int(cash_per / (price * (1+SLIPPAGE)) / 100))
    shares = lots * 100
    bp = price * (1+SLIPPAGE)
    tv = shares * bp
    cost = tv + max(5, tv*COMMISSION) + tv*TRANSFER
    total += cost

    if code.startswith('688'): sec = '科创'
    elif code.startswith('300') or code.startswith('301'): sec = '创业板'
    else: sec = '主板'

    print('%-14s %-10s %s %6d %8.2f %8.2f' % (code, name[:8], sec, shares, bp, cost))
    positions.append({'code':code,'name':name,'shares':shares,'price':float(price),
                      'buy_price':float(bp),'cost':float(cost),'sector':sec,'score':float(row['score'])})

print('-'*55)
fee = total - sum(p['shares']*p['price'] for p in positions)
print('合计: %d只 | 投入: %.2f | 现金: %.2f | 费用: %.2f' % (len(positions), total, CAPITAL-total, fee))

# 保存
acct = {'strategy':'小众战法_Top30_DD_SMART_v2','date':TARGET,'capital':CAPITAL,
    'gate':{'position':gate_pos,'hs300':float(close),'dd_2y':float(dd_2y)*100},
    'pairs':['%s_x_%s'%(a[:4],b[:4]) for a,b in TOP4],
    'cash':round(CAPITAL-total,2),'invested':round(total,2),'positions':positions}
with open('D:/AgentQuant/our/paper_portfolio_xiaozhong.json','w',encoding='utf-8') as f:
    json.dump(acct,f,ensure_ascii=False,indent=2)
print('已保存: paper_portfolio_xiaozhong.json')
