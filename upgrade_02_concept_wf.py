# -*- coding: utf-8 -*-
"""升级#2: 概念动量因子 · WF全量回测
7因子(6原始+概念动量) vs 6因子基准 · 3年滚动WF
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
from itertools import combinations
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("升级#2: 概念动量因子 WF验证")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print('[1] 加载数据...')
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 核心因子
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
print(f'  因子表: {len(fn)}行')

# K线
kline = con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(close*total_share/10000,close*vol/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# HS300
hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['high_2y'] = hs300['close'].rolling(504).max()
con.close()

# 概念动量
cm = pd.read_parquet('D:/AgentQuant/our/cache/concept_monthly.parquet')
cm['month'] = pd.to_datetime(cm['month'])
stock_concept = {}
tm = pd.read_parquet('D:/AgentQuant/our/cache/ts/ths_members_300.parquet')
def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()
tm['ncode'] = tm['con_code'].apply(norm)
# 筛纯主题概念(去风格)
wide = ['全A','沪深','科创','创业','主板','中小','ST','新股','次新','指数','综合','加权','等权',
        '减持','大盘','小盘','中盘','均衡','动量','盈利','价值','成长','除金融','除科创']
tm_filt = tm[~tm['concept_name'].str.contains('|'.join(wide))]
sizes = tm_filt.groupby('concept_code')['ncode'].nunique()
top60 = sizes[sizes>=20].head(60).index.tolist()
tm_clean = tm_filt[tm_filt['concept_code'].isin(top60)]
for ncode, grp in tm_clean.groupby('ncode'):
    stock_concept[ncode] = grp['concept_code'].tolist()
print(f'  概念因子: {cm.concept.nunique()}概念, {len(stock_concept)}只股票')

# 月频调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print(f'  月频调仓: {len(monthly_dates)}月')

# 前向收益映射
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# HS300状态
hs300_m = {}
for _, r in hs300.iterrows():
    hs300_m[r['trade_date']] = {'close': r['close'], 'high_2y': r['high_2y']}
for d in monthly_dates:
    if d not in hs300_m:
        nb = hs300[hs300['trade_date']<=d]
        if len(nb)>0:
            hs300_m[d] = {'close': nb.iloc[-1]['close'], 'high_2y': nb.iloc[-1]['high_2y']}

fn['month'] = fn['trade_date'].dt.to_period('M'); fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

# 添加概念动量到因子表
print('[2] 添加概念动量到月度因子表...')
# 月度概念动量rank (pct)
cm_pivot = cm.pivot(index='month', columns='concept', values='concept_mom')

def add_concept_mom(day):
    """给日度数据添加概念动量"""
    m = day['month'].iloc[0]
    if m not in cm_pivot.index:
        day['concept_mom'] = np.nan
        return day
    scores = []
    for nc in day['ts_code_norm'].values:
        if nc in stock_concept:
            cons = stock_concept[nc]
            sc = [cm_pivot.loc[m, c] for c in cons if c in cm_pivot.columns]
            scores.append(np.nanmean(sc) if sc else np.nan)
        else:
            scores.append(np.nan)
    day['concept_mom'] = scores
    return day

# 只为回测年份构建月度概念动量 (减少内存)
YEARS = sorted(set(d.year for d in monthly_dates))
FAST_START = max(2018, YEARS[0]+3)

# ============================================================
# 2. WF回测参数
# ============================================================
TRAIN = 3; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095; TOP_N = 30; COST = 0.0033
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10

def gate_fn(d, st):
    if d not in hs300_m: return 1.0, st
    info = hs300_m[d]; c = info['close']; h2 = info.get('high_2y', c)
    if pd.isna(h2): return 1.0, st
    if st['in']:
        if c/h2-1 < EXIT_THRESH: return FLOOR, {'in': False}
        return 1.0, st
    else:
        rc = c/h2 if h2 > 0 else 1
        if rc > REENTRY_THRESH: return 0.7, {'in': True}
        return FLOOR, st

FEATS_BASE = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_BASE = list(combinations(FEATS_BASE, 2))
ALL_CONCEPT = list(combinations(FEATS_BASE+['concept_mom'], 2))

print(f'[3] WF回测 {FAST_START}~{YEARS[-1]}...')
print(f'  基准: {len(ALL_BASE)}对, 含概念: {len(ALL_CONCEPT)}对')

# ============================================================
# 3. WF回测
# ============================================================
for mode, pairs_list in [('6因子基准', ALL_BASE), ('+概念动量7因子', ALL_CONCEPT)]:
    results = []; state = {'in': True}
    for yr in range(FAST_START, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01')
        train_e = pd.Timestamp(f'{yr-1}-12-31')
        test_mds = [d for d in monthly_dates if d.year==yr]

        # 训练期: 选最佳4对
        pair_ir = {}
        for fa, fb in pairs_list:
            sp = []
            for rd in [d for d in monthly_dates if train_s<=d<=train_e]:
                if rd not in rd_map: continue
                day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
                day = day[day['ts_code'].isin(set(px.index))]
                if len(day) < 50: continue
                if 'concept_mom' in (fa, fb):
                    day = add_concept_mom(day)
                    day = day.dropna(subset=['concept_mom'])
                    if len(day) < 50: continue
                for f in [fa, fb]:
                    if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
                if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
                day['score'] = day[fa+'_r']*day[fb+'_r']
                day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
                vd = day.dropna(subset=['score','fwd_ret'])
                if len(vd) < 50: continue
                nq = max(1, len(vd)//5)
                sp.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
            if len(sp) >= 6:
                mu = np.mean(sp); pair_ir[(fa,fb)] = mu/np.std(sp) if np.std(sp)>0 else 0

        active = sorted([(p,ir) for p,ir in pair_ir.items() if ir>0], key=lambda x: x[1], reverse=True)[:4]
        if len(active) < 2:
            active = sorted([(p,ir) for p,ir in pair_ir.items()], key=lambda x: x[1], reverse=True)[:2]

        # 测试期
        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = gate_fn(rd, state)
            if pos < 0.01: results.append(0.0); continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            day = day[day['ts_code'].isin(set(px.index))]
            if len(day) < 50: continue
            if 'concept_mom' in str(active):
                day = add_concept_mom(day)
            all_f = list(set([x for p, _ in active for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0; vn = 0
            for (fa, fb), _ in active:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r']*day[fb+'_r']; vn += 1
            if vn == 0: continue
            day['score'] /= vn
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue
            top = day.nlargest(TOP_N, 'score')
            if len(top) >= 10: results.append((top['fwd_ret'].mean()-COST)*pos)

    if results:
        r_all = np.array(results); n = len(r_all)
        ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
        sh = ann/vol if vol>0 else 0
        cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
        total = np.prod(1+r_all)-1
        win = np.mean(r_all>0)*100
        print(f'  {mode}: 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.1f}% 胜率{win:.0f}% 累积{total:+6.1%} ({n}月)')

print(f'\n耗时: {time.time()-t0:.0f}s')
