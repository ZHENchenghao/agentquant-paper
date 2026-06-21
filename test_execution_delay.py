# -*- coding: utf-8 -*-
"""执行延迟敏感度测试
基线(T+0): 信号日→次日开盘全量成交 (当前回测假设)
实验A(T+1): 信号日→再等1天→T+1开盘成交
实验B(T+3): 信号日→再等3天→T+3开盘成交

核心问题: concept_mom的alpha能承受几天延迟?
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
from itertools import combinations
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("执行延迟敏感度测试")
print("=" * 70)

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

# ═══════════ 1. 加载数据 ═══════════
print('[1] 加载...')
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

kline = con.execute("""SELECT ts_code,trade_date,open,close,pre_close,
    COALESCE(close*total_share/10000,close*vol/1000000) AS mcap
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['high_2y'] = hs300['close'].rolling(504).max()

# 概念动量
cm = pd.read_parquet('D:/AgentQuant/our/cache/concept_monthly.parquet')
cm['month'] = pd.to_datetime(cm['month'])
cm_pivot = cm.pivot(index='month', columns='concept', values='concept_mom')

tm = pd.read_parquet('D:/AgentQuant/our/cache/ts/ths_members_300.parquet')
tm['ncode'] = tm['con_code'].apply(norm)
wide = ['全A','沪深','科创','创业','主板','中小','ST','新股','次新','指数','综合','加权','等权',
        '减持','大盘','小盘','中盘','均衡','动量','盈利','价值','成长','除金融','除科创']
tm_filt = tm[~tm['concept_name'].str.contains('|'.join(wide))]
sizes = tm_filt.groupby('concept_code')['ncode'].nunique()
top60 = sizes[sizes>=20].head(60).index.tolist()
tm_clean = tm_filt[tm_filt['concept_code'].isin(top60)]
stock_conc = tm_clean.groupby('ncode')['concept_code'].apply(list).to_dict()

# 行业动量
ind_map = con.execute('SELECT ts_code, ind_name FROM stock_industry').df().rename(columns={'ind_name':'industry'})
ind_idx = con.execute("""SELECT industry, trade_date, close FROM proxy_industry_daily
    ORDER BY industry, trade_date""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M'); ind_idx['month'] = ind_idx['month'].dt.to_timestamp()
ind_m = ind_idx.groupby(['industry','month'])['close'].last().reset_index()
ind_m['ind_ret_1m'] = ind_m.groupby('industry')['close'].pct_change()

pit_st = con.execute("SELECT * FROM pit_st_periods").df()
pit_st['st_start'] = pd.to_datetime(pit_st['st_start']); pit_st['st_end'] = pd.to_datetime(pit_st['st_end'])
con.close()

# ═══════════ 2. 调仓日历 + 多版本fwd_ret ═══════════
print('[2] 构建延迟收益映射...')
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 构建 T+0 (基线), T+1, T+3 的fwd_ret映射
# T+0: 当月首→下月首open (当前)
# T+1: 当月首→下月+1天 open
# T+3: 当月首→下月+3天 open

def build_fwd_map(delay_days):
    """delay_days: 0=下月首日, 1=下月+1天, 3=下月+3天"""
    rd_map = {}
    for i in range(len(monthly_dates)-1):
        cur = monthly_dates[i]
        # 找卖出的月份首日
        nxt_month_start = monthly_dates[i+1]
        # 延迟: 从下月首日再往后delay_days个交易日
        if delay_days == 0:
            sell_date = nxt_month_start
        else:
            # 找下月首日之后的第delay_days个交易日
            later_dates = [d for d in dates if d > nxt_month_start]
            if len(later_dates) >= delay_days:
                sell_date = later_dates[delay_days - 1]
            else:
                sell_date = later_dates[-1] if later_dates else nxt_month_start

        cp = kline[kline['trade_date']==cur][['ts_code','close','mcap']].set_index('ts_code')
        # 延迟日的ret_1d (用于涨停过滤)
        ret_day = kline[kline['trade_date']==cur]
        if len(ret_day) > 0:
            ret_vals = ret_day.set_index('ts_code')
            # 用close/pre_close-1作为当日涨跌
            ret_vals['ret_1d'] = ret_vals['close']/ret_vals['pre_close']-1
            cp['ret_1d'] = ret_vals['ret_1d']

        sp = kline[kline['trade_date']==sell_date][['ts_code','open']].rename(columns={'open':'sell_open'}).set_index('ts_code')
        m = cp.join(sp, how='inner')
        m['fwd_ret'] = m['sell_open']/m['close']-1
        m['sell_date'] = sell_date
        rd_map[cur] = m
    return rd_map

print('  T+0 (基线)...')
rd_t0 = build_fwd_map(0)
print('  T+1 (延迟1天)...')
rd_t1 = build_fwd_map(1)
print('  T+3 (延迟3天)...')
rd_t3 = build_fwd_map(3)
del kline; gc.collect()

# ═══════════ 3. 预计算月度因子 ═══════════
print('[3] 预计算因子...')
fn['month'] = fn['trade_date'].dt.to_period('M'); fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

monthly_concept = {}
for m in cm_pivot.index:
    scores = {}
    for nc in fn[fn['month']==m]['ts_code_norm'].unique():
        if nc in stock_conc:
            cons = stock_conc[nc]
            sc = [cm_pivot.loc[m, c] for c in cons if c in cm_pivot.columns]
            scores[nc] = np.nanmean(sc) if sc else 0.5
    monthly_concept[m] = scores

monthly_ind = {}
ind_merge_map = ind_map.drop_duplicates(subset='ts_code').set_index('ts_code')
ind_merge_map['ncode'] = ind_merge_map.index.map(norm)
ind_map_lookup = ind_merge_map.set_index('ncode')['industry']
for m in cm_pivot.index:
    ind_avail = ind_m[ind_m['month'] <= m]
    if len(ind_avail) == 0: continue
    latest_m = ind_avail['month'].max()
    latest = ind_avail[ind_avail['month'] == latest_m].set_index('industry')['ind_ret_1m']
    scores = {}
    for nc in fn[fn['month']==m]['ts_code_norm'].unique():
        try:
            ind = ind_map_lookup.get(nc)
            if ind is not None and ind in latest.index:
                scores[nc] = latest[ind]
        except: pass
    if scores:
        vals = np.array(list(scores.values()))
        ranks = pd.Series(vals).rank(pct=True).values
        monthly_ind[m] = {k: v for k, v in zip(scores.keys(), ranks)}

monthly_st = {}
for m in cm_pivot.index:
    m_ts = pd.Timestamp(m)
    st_set = set()
    for _, r in pit_st.iterrows():
        if r['st_start'] <= m_ts <= r['st_end']:
            st_set.add(r['ts_code'])
    monthly_st[m_ts] = st_set

print(f'  概念: {len(monthly_concept)}月, 行业: {len(monthly_ind)}月, ST: {len(monthly_st)}月')

# ═══════════ 4. 敏感度测试 ═══════════
print('[4] 敏感度测试...')
FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr','ind_mom','concept_mom']
TRAIN = 3; MCAP_FLOOR = 0.20; TOP_N = 30; COST = 0.0066
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
YEARS = sorted(set(d.year for d in monthly_dates))
FAST_START = max(2008, YEARS[0]+TRAIN)

hs300_m = {}
for _, r in hs300.iterrows():
    hs300_m[r['trade_date']] = {'close': r['close'], 'high_2y': r['high_2y']}
for d in monthly_dates:
    if d not in hs300_m:
        nb = hs300[hs300['trade_date']<=d]
        if len(nb)>0:
            hs300_m[d] = {'close': nb.iloc[-1]['close'], 'high_2y': nb.iloc[-1]['high_2y']}

def gate_fn(d, st):
    if d not in hs300_m: return 1.0, st
    info = hs300_m[d]; c = info['close']; h2 = info.get('high_2y', c)
    if pd.isna(h2): return 1.0, st
    if st['in']:
        if c/h2-1 < EXIT_THRESH: return FLOOR, {'in': False}
        return 1.0, st
    else:
        if c/h2 > REENTRY_THRESH: return 0.7, {'in': True}
        return FLOOR, st

def run_wf_delay(rd_map_delay, label, delay_desc):
    """WF回测, 使用指定的延迟成交映射"""
    all_pairs = list(combinations(FEATS, 2))
    monthly_returns = []; monthly_dates_used = []; state = {'in': True}

    for yr in range(FAST_START, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01'); train_e = pd.Timestamp(f'{yr-1}-12-31')
        test_mds = [d for d in monthly_dates if d.year==yr]

        # 训练期选对
        pair_spreads = {p: [] for p in all_pairs}
        for rd in [d for d in monthly_dates if train_s<=d<=train_e]:
            if rd not in rd_t0: continue  # 训练期用T+0
            day = fn[fn['trade_date']==rd].copy(); px = rd_t0[rd]
            day = day[day['ts_code'].isin(set(px.index))]
            if len(day) < 50: continue
            m_ts = pd.Timestamp(rd)
            if m_ts in monthly_concept:
                day['concept_mom'] = day['ts_code_norm'].map(monthly_concept[m_ts]).fillna(0.5)
            if m_ts in monthly_ind:
                day['ind_mom'] = day['ts_code_norm'].map(monthly_ind[m_ts]).fillna(0.5)
            for fa, fb in all_pairs:
                if fa not in day.columns or fb not in day.columns: continue
                day[fa+'_r'] = day[fa].rank(pct=True); day[fb+'_r'] = day[fb].rank(pct=True)
                day['_score'] = day[fa+'_r']*day[fb+'_r']
                px_idx = day['ts_code'].values; fwd = px.reindex(px_idx)['fwd_ret'].values
                valid = ~np.isnan(fwd) & ~np.isnan(day['_score'].values)
                if valid.sum() < 50: continue
                nq = max(1, valid.sum()//5)
                sorted_idx = np.argsort(day['_score'].values[valid])
                sp = fwd[valid][sorted_idx[-nq:]].mean() - fwd[valid][sorted_idx[:nq]].mean()
                pair_spreads[(fa,fb)].append(sp)

        pair_ir = {}
        for p, sp in pair_spreads.items():
            if len(sp) >= 6:
                mu = np.mean(sp); pair_ir[p] = mu/np.std(sp) if np.std(sp)>0 else 0
        active = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)[:4]
        if len(active) < 2:
            active = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)[:2]

        # 🔴 测试期: 使用延迟成交映射
        for rd in test_mds:
            if rd not in rd_map_delay: continue
            pos, state = gate_fn(rd, state)
            if pos < 0.01:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map_delay[rd]
            day = day[day['ts_code'].isin(set(px.index))]
            if len(day) < 50:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue

            m_ts = pd.Timestamp(rd)
            if m_ts in monthly_concept:
                day['concept_mom'] = day['ts_code_norm'].map(monthly_concept[m_ts]).fillna(0.5)
            if m_ts in monthly_ind:
                day['ind_mom'] = day['ts_code_norm'].map(monthly_ind[m_ts]).fillna(0.5)

            all_f = list(set([x for (p,_) in active for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0; vn = 0
            for (fa, fb), _ in active:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r']*day[fb+'_r']; vn += 1
            if vn == 0:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue
            day['score'] /= vn

            px_idx = day['ts_code'].values; px_aligned = px.reindex(px_idx)
            day['mcap'] = px_aligned['mcap'].values
            day['ret_1d'] = px_aligned.get('ret_1d', pd.Series(0, index=px_aligned.index)).values
            day['fwd_ret'] = px_aligned['fwd_ret'].values
            day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r'] >= MCAP_FLOOR]
            day = day[day['ret_1d'].notna() & (day['ret_1d'] < 0.095)]
            day = day[day['fwd_ret'].notna()]
            st_set = monthly_st.get(m_ts, set())
            day = day[~day['ts_code'].isin(st_set)]
            if len(day) < 30:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue
            top = day.nlargest(TOP_N, 'score')
            if len(top) < 10:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue
            ret = (top['fwd_ret'].mean() - COST) * pos
            monthly_returns.append(ret)
            monthly_dates_used.append(rd)

    return np.array(monthly_returns), monthly_dates_used

# 跑三个版本
print('  T+0 基线...')
r_t0, d_t0 = run_wf_delay(rd_t0, 'T+0', '基线')
print('  T+1 延迟1天...')
r_t1, d_t1 = run_wf_delay(rd_t1, 'T+1', '延迟1天')
print('  T+3 延迟3天...')
r_t3, d_t3 = run_wf_delay(rd_t3, 'T+3', '延迟3天')

# ═══════════ 5. 结果对比 ═══════════
print('\n' + '=' * 70)
print('敏感度测试结果')
print('=' * 70)

def metrics(r, label):
    n = len(r); nz = r[r!=0]
    ann = (1+np.sum(r))**(12/n)-1 if n>0 else 0
    excess = r - 0.02/12
    sh = (np.mean(excess)/np.std(excess))*np.sqrt(12) if np.std(excess)>0 else 0
    eq = np.cumprod(1+r); cmax = np.maximum.accumulate(eq)
    mdd = np.min((eq-cmax)/cmax)
    total = (np.prod(1+r)-1)*100
    win = np.mean(r>0)*100
    print(f'{label}: 年化{ann*100:+.2f}% 夏普{sh:+.2f} MDD{mdd*100:.1f}% 胜率{win:.0f}% 累积{total:+.0f}% ({n}月,非零{len(nz)}月)')
    return {'ann': ann, 'sharpe': sh, 'mdd': mdd, 'win': win, 'total': total, 'n': n, 'nz': len(nz)}

m0 = metrics(r_t0, 'T+0  | 基线(当月首→次月首)')
m1 = metrics(r_t1, 'T+1  | 延迟1天        ')
m3 = metrics(r_t3, 'T+3  | 延迟3天        ')

# 衰减分析
print(f'\n[Alpha Decay 衰减分析]')
decay_1d = (m1['ann'] - m0['ann']) / abs(m0['ann']) * 100 if abs(m0['ann']) > 0.001 else 0
decay_3d = (m3['ann'] - m0['ann']) / abs(m0['ann']) * 100 if abs(m0['ann']) > 0.001 else 0
print(f'  T+0→T+1: 年化变化 {decay_1d:+.0f}%')
print(f'  T+0→T+3: 年化变化 {decay_3d:+.0f}%')

if decay_3d > -10:
    print(f'\n  >>> 判决: concept_mom持续性优秀! 延迟3天仅衰减{abs(decay_3d):.0f}%')
    print(f'  >>> T+3分批方案完全可行。建议进入第三步: 编写实盘执行路由。')
elif decay_3d > -30:
    print(f'\n  >>> 判决: concept_mom有一定持续性, 延迟3天衰减{abs(decay_3d):.0f}%')
    print(f'  >>> T+1分批可接受。T+3风险较大, 建议拆为2批而非3批。')
else:
    print(f'\n  >>> 判决: concept_mom见光死! 延迟3天衰减{abs(decay_3d):.0f}%')
    print(f'  >>> 必须保留开盘市价成交。放弃T+3分批方案, 从其他方向压成本。')

print(f'\n耗时: {(time.time()-t0)/60:.1f}min')
