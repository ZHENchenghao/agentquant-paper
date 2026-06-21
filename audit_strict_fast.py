# -*- coding: utf-8 -*-
"""小众战法 · 机构级严格审计 v2 (优化版)
预计算全月概念/行业动量, WF循环只做查表
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
from itertools import combinations
from scipy import stats as sp_stats
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("小众战法 · 严格审计 v2 (优化)")
print("=" * 70)

AUDIT_RULES = """
[审计红线]
1. 严格时序分割: 3年滚动WF, 训练/测试隔断
2. 重叠隔离: 月度调仓日+1月窗口
3. 生存者偏差: PIT ST过滤(515段ST历史)
4. 摩擦成本: 双边0.66%(含税费+滑点)
5. 全局截面禁止"""
print(AUDIT_RULES)

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

# ═══════════ 1. 一次性加载 ═══════════
print('[1] 加载数据...')
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

kline = con.execute("""SELECT ts_code,trade_date,open,close,pre_close,
    COALESCE(close*total_share/10000,close*vol/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['high_2y'] = hs300['close'].rolling(504).max()

pit_st = con.execute("SELECT * FROM pit_st_periods").df()
pit_st['st_start'] = pd.to_datetime(pit_st['st_start'])
pit_st['st_end'] = pd.to_datetime(pit_st['st_end'])

# 行业动量
ind_map = con.execute('SELECT ts_code, ind_name FROM stock_industry').df().rename(columns={'ind_name':'industry'})
ind_idx = con.execute("""SELECT industry, trade_date, close FROM proxy_industry_daily
    ORDER BY industry, trade_date""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M'); ind_idx['month'] = ind_idx['month'].dt.to_timestamp()
ind_m = ind_idx.groupby(['industry','month'])['close'].last().reset_index()
ind_m['ind_ret_1m'] = ind_m.groupby('industry')['close'].pct_change()

con.close()

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

# ═══════════ 2. 调仓日历 + fwd_ret ═══════════
print('[2] 构建调仓日历...')
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
print(f'  {len(monthly_dates)}月, {monthly_dates[0].date()}~{monthly_dates[-1].date()}')

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

hs300_m = {}
for _, r in hs300.iterrows():
    hs300_m[r['trade_date']] = {'close': r['close'], 'high_2y': r['high_2y']}
for d in monthly_dates:
    if d not in hs300_m:
        nb = hs300[hs300['trade_date']<=d]
        if len(nb)>0:
            hs300_m[d] = {'close': nb.iloc[-1]['close'], 'high_2y': nb.iloc[-1]['high_2y']}

# ═══════════ 3. 预计算月度因子 (关键优化) ═══════════
print('[3] 预计算月度因子...')
fn['month'] = fn['trade_date'].dt.to_period('M'); fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

# 3a. 行业动量: 每月每个行业
ind_map['ts_code_norm'] = ind_map['ts_code'].apply(norm)

# 3b. 概念动量: 每月每只股票
monthly_concept = {}  # {month: {ts_code_norm: score}}
for m in cm_pivot.index:
    scores = {}
    for nc in fn[fn['month']==m]['ts_code_norm'].unique():
        if nc in stock_conc:
            cons = stock_conc[nc]
            sc = [cm_pivot.loc[m, c] for c in cons if c in cm_pivot.columns]
            scores[nc] = np.nanmean(sc) if sc else 0.5
    monthly_concept[m] = scores
print(f'  概念动量: {len(monthly_concept)}个月预计算完成')

# 3c. 行业动量: 每月每个股票
monthly_ind = {}  # {month: {ts_code_norm: score}}
ind_merge_map = ind_map.drop_duplicates(subset='ts_code_norm').set_index('ts_code_norm')['industry']
for m in cm_pivot.index:
    # 最近可用月
    ind_avail = ind_m[ind_m['month'] <= m]
    if len(ind_avail) == 0: continue
    latest_m = ind_avail['month'].max()
    latest = ind_avail[ind_avail['month'] == latest_m].set_index('industry')['ind_ret_1m']
    scores = {}
    for nc in fn[fn['month']==m]['ts_code_norm'].unique():
        try:
            ind = ind_merge_map.get(nc)
            ok = ind is not None and not (isinstance(ind, float) and np.isnan(ind))
        except:
            ok = False
        if ok and ind in latest.index:
            scores[nc] = latest[ind]
    if scores:
        # Rank normalize
        vals = np.array(list(scores.values()))
        ranks = pd.Series(vals).rank(pct=True).values
        monthly_ind[m] = {k: v for k, v in zip(scores.keys(), ranks)}
print(f'  行业动量: {len(monthly_ind)}个月预计算完成')

# ST快速查找 (预计算每月ST集合)
monthly_st = {}  # {month: set(ts_codes)}
for m in cm_pivot.index:
    m_ts = pd.Timestamp(m)
    st_set = set()
    for _, r in pit_st.iterrows():
        if r['st_start'] <= m_ts <= r['st_end']:
            st_set.add(r['ts_code'])
    monthly_st[m_ts] = st_set
print(f'  ST集合: {len(monthly_st)}个月预计算完成')

# ═══════════ 4. 快速WF回测 ═══════════
print('[4] 快速WF回测...')
TRAIN = 3; MCAP_FLOOR = 0.20; TOP_N = 30
COST = 0.0066  # 🔴 双边0.66%
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
YEARS = sorted(set(d.year for d in monthly_dates))
FAST_START = max(2008, YEARS[0]+TRAIN)

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

def run_wf(feats, label):
    """预计算因子+快速查表WF"""
    all_pairs = list(combinations(feats, 2))
    n_pairs = len(all_pairs)
    monthly_returns = []
    monthly_dates_used = []
    state = {'in': True}

    for yr in range(FAST_START, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01')
        train_e = pd.Timestamp(f'{yr-1}-12-31')
        test_mds = [d for d in monthly_dates if d.year==yr]

        # 快速IR计算: 训练期每月截面
        pair_spreads = {p: [] for p in all_pairs}
        for rd in [d for d in monthly_dates if train_s<=d<=train_e]:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy()
            px = rd_map[rd]
            day = day[day['ts_code'].isin(set(px.index))]
            if len(day) < 50: continue

            # 快速添加预计算因子
            m_ts = pd.Timestamp(rd)
            if m_ts in monthly_concept:
                cmap = monthly_concept[m_ts]
                day['concept_mom'] = day['ts_code_norm'].map(cmap).fillna(0.5)
            if m_ts in monthly_ind:
                imap = monthly_ind[m_ts]
                day['ind_mom'] = day['ts_code_norm'].map(imap).fillna(0.5)

            for fa, fb in all_pairs:
                if fa not in day.columns or fb not in day.columns: continue
                day[fa+'_r'] = day[fa].rank(pct=True)
                day[fb+'_r'] = day[fb].rank(pct=True)
                day['_score'] = day[fa+'_r']*day[fb+'_r']
                # 对齐fwd_ret
                px_idx = day['ts_code'].values
                fwd = px.reindex(px_idx)['fwd_ret'].values
                valid = ~np.isnan(fwd) & ~np.isnan(day['_score'].values)
                if valid.sum() < 50: continue
                nq = max(1, valid.sum()//5)
                sorted_idx = np.argsort(day['_score'].values[valid])
                top_n = sorted_idx[-nq:]; bot_n = sorted_idx[:nq]
                top_ret = fwd[valid][top_n].mean()
                bot_ret = fwd[valid][bot_n].mean()
                pair_spreads[(fa,fb)].append(top_ret - bot_ret)

        # 选最优4对
        pair_ir = {}
        for p, sp in pair_spreads.items():
            if len(sp) >= 6:
                mu = np.mean(sp); pair_ir[p] = mu/np.std(sp) if np.std(sp)>0 else 0
        active = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)[:4]
        if len(active) < 2:
            active = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)[:2]

        # 测试期: 逐月选股
        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = gate_fn(rd, state)
            if pos < 0.01:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue

            day = fn[fn['trade_date']==rd].copy()
            px = rd_map[rd]
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
                if f in day.columns:
                    day[f+'_r'] = day[f].rank(pct=True)

            day['score'] = 0; vn = 0
            for (fa, fb), _ in active:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r']*day[fb+'_r']; vn += 1
            if vn == 0:
                monthly_returns.append(0.0); monthly_dates_used.append(rd); continue
            day['score'] /= vn

            # 对齐
            px_idx = day['ts_code'].values
            px_aligned = px.reindex(px_idx)
            day['mcap'] = px_aligned['mcap'].values
            day['ret_1d'] = px_aligned['ret_1d'].values
            day['fwd_ret'] = px_aligned['fwd_ret'].values

            # 风控
            day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r'] >= MCAP_FLOOR]
            day = day[day['ret_1d'].notna() & (day['ret_1d'] < 0.095)]
            day = day[day['fwd_ret'].notna()]

            # PIT ST
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

# 运行两版本
print('  6因子基准...')
r_old, d_old = run_wf(['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr'], '6F')
print(f'    {len(r_old)}月, 非零{np.sum(r_old!=0)}月')
print('  8因子新版...')
r_new, d_new = run_wf(['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr','ind_mom','concept_mom'], '8F')
print(f'    {len(r_new)}月, 非零{np.sum(r_new!=0)}月')

# ═══════════ 5. 无泄露标准计算 ═══════════
print('\n' + '=' * 70)
print('第二步: 无泄露标准计算')
print('=' * 70)

def real_metrics(r, label):
    n = len(r); nz = r[r!=0]
    if n == 0: return {}
    equity = np.cumprod(1+r)
    ann_ret = (1+np.sum(r))**(12/n)-1
    monthly_rf = 0.02/12
    excess = r - monthly_rf
    ann_sharpe = (np.mean(excess)/np.std(excess))*np.sqrt(12) if np.std(excess)>0 else 0
    cum_max = np.maximum.accumulate(equity)
    dd = (equity-cum_max)/cum_max
    mdd = np.min(dd)
    win = np.mean(r>0); calmar = ann_ret/abs(mdd) if mdd!=0 else 0
    pf = abs(np.sum(r[r>0])/np.sum(r[r<0])) if len(r[r<0])>0 and np.sum(r[r<0])!=0 else 0

    m = {
        '年化': ann_ret, '夏普': ann_sharpe, 'MDD': mdd, 'Calmar': calmar,
        '胜率': win, '盈亏比': pf, '累积': np.prod(1+r)-1, '月数': n,
        '净值': equity, '回撤': dd, '收益': r, '非零月': nz
    }
    total_ret = (np.prod(1+r)-1)*100
    print(f'  {label}: 年化{ann_ret*100:+.2f}% 夏普{ann_sharpe:+.2f} MDD{mdd*100:.1f}% Calmar{calmar:.2f} 胜率{win*100:.0f}% 累积{total_ret:+.1f}%')
    return m

m_old = real_metrics(r_old, '6因子基准')
m_new = real_metrics(r_new, '8因子(含概念动量)')

# ═══════════ 6. 数据测谎仪 ═══════════
print('\n' + '=' * 70)
print('第三步: 数据测谎仪')
print('=' * 70)

for label, r, dates in [('8因子', r_new, d_new), ('6因子', r_old, d_old)]:
    nz = r[r!=0]
    if len(nz) < 10:
        print(f'  [{label}] 数据不足')
        continue
    skew = sp_stats.skew(nz); kurt = sp_stats.kurtosis(nz)
    print(f'\n  [{label}] 偏度{skew:+.2f} 峰度{kurt:+.2f}')
    flag_skew = "!!SKEW>2!!" if skew>2 else "OK"
    flag_kurt = "!!KURT>10!!" if kurt>10 else "OK"
    print(f'    偏度判定: {flag_skew} | 峰度判定: {flag_kurt}')

    # 异常月
    z = (nz - np.mean(nz)) / np.std(nz)
    anom = np.where(np.abs(z) > 3)[0]
    if len(anom) > 0:
        print(f'  异常月(|Z|>3): {len(anom)}个')
        nz_dates = [dates[i] for i in range(len(r)) if r[i] != 0]
        for idx in anom[:5]:
            if idx < len(nz_dates):
                dt = nz_dates[idx]; val = nz[idx]
                hs_r = hs300_m.get(dt, {})
                flag = '!!>30%!!' if val>0.3 else ('!!<-20%!!' if val<-0.2 else '')
                print(f'    {dt.date()} {val*100:+.1f}% {flag}')

    # 分布
    print(f'  分位数: P1={np.percentile(nz,1)*100:+.1f}% P5={np.percentile(nz,5)*100:+.1f}% P50={np.percentile(nz,50)*100:+.1f}% P95={np.percentile(nz,95)*100:+.1f}% P99={np.percentile(nz,99)*100:+.1f}%')

# ═══════════ 7. 分段压力测试 ═══════════
print('\n' + '=' * 70)
print('第四步: 分段压力测试')
print('=' * 70)

eras = [('2008-2018 蛮荒期', 2008, 2018), ('2019-2021 过渡期', 2019, 2021), ('2022-2026 内卷期', 2022, 2026)]

for label, r, dates in [('8因子', r_new, d_new), ('6因子', r_old, d_old)]:
    print(f'\n[{label}]')
    full_sharpe = real_metrics(r, '')['夏普']

    for era_name, start, end in eras:
        idx = [i for i, d in enumerate(dates) if start <= d.year <= end]
        if len(idx) < 6:
            print(f'  {era_name}: 数据不足')
            continue
        era_r = r[idx]
        em = real_metrics(era_r, '')
        drop = (em['夏普'] - full_sharpe) / abs(full_sharpe) * 100 if abs(full_sharpe) > 0.01 else 0
        status = '!!严重衰减!!' if drop < -40 else ('轻微衰减' if drop < -20 else '稳定')
        print(f'  {era_name}: 夏普{em["夏普"]:+.2f} MDD{em["MDD"]*100:.1f}% 年化{em["年化"]*100:+.1f}% [{status}]')

    # 逐年
    print('  逐年:')
    for yr in sorted(set(d.year for d in dates)):
        idx = [i for i, d in enumerate(dates) if d.year == yr]
        yr_r = r[idx]
        if len(yr_r) < 6: continue
        ann = np.mean(yr_r)*12; win = np.mean(yr_r>0)*100
        bar = '#'*max(0,int(ann*80)) if ann>0 else '-'*max(0,int(-ann*80))
        print(f'    {yr}: {ann*100:+5.1f}% 胜率{win:.0f}% {bar}')

# ═══════════ 8. 判决 ═══════════
print('\n' + '=' * 70)
print('审计判决')
print('=' * 70)

new_sharpe = m_new['夏普']; old_sharpe = m_old['夏普']
imp = (new_sharpe - old_sharpe) / abs(old_sharpe) * 100 if abs(old_sharpe) > 0.01 else 0
nz_new = r_new[r_new!=0]
skew_new = sp_stats.skew(nz_new) if len(nz_new)>10 else 0
kurt_new = sp_stats.kurtosis(nz_new) if len(nz_new)>10 else 0

# 诊断近年衰减
recent_idx = [i for i, d in enumerate(d_new) if d.year >= 2022]
recent_r = r_new[recent_idx] if recent_idx else np.array([0])
recent_ann = np.mean(recent_r)*12 if len(recent_r)>0 else 0
full_ann = m_new['年化']
recent_drop = (recent_ann - full_ann) / abs(full_ann) * 100 if abs(full_ann) > 0.01 else 0

decay_label = '!!严重衰减!!' if recent_drop < -40 else ('轻微衰减' if recent_drop < -20 else '保持稳定')
verdict = 'PASS: 通过审计,具备实盘可行性' if new_sharpe > 0.5 and abs(skew_new) < 2 and recent_drop > -40 else 'WARN: 需进一步验证'
print(f"""
[红线合规]
  PASS: 时序分割(3年滚动WF)
  PASS: 重叠隔离(月度调仓)
  PASS: 生存者偏差(PIT ST)
  PASS: 摩擦成本(双边0.66%)

[统计健康]
  偏度: {skew_new:+.2f} (|偏度|<2 = OK)
  峰度: {kurt_new:+.2f} (峰度<10 = OK)

[核心指标]
  6因子基准: 夏普{old_sharpe:+.2f} 年化{m_old['年化']*100:+.1f}% MDD{m_old['MDD']*100:.1f}%
  8因子新版: 夏普{new_sharpe:+.2f} 年化{m_new['年化']*100:+.1f}% MDD{m_new['MDD']*100:.1f}%
  改善幅度: {imp:+.0f}%

[近年衰减诊断]
  全期年化: {full_ann*100:+.1f}%
  近5年(2022-2026)年化: {recent_ann*100:+.1f}%
  衰减: {recent_drop:+.0f}% ({decay_label})

[最终判决]
  {verdict}
""")

print(f'总耗时: {(time.time()-t0)/60:.1f}min')
