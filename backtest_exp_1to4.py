# -*- coding: utf-8 -*-
"""
小众战法 · 四实验并行验证
================================
EXP1: 基准校准 — DD_SMART体温计从HS300 → 自身标的池
EXP2: 非线性中性化 — 分5个市值桶内排名(每桶选3只=15总)
EXP3: 新规生存 — 净资产>0 + 净利润>0硬门禁
EXP4: 持仓分散度 — Top15/30/50 对比
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
FEATS = ['amihud', 'max_rev', 'price_rev', 'turnover_rev', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),('amihud','sr5'),('amihud','vp_corr'),
    ('max_rev','price_rev'),('max_rev','turnover_rev'),('max_rev','sr5'),('max_rev','vp_corr'),
    ('price_rev','turnover_rev'),('price_rev','sr5'),('price_rev','vp_corr'),
    ('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')
]
N_BUCKETS = 5

print("=" * 70)
print("小众战法 · EXP1-4 并行验证")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载数据...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

kline = con.execute("""
    SELECT ts_code, trade_date, open, close, vol,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()

# 财报: 净资产+净利润
fin = con.execute("""
    SELECT ts_code, report_date, shareholders_equity, net_profit, roe
    FROM financial_statements WHERE report_date >= '2001-01-01'
    ORDER BY ts_code, report_date
""").df()
fin['report_date'] = pd.to_datetime(fin['report_date'])
con.close()

# ============ 月度调仓日 ============
print("[2] 预处理...")
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 价格映射 (含mcap用于分桶+自建指数)
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# HS300信号
hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[d] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[d] = {'close':r['close'],'ma50':r['ma50'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

# EXP1: 构建自身标的池指数 (市值>20%分位的所有股票等权)
print("[2b] 构建自身标的池指数...")
self_pool_idx = {}  # {date: {'close': ew_close_index, 'ma50':..., 'high_2y':..., 'low_1y':...}}
ew_level = 100.0
prev_ew = None
for i, rd in enumerate(monthly_dates):
    if rd not in rd_map: continue
    px = rd_map[rd]
    pool = px[px['mcap'].rank(pct=True) >= MCAP_FLOOR]
    if len(pool) < 50: continue
    if prev_ew is not None:
        # 计算池内股票的等权收益
        common = pool.index.intersection(prev_pool.index) if prev_pool is not None else pool.index
        if len(common) > 0:
            ret = (pool.loc[common]['close'] / prev_pool.loc[common]['close']).mean() - 1
            ew_level *= (1 + ret)
    prev_pool = pool
    self_pool_idx[rd] = {'close': ew_level}

# 为自建指数添加MA/高/低
self_dates = sorted(self_pool_idx.keys())
self_closes = pd.Series({d: self_pool_idx[d]['close'] for d in self_dates}).sort_index()
self_ma50 = self_closes.rolling(12, min_periods=3).mean()  # 12月≈50周
self_high_2y = self_closes.rolling(24, min_periods=6).max()
self_low_1y = self_closes.rolling(12, min_periods=3).min()
for d in self_dates:
    self_pool_idx[d]['ma50'] = self_ma50.get(d, self_pool_idx[d]['close'])
    self_pool_idx[d]['high_2y'] = self_high_2y.get(d, self_pool_idx[d]['close'])
    self_pool_idx[d]['low_1y'] = self_low_1y.get(d, self_pool_idx[d]['close'])

# 财报月度索引
print("[2c] 构建财报索引...")
fin_sorted = fin.sort_values(['ts_code','report_date'])
fin_lookup = {}
for d in monthly_dates:
    cutoff = d - pd.DateOffset(months=2)
    latest = fin_sorted[fin_sorted['report_date'] <= cutoff].groupby('ts_code').last()
    fin_lookup[d] = latest[['shareholders_equity','net_profit','roe']]
# 连续亏损检测: 需要过去2年财报
fin_loss_lookup = {}
for d in monthly_dates:
    cutoff = d - pd.DateOffset(months=2)
    cutoff_2y = d - pd.DateOffset(months=26)
    recent = fin_sorted[(fin_sorted['report_date'] > cutoff_2y) & (fin_sorted['report_date'] <= cutoff)]
    # 对每只股票, 统计最近报告中有几个亏损
    loss_count = recent.groupby('ts_code').apply(
        lambda g: (g['net_profit'] < 0).sum(), include_groups=False
    )
    fin_loss_lookup[d] = loss_count

# ============ 门禁函数 ============
def dd_smart_gate(cur_date, state, pool_idx):
    """通用DD_SMART门禁, pool_idx可以是hs300_m或self_pool_idx"""
    if cur_date not in pool_idx: return 1.0, state
    info = pool_idx[cur_date]
    close = info['close']; ma50 = info['ma50']
    high_2y = info['high_2y']; low_1y = info['low_1y']
    if pd.isna(high_2y) or pd.isna(ma50): return 1.0, state
    if state['in_market']:
        dd_2y = close/high_2y - 1
        if dd_2y < -0.20: return 0.2, {'in_market':False,'exit_date':cur_date}
        elif dd_2y < -0.15: return 0.4, {'in_market':False,'exit_date':cur_date}
        else: return 1.0, state
    else:
        recovery = close/low_1y - 1 if pd.notna(low_1y) and low_1y>0 else 0
        above_ma50 = close > ma50
        if recovery > 0.15 and above_ma50: return 0.7, {'in_market':True,'exit_date':None}
        elif recovery > 0.10: return 0.4, state
        elif recovery > 0.05 and above_ma50: return 0.3, state
        else: return 0.15, state

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

# 训练选对 (共享)
print("[3] 训练交互对...")
fold_pairs = {}
for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_mds = [d for d in monthly_dates if test_yr-TRAIN_YEARS <= d.year < test_yr]
    if len(train_mds) < 24: continue
    pair_ir = {}
    for (fa,fb) in ALL_PAIRS:
        spreads = []
        for rd in train_mds:
            if rd not in rd_map: continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
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
    fold_pairs[test_yr] = [p for p,ir in sorted_pairs[:4]]

# ============ 实验定义 ============
EXPERIMENTS = {
    'BASE':    {'gate':'hs300','bucket':False,'fin_filter':False,'top_n':15},
    'EXP1':    {'gate':'self', 'bucket':False,'fin_filter':False,'top_n':15},  # 自建池体温计
    'EXP2':    {'gate':'hs300','bucket':True, 'fin_filter':False,'top_n':15},  # 市值桶内排名
    'EXP3':    {'gate':'hs300','bucket':False,'fin_filter':True, 'top_n':15},  # 财报硬门禁
    'EXP4_30': {'gate':'hs300','bucket':False,'fin_filter':False,'top_n':30},  # 分散度30
    'EXP4_50': {'gate':'hs300','bucket':False,'fin_filter':False,'top_n':50},  # 分散度50
    'EXP123':  {'gate':'self', 'bucket':True, 'fin_filter':True, 'top_n':15},  # 三合一
}

all_exp_results = {}

for EXP, cfg in EXPERIMENTS.items():
    GATE = cfg['gate']; BUCKET = cfg['bucket']; FIN = cfg['fin_filter']; TN = cfg['top_n']
    pool = self_pool_idx if GATE == 'self' else hs300_m

    print(f"\n--- {EXP}: gate={GATE} bucket={BUCKET} fin={FIN} n={TN} ---")
    all_results = []; state = {'in_market': True, 'exit_date': None}
    fin_stats = {'equity_kicked':0, 'loss_kicked':0, 'total_checked':0}

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = dd_smart_gate(rd, state, pool)
            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos,'exp':EXP})
                continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            # EXP3: 财务硬门禁
            if FIN and rd in fin_lookup:
                fin_month = fin_lookup[rd]
                loss_info = fin_loss_lookup.get(rd)
                before = len(day)
                day_fin = day[['ts_code']].join(fin_month, on='ts_code', how='left')
                # 净资产>0
                equity_ok = day_fin['shareholders_equity'].fillna(1) > 0
                # 净利润>0
                profit_ok = day_fin['net_profit'].fillna(0) > 0
                # 无财报数据的不杀
                no_fin = day_fin['shareholders_equity'].isna()
                keep = (equity_ok & profit_ok) | no_fin
                day = day[keep.values]
                fin_stats['equity_kicked'] += before - len(day)
                # 连续2年亏损检测
                if loss_info is not None and len(day) > 0:
                    before2 = len(day)
                    day_codes = day['ts_code'].values
                    consecutive_loss = set(loss_info[loss_info >= 2].index)
                    day = day[~day['ts_code'].isin(consecutive_loss)]
                    fin_stats['loss_kicked'] += before2 - len(day)

            fin_stats['total_checked'] += 1
            if len(day) < 100: continue

            # 因子排名
            all_f = list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

            px_match = px.loc[day['ts_code'].values]
            day['mcap'] = px_match['mcap'].values
            day['ret_1d'] = px_match['ret_1d'].values
            day['fwd_ret'] = px_match['fwd_ret'].values

            # EXP2: 市值桶内排名选股
            if BUCKET:
                day['mcap_r'] = day['mcap'].rank(pct=True)
                day = day[day['mcap_r'] >= MCAP_FLOOR]
                day = day[day['ret_1d'] < LIMIT_UP]
                day = day[day['fwd_ret'].notna()]
                if len(day) < 100: continue

                # 分5桶
                day['bucket'] = pd.cut(day['mcap_r'], bins=N_BUCKETS, labels=range(N_BUCKETS))
                day['score'] = 0
                for fa,fb in top4:
                    if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                        day['score'] += day[f'{fa}_r']*day[f'{fb}_r']

                # 每桶选TN/N_BUCKETS只
                picks_per_bucket = max(1, TN // N_BUCKETS)
                selected = []
                for b in range(N_BUCKETS):
                    bucket_data = day[day['bucket']==b]
                    if len(bucket_data) > 0:
                        top_b = bucket_data.nlargest(picks_per_bucket, 'score')
                        selected.append(top_b)
                if not selected: continue
                top = pd.concat(selected)
                if len(top) < 5: continue
            else:
                # 原版: 全局排名
                day['mcap_r'] = day['mcap'].rank(pct=True)
                day = day[day['mcap_r'] >= MCAP_FLOOR]
                day = day[day['ret_1d'] < LIMIT_UP]
                day = day[day['fwd_ret'].notna()]
                if len(day) < 50: continue

                day['score'] = 0
                for fa,fb in top4:
                    if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                        day['score'] += day[f'{fa}_r']*day[f'{fb}_r']

                top = day.nlargest(TN, 'score')
                if len(top) < 5: continue

            month_ret = (top['fwd_ret'].mean() - COST) * pos
            all_results.append({
                'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                'yr':rd.year,'pos':pos,'exp':EXP,
                'mcap_med':top['mcap'].median()
            })

    r_all = np.array([x['ret'] for x in all_results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100
    avg_pos = np.mean([x['pos'] for x in all_results if x['pos']>0.01])*100
    active_mo = sum(1 for x in all_results if x['pos']>0.01)
    calmar = ann/abs(mdd) if mdd!=0 else 0

    all_exp_results[EXP] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'avg_pos':avg_pos, 'active':active_mo, 'calmar':calmar,
        'total_ret':np.prod(1+r_all)-1, 'fin_stats':fin_stats
    }

    # 关键年
    for chk_yr in [2008,2009,2011,2017,2018,2022,2024]:
        dr_items = [x for x in all_results if x['yr']==chk_yr]
        if len(dr_items)>=3:
            yr_ret = np.prod(1+np.array([x['ret'] for x in dr_items]))-1
            print(f"  {chk_yr}: {yr_ret*100:+.1f}%", end=' ')
    print(f"\n  => 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% 活{active_mo}月")

# ============ 最终对比 ============
print(f"\n{'='*70}")
print("四实验最终对比")
print(f"{'='*70}")
print(f"{'实验':<10s} {'年化':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Calmar':>8s} {'累计':>8s} {'活月':>5s} {'2017':>8s} {'2024':>8s}")
print("-"*85)
for exp in EXPERIMENTS:
    r = all_exp_results[exp]
    # Get 2017 and 2024 specific returns
    d17 = [x['ret'] for x in r['results'] if x['yr']==2017]
    d24 = [x['ret'] for x in r['results'] if x['yr']==2024]
    r17 = np.prod(1+np.array(d17))-1 if len(d17)>=3 else 0
    r24 = np.prod(1+np.array(d24))-1 if len(d24)>=3 else 0
    print(f"{exp:<10s} {r['ann']*100:>+7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['calmar']:>+7.2f} {r['total_ret']*100:>+7.1f}% {r['active']:>5d} {r17*100:>+7.1f}% {r24*100:>+7.1f}%")

# 分年对比
print(f"\n--- 分年 ---")
exp_list = list(EXPERIMENTS.keys())
print(f"{'年':<6s} " + " ".join([f"{e:>9s}" for e in exp_list]))
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    print(f"{yr:<6d}", end='')
    for exp in exp_list:
        dr = [x['ret'] for x in all_exp_results[exp]['results'] if x['yr']==yr]
        if len(dr)>=3:
            yr_ret = np.prod(1+np.array(dr))-1
            print(f" {yr_ret*100:>+7.1f}%", end='')
        else:
            print(f" {'':>8s}", end='')
    print()

# 熊市保护
crash_years = [2008,2011,2017,2018,2022]
print(f"\n{'实验':<10s} {'5熊累计':>10s}")
for exp in exp_list:
    crash = np.prod(1+np.array([x['ret'] for x in all_exp_results[exp]['results'] if x['yr'] in crash_years]))-1
    print(f"{exp:<10s} {crash*100:>+9.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
