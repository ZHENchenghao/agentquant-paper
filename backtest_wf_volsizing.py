# -*- coding: utf-8 -*-
"""
小众战法 + 波动率目标定仓 · Walk-Forward (2002-2026)
===================================================
对比:
  NONE      — 满仓(基准)
  MA200     — 沪深300<MA200空仓
  DRAWDOWN  — 指数2年高点回撤>15%空仓
  VOL20     — 仓位=min(1.0, 20%年化波动/已实现波动)  连续调节
  VOL15     — 仓位=min(1.0, 15%年化波动/已实现波动)  更保守
  HYBRID    — VOL20 + MA200双保险: MA200下仓位折半
"""
import sys, io; sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

TOP_N = 15; COST = 0.0033; TRAIN_YEARS = 5
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095

FEATS = ['amihud', 'max_rev', 'gap', 'sr5', 'vp_corr']
ALL_PAIRS = [
    ('amihud','max_rev'), ('amihud','gap'), ('amihud','sr5'), ('amihud','vp_corr'),
    ('max_rev','gap'), ('max_rev','sr5'), ('max_rev','vp_corr'),
    ('gap','sr5'), ('gap','vp_corr'), ('sr5','vp_corr')
]

print("=" * 70)
print("小众战法 + 波动率定仓 · Walk-Forward")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close, vol,
           COALESCE(amount, GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000, GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

# 沪深300
hs300 = con.execute("""
    SELECT trade_date, close, close/LAG(close) OVER(ORDER BY trade_date)-1 AS ret
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma200'] = hs300['close'].rolling(200).mean()
# 60日已实现波动率(年化)
hs300['ret_clean'] = hs300['ret'].clip(-0.10, 0.10)
hs300['realized_vol'] = hs300['ret_clean'].rolling(60).std() * np.sqrt(252)
con.close()

# 月度调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 价格映射
print("[2] 构建价格映射...")
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# 门禁信号: 每月查表
hs300_at_date = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        hs300_at_date[d] = {'close':row['close'].iloc[0], 'ma200':row['ma200'].iloc[0],
                            'vol':row['realized_vol'].iloc[0]}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            last = nearby.iloc[-1]
            hs300_at_date[d] = {'close':last['close'], 'ma200':last['ma200'],
                                'vol':last['realized_vol']}

# 2年高点回撤辅助
hs300_peaks = {}
for i, d in enumerate(monthly_dates):
    lookback = [md for md in monthly_dates[:i+1] if md.year >= d.year-2]
    if lookback and all(md in hs300_at_date for md in lookback):
        peak_2y = max(hs300_at_date[md]['close'] for md in lookback)
        hs300_peaks[d] = peak_2y

def get_position(cur_date, gate_type):
    """返回仓位权重 0~1, 连续"""
    if gate_type == 'NONE': return 1.0
    if cur_date not in hs300_at_date: return 1.0

    info = hs300_at_date[cur_date]
    close = info['close']; ma = info['ma200']; vol = info['vol']

    if gate_type == 'MA200':
        return 1.0 if pd.notna(ma) and close >= ma else 0.0

    if gate_type == 'DRAWDOWN':
        if cur_date in hs300_peaks:
            peak = hs300_peaks[cur_date]
            dd = close / peak - 1
            if dd < -0.20: return 0.0
            if dd < -0.15: return 0.3
            if dd < -0.10: return 0.6
        return 1.0

    if gate_type == 'VOL20':
        if pd.isna(vol) or vol <= 0: return 1.0
        target_vol = 0.20  # 年化20%
        size = target_vol / vol
        return np.clip(size, 0.15, 1.0)  # 最少15%仓位, 最多100%

    if gate_type == 'VOL15':
        if pd.isna(vol) or vol <= 0: return 1.0
        size = 0.15 / vol
        return np.clip(size, 0.10, 1.0)

    if gate_type == 'HYBRID':
        # VOL20 + MA200: 先算vol仓位, MA200下再折半
        if pd.isna(vol) or vol <= 0: base = 1.0
        else: base = np.clip(0.20/vol, 0.15, 1.0)
        if pd.notna(ma) and close < ma:
            base *= 0.5  # MA200下仓位折半
        return np.clip(base, 0.10, 1.0)

    return 1.0

# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

GATES = ['NONE', 'MA200', 'DRAWDOWN', 'VOL20', 'VOL15', 'HYBRID']
all_gate_results = {}

# 为节省时间, 只跑一次训练选对, 所有门禁共用OOS选对
# (交互对选择不受门禁影响)

print("\n[3] Walk-Forward (所有门禁共享训练选对)...")

# 先跑训练选对
fold_pairs = {}  # {test_yr: top4_pairs}
for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    train_mds = [d for d in monthly_dates if test_yr-TRAIN_YEARS <= d.year < test_yr]
    if len(train_mds) < 24: continue

    pair_ir = {}
    for (fa, fb) in ALL_PAIRS:
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
    pairs_str = ' | '.join([f'{a[:4]}×{b[:4]}' for a,b in fold_pairs[test_yr]])
    print(f"  {test_yr} 训练{test_yr-TRAIN_YEARS}-{test_yr-1}: {pairs_str}")

# 各门禁OOS测试
for GATE in GATES:
    print(f"\n--- {GATE} ---")
    all_results = []
    positions_log = []  # 记录仓位变化

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos = get_position(rd, GATE)
            positions_log.append({'date':str(rd)[:7], 'yr':rd.year, 'pos':pos, 'gate':GATE})

            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':0.0})
                continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            all_f = list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f'{f}_r'] = day[f].rank(pct=True)

            day['score'] = 0; ok = True
            for fa,fb in top4:
                if f'{fa}_r' in day.columns and f'{fb}_r' in day.columns:
                    day['score'] += day[f'{fa}_r']*day[f'{fb}_r']
                else: ok = False
            if not ok: continue

            px_match = px.loc[day['ts_code'].values]
            day['mcap'] = px_match['mcap'].values
            day['ret_1d'] = px_match['ret_1d'].values
            day['fwd_ret'] = px_match['fwd_ret'].values
            day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]
            day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day) < 50: continue

            top = day.nlargest(TOP_N,'score')
            if len(top) < 5: continue

            month_ret = (top['fwd_ret'].mean() - COST) * pos
            all_results.append({'date':str(rd)[:7],'ret':month_ret,'n':len(top),
                               'yr':rd.year,'pos':pos,'mcap_med':top['mcap'].median()})

    r_all = np.array([x['ret'] for x in all_results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100
    avg_pos = np.mean([x['pos'] for x in all_results])*100
    active_months = sum(1 for x in all_results if x['pos']>0.01)

    all_gate_results[GATE] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'months':len(r_all), 'active':active_months,
        'avg_pos':avg_pos, 'total_ret':np.prod(1+r_all)-1
    }

    print(f"  {GATE}: 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% Win{win:.0f}% 均仓{avg_pos:.0f}% 活跃{active_months}月")

# ============ 对比 ============
print(f"\n{'='*70}")
print("最终对比")
print(f"{'='*70}")
print(f"{'方案':<12s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win':>6s} {'均仓':>6s} {'活跃月':>7s} {'累计':>8s}")
print("-"*78)
for g in GATES:
    r = all_gate_results[g]
    print(f"{g:<12s} {r['ann']*100:>+7.1f}% {r['vol']*100:>7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['win']:>5.0f}% {r['avg_pos']:>5.0f}% {r['active']:>7d} {r['total_ret']*100:>+7.1f}%")

# 分年对比
print(f"\n--- 分年对比 (年收益%) ---")
print(f"{'年':<6s}", end='')
for g in GATES: print(f"{g:>9s}", end=' ')
print()
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    print(f"{yr:<6d}", end='')
    for g in GATES:
        dr = [x['ret'] for x in all_gate_results[g]['results'] if x['yr']==yr]
        if len(dr)>=3:
            yr_ret = np.prod(1+np.array(dr))-1
            print(f"{yr_ret*100:>+8.1f}%", end=' ')
        else:
            print(f"{'':>9s}", end=' ')
    print()

# 波动率分布
print(f"\n--- VOL20 仓位分布 ---")
vol_positions = [x for x in all_gate_results['VOL20']['results']]
pos_values = [x['pos'] for x in vol_positions if x['pos']>0]
print(f"仓位范围: {min(pos_values)*100:.0f}% ~ {max(pos_values)*100:.0f}%")
print(f"平均仓位: {np.mean(pos_values)*100:.0f}%")
for pct_range in [(15,25),(25,40),(40,60),(60,80),(80,100)]:
    cnt = sum(1 for p in pos_values if pct_range[0]<=p*100<pct_range[1])
    print(f"  {pct_range[0]}-{pct_range[1]}%: {cnt}月")

print(f"\n总耗时: {time.time()-t0:.0f}s")
