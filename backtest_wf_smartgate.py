# -*- coding: utf-8 -*-
"""
小众战法 + 智能门禁 · Walk-Forward (2002-2026)
=============================================
对比:
  NONE      — 满仓(基准)
  DD_SMART  — 回撤出场 + 动量回场(解决2009踏空)
  DD_GRAD   — 分级仓位: 根据回撤深度 100%→70%→40%→20%(永不全空)

智能回场规则:
  出场: 沪深300 < 2年高点*0.85
  回场: 沪深300 > 52周低点*1.15 AND > MA50 → 恢复仓位
  分级: -10~-15%回撤→70%仓, -15~-25%→40%, >-25%→20%
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
print("小众战法 + 智能门禁 · 修复2009踏空")
print("=" * 70)

# ============ 加载 ============
print("[1] 加载...")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_5f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline = con.execute("""
    SELECT ts_code, trade_date, open, close,
           COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
           COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
    FROM kline_daily WHERE trade_date>='2001-07-01'
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

hs300 = con.execute("""
    SELECT trade_date, close, close/LAG(close) OVER(ORDER BY trade_date)-1 AS ret
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date
""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['ma200'] = hs300['close'].rolling(200).mean()
hs300['ma50'] = hs300['close'].rolling(50).mean()
hs300['high_1y'] = hs300['close'].rolling(252).max()
hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

# 月度调仓日
dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

print("[2] 构建价格映射...")
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1
    rd_map[cur] = m
del kline; gc.collect()

# 构建HS300月度信号表
hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0:
        r = row.iloc[0]
        hs300_m[d] = {'close':r['close'], 'ma200':r['ma200'], 'ma50':r['ma50'],
                       'high_1y':r['high_1y'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0:
            r = nearby.iloc[-1]
            hs300_m[d] = {'close':r['close'], 'ma200':r['ma200'], 'ma50':r['ma50'],
                           'high_1y':r['high_1y'], 'high_2y':r['high_2y'], 'low_1y':r['low_1y']}

def get_position_dd_smart(cur_date, state):
    """
    智能回撤门禁 v2
    state: {'in_market': bool, 'exit_date': date or None}
    返回: (position, new_state)

    出场: 沪深300 < 2年高点*0.85
    回场: 沪深300 > 1年低点*1.15 AND > MA50
    """
    if cur_date not in hs300_m:
        return 1.0, state

    info = hs300_m[cur_date]
    close = info['close']; ma50 = info['ma50']
    high_2y = info['high_2y']; low_1y = info['low_1y']

    if pd.isna(high_2y) or pd.isna(ma50):
        return 1.0, state

    if state['in_market']:
        # 在场内: 检查出场条件
        dd_2y = close / high_2y - 1
        if dd_2y < -0.20:
            return 0.2, {'in_market': False, 'exit_date': cur_date}
        elif dd_2y < -0.15:
            return 0.4, {'in_market': False, 'exit_date': cur_date}
        else:
            return 1.0, state
    else:
        # 在场外: 检查回场条件
        recovery = close / low_1y - 1 if pd.notna(low_1y) and low_1y > 0 else 0
        above_ma50 = close > ma50

        if recovery > 0.15 and above_ma50:
            # 明确回场信号: 涨超15%+站上MA50
            return 0.7, {'in_market': True, 'exit_date': None}  # 先7成仓
        elif recovery > 0.10:
            return 0.4, state  # 初步反弹, 小仓位试探
        elif recovery > 0.05 and above_ma50:
            return 0.3, state  # 弱反弹但站上均线
        else:
            return 0.15, state  # 仍在场外, 保留15%底仓不踏空


def get_position_dd_grad(cur_date):
    """
    分级仓位门禁: 永不空仓, 根据回撤深度调整
    回撤 < 10%: 满仓
    回撤 10-15%: 70%
    回撤 15-20%: 50%
    回撤 20-30%: 30%
    回撤 > 30%: 20%
    """
    if cur_date not in hs300_m: return 1.0
    info = hs300_m[cur_date]
    close = info['close']; high_2y = info['high_2y']
    if pd.isna(high_2y): return 1.0

    dd = close / high_2y - 1
    if dd > -0.10: return 1.0
    elif dd > -0.15: return 0.70
    elif dd > -0.20: return 0.50
    elif dd > -0.30: return 0.30
    else: return 0.20


def get_position_ma200_grad(cur_date):
    """
    分级MA200门禁: 根据与MA200的距离调整
    """
    if cur_date not in hs300_m: return 1.0
    info = hs300_m[cur_date]
    close = info['close']; ma200 = info['ma200']
    if pd.isna(ma200): return 1.0

    ratio = close / ma200
    if ratio > 1.05: return 1.0
    elif ratio > 1.00: return 0.80
    elif ratio > 0.95: return 0.60
    elif ratio > 0.90: return 0.40
    elif ratio > 0.85: return 0.25
    else: return 0.15


# ============ Walk-Forward ============
YEARS = sorted(set(d.year for d in monthly_dates))
FIRST_TEST_YR = YEARS[0] + TRAIN_YEARS

# 训练选对 (共享)
print("[3] 训练选对...")
fold_pairs = {}
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

# OOS测试各门禁
GATES = {
    'NONE': lambda d,s: (1.0, s),
    'DD_GRAD': lambda d,s: (get_position_dd_grad(d), s),
    'DD_SMART': get_position_dd_smart,
    'MA_GRAD': lambda d,s: (get_position_ma200_grad(d), s),
}

all_gate_results = {}

for GATE, gate_fn in GATES.items():
    print(f"\n--- {GATE} ---")
    all_results = []; state = {'in_market': True, 'exit_date': None}

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year==test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = gate_fn(rd, state)
            if pos < 0.01:
                all_results.append({'date':str(rd)[:7],'ret':0.0,'n':0,'yr':rd.year,'pos':pos})
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
    active = sum(1 for x in all_results if x['pos']>0.01)

    all_gate_results[GATE] = {
        'results': all_results, 'ann':ann, 'vol':vol, 'sharpe':sh, 'mdd':mdd,
        'win':win, 'months':len(r_all), 'active':active,
        'avg_pos':avg_pos, 'total_ret':np.prod(1+r_all)-1
    }

    # 关键年检查
    for chk_yr in [2008, 2009, 2011, 2015, 2018]:
        dr_items = [x for x in all_results if x['yr']==chk_yr]
        if len(dr_items)>=3:
            yr_ret = np.prod(1+np.array([x['ret'] for x in dr_items]))-1
            avg_p = np.mean([x['pos'] for x in dr_items])*100
            print(f"  {chk_yr}: {yr_ret*100:+.1f}% (均仓{avg_p:.0f}%)")

    print(f"  => 年化{ann*100:+.1f}% Sharpe{sh:+.2f} MDD{mdd*100:.1f}% 均仓{avg_pos:.0f}%")

# ============ 最终对比 ============
print(f"\n{'='*70}")
print("最终对比: 智能门禁")
print(f"{'='*70}")
print(f"{'方案':<12s} {'年化':>8s} {'波动':>8s} {'Sharpe':>8s} {'MDD':>8s} {'Win':>6s} {'均仓':>6s} {'活跃月':>7s} {'累计':>8s} {'Calmar':>8s}")
print("-"*88)
for g in ['NONE', 'DD_GRAD', 'DD_SMART', 'MA_GRAD']:
    r = all_gate_results[g]
    calmar = r['ann'] / abs(r['mdd']) if r['mdd'] != 0 else 0
    print(f"{g:<12s} {r['ann']*100:>+7.1f}% {r['vol']*100:>7.1f}% {r['sharpe']:>+7.2f} {r['mdd']*100:>7.1f}% {r['win']:>5.0f}% {r['avg_pos']:>5.0f}% {r['active']:>7d} {r['total_ret']*100:>+7.1f}% {calmar:>+7.2f}")

# 分年
print(f"\n--- 分年对比 (年收益%) ---")
print(f"{'年':<6s} {'NONE':>9s} {'DD_GRAD':>9s} {'DD_SMART':>9s} {'MA_GRAD':>9s}")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    print(f"{yr:<6d}", end='')
    for g in ['NONE', 'DD_GRAD', 'DD_SMART', 'MA_GRAD']:
        dr = [x['ret'] for x in all_gate_results[g]['results'] if x['yr']==yr]
        if len(dr)>=3:
            yr_ret = np.prod(1+np.array(dr))-1
            print(f"{yr_ret*100:>+8.1f}%", end=' ')
        else:
            print(f"{'':>9s}", end=' ')
    print()

# 仓位时序分析
print(f"\n--- DD_SMART 仓位变化轨迹(每年平均) ---")
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    poses = [x['pos'] for x in all_gate_results['DD_SMART']['results'] if x['yr']==yr]
    if poses:
        avg_p = np.mean(poses)*100
        min_p = min(poses)*100
        bar = '█'*int(avg_p/5) + '░'*(20-int(avg_p/5))
        print(f"  {yr}: {bar} {avg_p:.0f}% (最低{min_p:.0f}%)")

print(f"\n总耗时: {time.time()-t0:.0f}s")
