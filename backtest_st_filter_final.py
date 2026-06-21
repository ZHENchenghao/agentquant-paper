# -*- coding: utf-8 -*-
"""小众战法 · ST过滤+暴跌止损 完整WF回测
两道防线:
  1. ST硬过滤: 排除is_st=true或已退市的股票(调仓时)
  2. 暴跌止损: 持仓股月跌>30%→下月强制清仓不补
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()
TOP_N = 30; COST = 0.0033; TRAIN_YEARS = 5; MCAP_FLOOR = 0.20; LIMIT_UP = 0.095
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10
CRASH_STOP = -0.30  # 月跌超30%→止损
FEATS = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
ALL_PAIRS = [('amihud','max_rev'),('amihud','price_rev'),('amihud','turnover_rev'),
    ('amihud','sr5'),('amihud','vp_corr'),('max_rev','price_rev'),('max_rev','turnover_rev'),
    ('max_rev','sr5'),('max_rev','vp_corr'),('price_rev','turnover_rev'),('price_rev','sr5'),
    ('price_rev','vp_corr'),('turnover_rev','sr5'),('turnover_rev','vp_corr'),('sr5','vp_corr')]

print('=' * 60)
print('小众战法 · ST过滤+暴跌止损 终验')
print('ST过滤=硬排除 止损=月跌>%.0f%%清仓' % (CRASH_STOP*100))
print('=' * 60)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 因子
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

# K线 + ST信息
kline = con.execute("""SELECT ts_code,trade_date,open,close,
    COALESCE(amount,GREATEST(vol*close,1.0)) AS amount_proxy,
    COALESCE(close*total_share/10000,GREATEST(COALESCE(amount,GREATEST(vol*close,1.0)),close*vol)/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2001-07-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

st_list = set(con.execute("SELECT ts_code FROM stock_basic WHERE is_st=true").fetchdf()['ts_code'].values)
delisted_list = set(con.execute("SELECT ts_code FROM stock_basic WHERE delist_date IS NOT NULL").fetchdf()['ts_code'].values)
print(f'ST名单: {len(st_list)}只  退市名单: {len(delisted_list)}只')

hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2001-07-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date']); hs300['high_2y'] = hs300['close'].rolling(504).max()
hs300['low_1y'] = hs300['close'].rolling(252).min()
con.close()

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

# 预计算 fwd_ret
rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m

hs300_m = {}
for d in monthly_dates:
    row = hs300[hs300['trade_date']==d]
    if len(row)>0: r = row.iloc[0]; hs300_m[d] = {'close':r['close'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}
    else:
        nearby = hs300[hs300['trade_date']<=d]
        if len(nearby)>0: r = nearby.iloc[-1]; hs300_m[d] = {'close':r['close'],'high_2y':r['high_2y'],'low_1y':r['low_1y']}

def dd_smart_v2(cur_date, state):
    if cur_date not in hs300_m: return 1.0, state
    info = hs300_m[cur_date]; close = info['close']; high_2y = info['high_2y']; low_1y = info['low_1y']
    if pd.isna(high_2y): return 1.0, state
    if state['in_market']:
        dd_2y = close/high_2y-1
        if dd_2y < EXIT_THRESH-0.05: return FLOOR, {'in_market':False,'exit_date':cur_date}
        elif dd_2y < EXIT_THRESH: return FLOOR*2, {'in_market':False,'exit_date':cur_date}
        else: return 1.0, state
    else:
        recovery = close/low_1y-1 if pd.notna(low_1y) and low_1y>0 else 0
        if recovery > REENTRY_THRESH: return 0.7, {'in_market':True,'exit_date':None}
        elif recovery > REENTRY_THRESH*0.7: return FLOOR*2, state
        elif recovery > 0.05: return FLOOR, state
        else: return FLOOR, state

YEARS = sorted(set(d.year for d in monthly_dates)); FIRST_TEST_YR = YEARS[0]+TRAIN_YEARS

# 选pair
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
            for f in [fa, fb]:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
            day['score'] = day[fa+'_r']*day[fb+'_r']
            day['fwd_ret'] = px.loc[day['ts_code'].values]['fwd_ret'].values
            vd = day.dropna(subset=['score','fwd_ret'])
            if len(vd) < 50: continue
            nq = int(len(vd)*0.2)
            spreads.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
        if len(spreads) >= 12: mu = np.mean(spreads); std = np.std(spreads); pair_ir[(fa,fb)] = mu/std if std>0 else 0
    sorted_pairs = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)
    fold_pairs[test_yr] = [p for p, ir in sorted_pairs[:4]]

# ==== 对比实验 ====
configs = {
    '无过滤(基准)': {'filter_st': False, 'crash_stop': False},
    'ST过滤': {'filter_st': True, 'crash_stop': False},
    'ST过滤+暴跌止损': {'filter_st': True, 'crash_stop': True},
}

all_cfg_results = {}
all_cfg_raw = {}

for cfg_name, cfg in configs.items():
    results = []
    state = {'in_market': True, 'exit_date': None}
    crash_count = 0; st_filtered_count = 0
    prev_portfolio = {}  # ts_code → 上月入选价(用于止损判断)

    for test_yr in range(FIRST_TEST_YR, YEARS[-1]+1):
        if test_yr not in fold_pairs: continue
        top4 = fold_pairs[test_yr]
        test_mds = [d for d in monthly_dates if d.year == test_yr]
        if len(test_mds) < 3: continue

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = dd_smart_v2(rd, state)

            # 🆕 暴跌止损: 检查上月持仓
            if cfg['crash_stop'] and prev_portfolio:
                stopped = []
                for ts_code, entry_price in prev_portfolio.items():
                    # 查本月是否有这个股票
                    if rd in rd_map and ts_code in rd_map[rd].index:
                        this_close = rd_map[rd].loc[ts_code, 'close']
                        monthly_ret = this_close / entry_price - 1
                        if monthly_ret < CRASH_STOP:
                            stopped.append(ts_code)
                crash_count += len(stopped)
                if stopped:
                    pos = max(FLOOR, pos * 0.5)  # 有止损→降低仓位

            if pos < 0.01:
                results.append({'date': str(rd)[:7], 'ret': 0.0, 'yr': rd.year, 'pos': pos})
                prev_portfolio = {}
                continue

            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day) < 100: continue

            # 🆕 ST过滤
            if cfg['filter_st']:
                before = len(day)
                day = day[~day['ts_code'].isin(st_list)]
                day = day[~day['ts_code'].isin(delisted_list)]
                st_filtered_count += before - len(day)
                if len(day) < 100: continue

            all_f = list(set([x for p in top4 for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0
            for fa, fb in top4:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    day['score'] += day[fa+'_r']*day[fb+'_r']

            px_match = px.loc[day['ts_code'].values]
            day['mcap'] = px_match['mcap'].values; day['ret_1d'] = px_match['ret_1d'].values
            day['fwd_ret'] = px_match['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]; day = day[day['fwd_ret'].notna()]
            if len(day) < 80: continue

            top = day.nlargest(TOP_N, 'score')
            if len(top) < 15: continue

            # 🆕 记录入选价(用于下月止损判断)
            if cfg['crash_stop']:
                prev_portfolio = {}
                for _, r in top.iterrows():
                    prev_portfolio[r['ts_code']] = px_match.loc[r['ts_code'], 'close']

            results.append({
                'date': str(rd)[:7], 'ret': (top['fwd_ret'].mean()-COST)*pos,
                'yr': rd.year, 'pos': pos
            })

    r_all = np.array([x['ret'] for x in results])
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12); sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    win = (r_all>0).mean()*100; total_ret = np.prod(1+r_all)-1

    all_cfg_results[cfg_name] = {
        'ann': ann, 'sh': sh, 'mdd': mdd, 'win': win, 'cum': total_ret,
        'crash_count': crash_count, 'st_filtered': st_filtered_count,
        'months': len(r_all), 'avg_pos': np.mean([x['pos'] for x in results if x['pos']>0.01])*100
    }
    all_cfg_raw[cfg_name] = results

# ==== 输出 ====
print(f'\n{"="*70}')
print(f'{"配置":<20s} {"年化":>8s} {"Sharpe":>7s} {"MDD":>7s} {"胜率":>6s} {"累积":>8s} {"均仓":>5s} {"止损次":>5s}')
print('-'*70)
for cfg_name, r in all_cfg_results.items():
    print(f'{cfg_name:<20s} {r["ann"]*100:+7.1f}% {r["sh"]:+6.2f} {r["mdd"]*100:+6.1f}% {r["win"]:>5.0f}% {r["cum"]*100:+7.1f}% {r["avg_pos"]:>4.0f}% {r["crash_count"]:>5d}')

# 分年对比
print(f'\n分年收益对比:')
print(f'{"年":>6s} {"无过滤":>8s} {"ST过滤":>8s} {"ST+止损":>8s}')
print('-'*35)
for yr in range(FIRST_TEST_YR, YEARS[-1]+1):
    yr_data = {}
    for cfg_name in configs:
        dr = [x['ret'] for x in all_cfg_raw[cfg_name] if x['yr']==yr]
        if len(dr)>=6:
            yr_data[cfg_name] = np.prod(1+np.array(dr))-1
    if len(yr_data)==3:
        print(f'{yr:>6d} {yr_data["无过滤(基准)"]*100:+7.1f}% {yr_data["ST过滤"]*100:+7.1f}% {yr_data["ST过滤+暴跌止损"]*100:+7.1f}%')

print(f'\n耗时: {time.time()-t0:.0f}s')
