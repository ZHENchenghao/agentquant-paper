# -*- coding: utf-8 -*-
"""IR加权 vs 等权"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
ind_map = con.execute('SELECT ts_code, ind_name FROM stock_industry').df().rename(columns={'ind_name':'industry'})
kline = con.execute("""SELECT ts_code,trade_date,open,close,COALESCE(close*total_share/10000,close*vol/1000000) AS mcap,
    close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d
FROM kline_daily WHERE trade_date>='2005-01-01'""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
kline = kline.merge(ind_map, on='ts_code', how='left')
hs300 = con.execute("""SELECT trade_date,close FROM kline_daily
    WHERE ts_code='sh000300' AND trade_date>='2005-01-01' ORDER BY trade_date""").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['high_2y'] = hs300['close'].rolling(504).max()
con.close()

kline['month'] = kline['trade_date'].dt.to_period('M')
im = kline.groupby(['industry','month'])['close'].last().reset_index()
im['month'] = im['month'].dt.to_timestamp()
im['ind_ret_1m'] = im.groupby('industry')['close'].pct_change()
fn['month'] = fn['trade_date'].dt.to_period('M')
fn['month'] = fn['month'].dt.to_timestamp()
fn = fn.merge(ind_map, on='ts_code', how='left').merge(im[['industry','month','ind_ret_1m']], on=['industry','month'], how='left')
fn['ind_mom'] = fn.groupby('month')['ind_ret_1m'].rank(pct=True)
fn = fn.dropna(subset=['ind_mom'])

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

rd_map = {}
for i in range(len(monthly_dates)-1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cp = kline[kline['trade_date']==cur][['ts_code','close','mcap','ret_1d']].set_index('ts_code')
    np_ = kline[kline['trade_date']==nxt][['ts_code','open']].rename(columns={'open':'next_open'}).set_index('ts_code')
    m = cp.join(np_, how='inner'); m['fwd_ret'] = m['next_open']/m['close']-1; rd_map[cur] = m
del kline; gc.collect()

hs300_m = {}
for _, r in hs300.iterrows():
    hs300_m[r['trade_date']] = {'close':r['close'],'high_2y':r['high_2y']}
for d in monthly_dates:
    if d not in hs300_m:
        nb = hs300[hs300['trade_date']<=d]
        if len(nb)>0: hs300_m[d] = {'close':nb.iloc[-1]['close'],'high_2y':nb.iloc[-1]['high_2y']}

YEARS = sorted(set(d.year for d in monthly_dates))
TRAIN = 3; FIRST = YEARS[0]+TRAIN
MCAP_FLOOR = 0.20; LIMIT_UP = 0.095; TOP_N = 30; COST = 0.0033
EXIT_THRESH = -0.12; REENTRY_THRESH = 0.10; FLOOR = 0.10

ALL_PAIRS = [('price_rev','turnover_rev'),('amihud','max_rev'),('amihud','price_rev'),
    ('max_rev','price_rev'),('turnover_rev','ind_mom'),('price_rev','ind_mom'),
    ('max_rev','ind_mom'),('amihud','ind_mom')]

def gate_fn(d, st):
    if d not in hs300_m: return 1.0, st
    info = hs300_m[d]; c = info['close']; h2 = info.get('high_2y', c)
    if pd.isna(h2): return 1.0, st
    if st['in']:
        if c/h2-1 < EXIT_THRESH: return FLOOR, {'in': False}
        return 1.0, st
    else:
        rc = c/h2 if h2>0 else 1
        if rc > REENTRY_THRESH: return 0.7, {'in': True}
        return FLOOR, st

print('IR加权 vs 等权')
for weight_mode in ['equal', 'ir_weighted']:
    results = []; state = {'in': True}
    for yr in range(FIRST, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{yr-TRAIN}-01-01')
        train_e = pd.Timestamp(f'{yr-1}-12-31')
        test_mds = [d for d in monthly_dates if d.year==yr]

        pair_ir = {}
        for fa, fb in ALL_PAIRS:
            sp = []
            for rd in [d for d in monthly_dates if train_s<=d<=train_e]:
                if rd not in rd_map: continue
                day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
                valid = set(px.index); day = day[day['ts_code'].isin(valid)]
                if len(day)<50: continue
                for f in [fa,fb]:
                    if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
                if fa+'_r' not in day.columns or fb+'_r' not in day.columns: continue
                day['score'] = day[fa+'_r']*day[fb+'_r']
                px_m = px.loc[day['ts_code'].values]
                day['fwd_ret'] = px_m['fwd_ret'].values
                vd = day.dropna(subset=['score','fwd_ret'])
                if len(vd)<50: continue
                nq = int(len(vd)*0.2)
                sp.append(vd.nlargest(nq,'score')['fwd_ret'].mean()-vd.nsmallest(nq,'score')['fwd_ret'].mean())
            if len(sp)>=6: mu=np.mean(sp); std=np.std(sp); pair_ir[(fa,fb)]=mu/std if std>0 else 0

        sorted_pairs = sorted(pair_ir.items(), key=lambda x: x[1], reverse=True)
        active = [(p,ir) for p,ir in sorted_pairs[:4] if ir>0]
        if len(active)<2: active = [(p,ir) for p,ir in sorted_pairs[:2]]
        total_ir = sum(abs(ir) for _,ir in active)
        if total_ir==0: total_ir=1

        for rd in test_mds:
            if rd not in rd_map: continue
            pos, state = gate_fn(rd, state)
            if pos<0.01: results.append(0.0); continue
            day = fn[fn['trade_date']==rd].copy(); px = rd_map[rd]
            valid = set(px.index); day = day[day['ts_code'].isin(valid)]
            if len(day)<50: continue
            all_f = list(set([x for p,_ in active for x in p]))
            for f in all_f:
                if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
            day['score'] = 0; n_valid = 0
            for (fa,fb), ir in active:
                if fa+'_r' in day.columns and fb+'_r' in day.columns:
                    w = abs(ir)/total_ir if weight_mode=='ir_weighted' else 1.0
                    day['score'] += day[fa+'_r']*day[fb+'_r']*w
                    n_valid += (1 if weight_mode=='ir_weighted' else 0)
            if weight_mode == 'equal':
                n_active = 0
                for (fa2, fb2), _ in active:
                    if fa2+'_r' in day.columns and fb2+'_r' in day.columns:
                        n_active += 1
                if n_active > 0:
                    day['score'] /= n_active
            px_m = px.loc[day['ts_code'].values]
            day['mcap'] = px_m['mcap'].values; day['ret_1d'] = px_m['ret_1d'].values
            day['fwd_ret'] = px_m['fwd_ret'].values; day['mcap_r'] = day['mcap'].rank(pct=True)
            day = day[day['mcap_r']>=MCAP_FLOOR]; day = day[day['ret_1d']<LIMIT_UP]
            day = day[day['fwd_ret'].notna()]
            if len(day)<50: continue
            top = day.nlargest(TOP_N,'score')
            if len(top)>=10: results.append((top['fwd_ret'].mean()-COST)*pos)

    r_all = np.array(results)
    ann = np.mean(r_all)*12; vol = np.std(r_all)*np.sqrt(12)
    sh = ann/vol if vol>0 else 0
    cum = np.cumprod(1+r_all); mdd = np.min(cum/np.maximum.accumulate(cum)-1)
    total_cum = np.prod(1+r_all)-1
    print(f'  {weight_mode}: 年化{ann*100:+5.1f}% Sharpe{sh:+5.2f} MDD{mdd*100:+5.1f}% 累积{total_cum:+6.1%}')

print(f'耗时: {time.time()-t0:.0f}s')
