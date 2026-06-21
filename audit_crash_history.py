# -*- coding: utf-8 -*-
"""小众历史持仓 雷股追踪 · 高效版"""
import duckdb, pandas as pd, numpy as np, time
from collections import defaultdict
t0 = time.time()

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# K线(只拿close做后续收益)
kline = con.execute("""SELECT ts_code,trade_date,close FROM kline_daily
    WHERE trade_date>='2002-01-01' ORDER BY ts_code,trade_date""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])
all_max_date = kline['trade_date'].max()

# 每只股票的最后日期
last_dates = kline.groupby('ts_code')['trade_date'].max()
kline_close = kline.set_index(['ts_code','trade_date'])['close'].sort_index()
con.close()

fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

dates = sorted(fn['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)
md_index = {d: i for i, d in enumerate(monthly_dates)}

PAIRS = [('price_rev','turnover_rev'),('price_rev','sr5'),('amihud','max_rev'),('turnover_rev','sr5')]
TOP_N = 30

# 追踪
crash_events = []       # 入选后1月<-30%或3月<-50%
gone_events = []        # 入选后12月内数据消失
all_picks = defaultdict(list)

for i, rd in enumerate(monthly_dates):
    if rd.year < 2005 or rd not in md_index: continue
    day = fn[fn['trade_date'] == rd].copy()
    if len(day) < 100: continue

    for f in ['price_rev','turnover_rev','amihud','max_rev','sr5','vp_corr']:
        if f in day.columns: day[f+'_r'] = day[f].rank(pct=True)
    day['score'] = 0
    for fa, fb in PAIRS:
        if fa+'_r' in day.columns and fb+'_r' in day.columns:
            day['score'] += day[fa+'_r'] * day[fb+'_r']

    top = day.nlargest(TOP_N, 'score')

    for _, row in top.iterrows():
        ts = row['ts_code']
        score = row['score']

        # 查后续收益(用预计算的月度close)
        fwd = {}
        try:
            close_now = kline_close.loc[(ts, rd)]
            for horizon, offset in [('1m', 1), ('3m', 3), ('6m', 6)]:
                if i+offset < len(monthly_dates):
                    fut_d = monthly_dates[i+offset]
                    try:
                        close_fut = kline_close.loc[(ts, fut_d)]
                        fwd[horizon] = float(close_fut / close_now - 1)
                    except KeyError:
                        fwd[horizon] = np.nan
        except KeyError:
            continue

        # 退市检查
        last_d = last_dates.get(ts, all_max_date)
        days_gone = (all_max_date - last_d).days

        picks = {'date': rd, 'ts': ts, 'score': score, 'fwd': fwd,
                  'last_d': last_d, 'gone': days_gone}
        all_picks[ts].append(picks)

        # 暴跌
        f1 = fwd.get('1m', 0)
        f3 = fwd.get('3m', 0)
        if (not np.isnan(f1) and f1 < -0.30) or (not np.isnan(f3) and f3 < -0.50):
            crash_events.append(picks)

# === 汇总 ===
print("=" * 70)
print("小众历史持仓 · 雷股追踪")
print("=" * 70)

total_picks = sum(len(v) for v in all_picks.values())
print(f"\n总入选: {total_picks}次 | 涉及{len(all_picks)}只股票 | {monthly_dates[0].date()}~{monthly_dates[-1].date()}")

# 1. 退市/消失
gone_stocks = [(ts, recs) for ts, recs in all_picks.items()
               if recs[0]['gone'] > 365]
print(f"\n[1] 疑似退市(最后数据>1年): {len(gone_stocks)}只")
for ts, recs in gone_stocks:
    print(f"  {ts:<12s} 入选{len(recs):>3d}次  首{recs[0]['date'].date()}  末{recs[-1]['date'].date()}  最后数据{str(recs[0]['last_d'].date())}")
    # 最后一次入选的后续收益
    last = recs[-1]
    for k, v in last['fwd'].items():
        if not np.isnan(v):
            print(f"    最后入选后{k}: {v*100:+.1f}%")

# 2. 暴跌
crash_by_stock = defaultdict(list)
for e in crash_events:
    crash_by_stock[e['ts']].append(e)

print(f"\n[2] 暴跌事件(入选后1月<-30% 或 3月<-50%): {len(crash_events)}次, 涉及{len(crash_by_stock)}只")
for ts, events in sorted(crash_by_stock.items(), key=lambda x: len(x[1]), reverse=True):
    for e in events:
        f1 = e['fwd'].get('1m', np.nan)
        f3 = e['fwd'].get('3m', np.nan)
        f6 = e['fwd'].get('6m', np.nan)
        gone = e['gone']
        flag = ' !!退市' if gone > 365 else ''
        print(f"  {e['date'].date()} {ts:<12s} 1m={f1*100:+6.1f}% 3m={f3*100:+6.1f}% 6m={f6*100:+6.1f}% score={e['score']:.3f}{flag}")

# 3. 最差20次
print(f"\n[3] 入选后最差1月收益Top20:")
all_recs = []
for ts, recs in all_picks.items():
    for r in recs:
        if '1m' in r.get('fwd', {}) and not np.isnan(r['fwd']['1m']):
            all_recs.append(r)
all_recs.sort(key=lambda x: x['fwd']['1m'])
for r in all_recs[:20]:
    f1 = r['fwd']['1m']
    f3 = r['fwd'].get('3m', np.nan)
    f6 = r['fwd'].get('6m', np.nan)
    print(f"  {r['date'].date()} {r['ts']:<12s} 1m={f1*100:+6.1f}% 3m={f3*100:+6.1f}% 6m={f6*100:+6.1f}%")

# 4. 最差股票的全程轨迹
print(f"\n[4] 问题股票详细轨迹:")
problem_stocks = set()
for ts, events in crash_by_stock.items():
    problem_stocks.add(ts)
for ts, recs in gone_stocks:
    problem_stocks.add(ts)

for ts in sorted(problem_stocks)[:10]:
    recs = all_picks[ts]
    fwd_vals = [r['fwd'].get('1m', np.nan) for r in recs]
    avg_fwd = np.nanmean(fwd_vals) if fwd_vals else 0
    print(f"  {ts:<12s} 入选{len(recs)}次  平均1月收益{avg_fwd*100:+.1f}%  "
          f"首{recs[0]['date'].date()} 末{recs[-1]['date'].date()}  gone={recs[0]['gone']}天")

print(f"\n耗时: {time.time()-t0:.0f}s")
