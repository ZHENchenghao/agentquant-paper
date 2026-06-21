# -*- coding: utf-8 -*-
"""未来函数全面审计
=================
1. 幸存者偏差: kline_daily是否包含退市股?
2. 因子时点: 月首因子用了当日数据吗?
3. 执行滑点: close买入 vs next_open买入 差多少?
4. 财报前瞻: financial_statements的report_date=截止日还是公告日?
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("未来函数 全面审计")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ===== 1. 幸存者偏差 =====
print("\n[1] 幸存者偏差检查")
# 每年有多少只股票在kline_daily中?
for yr in [2002, 2005, 2008, 2010, 2015, 2020, 2024, 2026]:
    cnt = con.execute(f"""
        SELECT COUNT(DISTINCT ts_code) FROM kline_daily
        WHERE trade_date >= DATE '{yr}-01-01' AND trade_date <= DATE '{yr}-12-31'
          AND ts_code LIKE 'sh%' OR ts_code LIKE 'sz%'
    """).fetchone()[0]
    print(f"  {yr}: {cnt}只")

# 从2002至今的股票存活率
old = con.execute("""
    SELECT COUNT(DISTINCT ts_code) FROM kline_daily
    WHERE trade_date >= DATE '2002-01-01' AND trade_date <= DATE '2002-12-31'
    AND (ts_code LIKE 'sh%' OR ts_code LIKE 'sz%')
""").fetchone()[0]
recent = con.execute("""
    SELECT COUNT(DISTINCT ts_code) FROM kline_daily
    WHERE trade_date >= DATE '2025-01-01' AND trade_date <= DATE '2025-12-31'
    AND (ts_code LIKE 'sh%' OR ts_code LIKE 'sz%')
""").fetchone()[0]
print(f"\n  2002年: {old}只 → 2025年: {recent}只")
print(f"  A股总数: 约5400只(2025)")
print(f"  2002至今消失的: 至少{max(0, old - 500)}只(从2002总数增长推算)")

# 检查: 是否有股票在2002有数据但2024+没有?
old_stocks = set()
for r in con.execute("""
    SELECT DISTINCT ts_code FROM kline_daily
    WHERE trade_date >= DATE '2002-01-01' AND trade_date <= DATE '2003-12-31'
    AND (ts_code LIKE 'sh%' OR ts_code LIKE 'sz%')
""").fetchall():
    old_stocks.add(r[0])

recent_stocks = set()
for r in con.execute("""
    SELECT DISTINCT ts_code FROM kline_daily
    WHERE trade_date >= DATE '2025-01-01'
    AND (ts_code LIKE 'sh%' OR ts_code LIKE 'sz%')
""").fetchall():
    recent_stocks.add(r[0])

disappeared = old_stocks - recent_stocks
still_here = old_stocks & recent_stocks

print(f"\n  2002-2003有数据的股票: {len(old_stocks)}只")
print(f"  2025仍有数据: {len(still_here)}只")
print(f"  消失的(可能退市): {len(disappeared)}只 ({len(disappeared)/len(old_stocks)*100:.1f}%)")

if len(disappeared) > 0:
    # 检查消失的股票是否有退市特征(最后价格很低)
    sample_disappeared = list(disappeared)[:10]
    for ts in sample_disappeared[:5]:
        last = con.execute(f"""
            SELECT MAX(trade_date), close FROM kline_daily
            WHERE ts_code='{ts}' GROUP BY close ORDER BY MAX(trade_date) DESC LIMIT 1
        """).fetchone()
        if last:
            print(f"    {ts}: 最后日期{last[0]}, 最后价{last[1]}")

# ===== 2. 因子时点检查 =====
print("\n[2] 因子时点检查")
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])

# 取一个月首日期, 看因子值是否包含了当日的return
# 方法: 取相邻两天的因子, 如果因子变化大, 说明当日数据被纳入
sample_dates = fn['trade_date'].unique()
# 取一个随机月首前后的因子
import random
random.seed(42)
test_dates = sorted(random.sample(list(sample_dates[2000:4000]), 5))

for d in test_dates:
    day_data = fn[fn['trade_date'] == d]
    prev_data = fn[fn['trade_date'] == d - pd.Timedelta(days=1)] if (d - pd.Timedelta(days=1)) in sample_dates else None

    print(f"\n  日期: {d.date()}")
    # 看sr5 (5日反转) - 这个最简单
    sr5_now = day_data['sr5'].dropna()
    if prev_data is not None:
        sr5_prev = prev_data['sr5'].dropna()
        common = sr5_now.index.intersection(sr5_prev.index)
        if len(common) > 10:
            diff = (sr5_now.loc[common] - sr5_prev.loc[common]).abs()
            pct_changed = (diff > 0.001).mean()
            print(f"    sr5变化>0.1%: {pct_changed*100:.0f}% (说明当日被纳入因子计算)")

    # 检查amihud
    amihud_now = day_data['amihud'].dropna()
    if prev_data is not None:
        amihud_prev = prev_data['amihud'].dropna()
        common = amihud_now.index.intersection(amihud_prev.index)
        if len(common) > 10:
            diff = (amihud_now.loc[common] - amihud_prev.loc[common]).abs()
            pct_changed = (diff > 0.0001).mean()
            print(f"    amihud变化>0.01%: {pct_changed*100:.0f}%")

# ===== 3. 执行时点偏差 =====
print("\n[3] 执行时点偏差: close买入 vs next_open买入")

# 取月度首日, 对比close→next_open vs open→next_open
kline = con.execute("""
    SELECT ts_code, trade_date, open, close FROM kline_daily
    WHERE trade_date >= DATE '2010-01-01' AND (ts_code LIKE 'sh%' OR ts_code LIKE 'sz%')
    ORDER BY ts_code, trade_date
""").df()
kline['trade_date'] = pd.to_datetime(kline['trade_date'])

dates = sorted(kline['trade_date'].unique())
monthly_dates = []
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates = sorted(monthly_dates)

diffs = []
for i in range(len(monthly_dates) - 1):
    cur = monthly_dates[i]; nxt = monthly_dates[i+1]
    cur_data = kline[kline['trade_date'] == cur][['ts_code', 'open', 'close']].set_index('ts_code')
    nxt_data = kline[kline['trade_date'] == nxt][['ts_code', 'open']].set_index('ts_code')
    common = cur_data.index.intersection(nxt_data.index)
    if len(common) < 100: continue

    # 原策略: buy at close(cur), sell at open(nxt)
    orig_ret = nxt_data.loc[common, 'open'].values / cur_data.loc[common, 'close'].values - 1
    # 现实: buy at open(cur), sell at open(nxt)
    real_ret = nxt_data.loc[common, 'open'].values / cur_data.loc[common, 'open'].values - 1
    diff = np.mean(orig_ret) - np.mean(real_ret)
    diffs.append(diff)

if diffs:
    avg_diff = np.mean(diffs) * 100
    print(f"  原策略(close买入) vs 现实(open买入) 月均偏差: {avg_diff:+.3f}%")
    print(f"  年化偏差: {avg_diff*12:+.2f}%")
    print(f"  (正值=原策略高估收益)")

# ===== 4. 财报前瞻偏差 =====
print("\n[4] 财报公告延迟检查")
fin = con.execute("""
    SELECT ts_code, report_date, report_type FROM financial_statements
    WHERE ts_code = '000001.SZ' ORDER BY report_date
""").df()
fin['report_date'] = pd.to_datetime(fin['report_date'])
print(f"  000001.SZ 财报日期(前5条):")
for _, r in fin.head(5).iterrows():
    print(f"    {r['report_date'].date()} type={r['report_type']}")

# 所有年报的report_date分布
annual_dates = con.execute("""
    SELECT report_date, COUNT(*) as cnt FROM financial_statements
    WHERE report_type='annual' AND report_date >= DATE '2015-01-01'
    GROUP BY report_date ORDER BY report_date
""").df()
annual_dates['report_date'] = pd.to_datetime(annual_dates['report_date'])
annual_dates['month'] = annual_dates['report_date'].dt.month
month_dist = annual_dates.groupby('month')['cnt'].sum()
print(f"\n  年报report_date的月份分布(按截止日期):")
for m, cnt in month_dist.items():
    print(f"    {int(m)}月: {int(cnt)}条")

# 如果所有年报都是12-31, 说明report_date=截止日不是公告日
dec_cnt = month_dist.get(12, 0)
total = month_dist.sum()
print(f"  12月截止占比: {dec_cnt/total*100:.0f}%")
if dec_cnt / total > 0.5:
    print(f"  ⚠️ report_date=报告截止日, 不是公告日! 用财报因子必须加4个月延迟!")

con.close()
print(f"\n耗时: {time.time()-t0:.0f}s")
