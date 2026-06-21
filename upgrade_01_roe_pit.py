# -*- coding: utf-8 -*-
"""升级#1: ROE未来函数修复
用disclosure_dates的实际公告日替代financial_statements的report_date
→ 构建PIT(Point-in-Time) ROE因子 → WF回测验证是否还有alpha
"""
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("升级#1: ROE未来函数修复")
print("=" * 60)

# 1. 加载披露日期映射
print('[1] 加载数据...')
dd = pd.read_parquet('D:/AgentQuant/our/cache/ts/disclosure_dates.parquet')
dd['ann_date'] = pd.to_datetime(dd['ann_date'], errors='coerce')
dd['end_date'] = pd.to_datetime(dd['end_date'], errors='coerce')
# 过滤有效数据
dd = dd.dropna(subset=['ann_date', 'end_date'])
print(f'  disclosure_dates: {len(dd)}行, {dd.ts_code.nunique()}只股票')

# 2. 加载财务数据
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
fs = con.execute("""
    SELECT ts_code, report_date, roe, net_profit, shareholders_equity, revenue, eps
    FROM financial_statements
    WHERE report_date >= '2005-01-01'
    ORDER BY ts_code, report_date
""").df()
fs['report_date'] = pd.to_datetime(fs['report_date'])
print(f'  financial_statements: {len(fs)}行, {fs.ts_code.nunique()}只')

# 3. 构建PIT ROE映射
# 对于每个(ts_code, report_date), 找到对应的实际公告日
print('[2] 构建PIT映射...')
def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

dd['ts_code_norm'] = dd['ts_code'].apply(norm)
fs['ts_code_norm'] = fs['ts_code'].apply(norm)

# disclosure_dates的end_date是报告期截止日
# 映射: (ts_code, end_date) -> ann_date
pit_map = {}
for _, r in dd.iterrows():
    key = (r['ts_code_norm'], r['end_date'])
    if key not in pit_map or r['ann_date'] < pit_map[key]:
        pit_map[key] = r['ann_date']
print(f'  PIT映射: {len(pit_map)}条')

# 给fs打上实际公告日
ann_dates = []
for _, r in fs.iterrows():
    key = (r['ts_code_norm'], r['report_date'])
    ann_dates.append(pit_map.get(key, r['report_date']))  # fallback到report_date
fs['ann_date'] = pd.to_datetime(ann_dates)
print(f'  有公告日的: {(fs.ann_date != fs.report_date).sum()}行')
print(f'  公告日=报告日(fallback): {(fs.ann_date == fs.report_date).sum()}行')

# 4. 构建每月可用的最新ROE
print('[3] 构建月度PIT ROE...')
# 对每个月底, 取已公告的最新财报ROE
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
fn['month'] = fn['trade_date'].dt.to_period('M')
fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

months = sorted(fn['month'].unique())
print(f'  月数: {len(months)}, {months[0].date()} ~ {months[-1].date()}')

# 构建PIT ROE序列
# 对于每只股票和每个月底, 取ann_date <= 月底的最新roe
pit_roe_data = []
for i, m in enumerate(months):
    m_ts = pd.Timestamp(m)
    # 对于每只股票, 该月前已公告的最新财报
    fs_m = fs[fs['ann_date'] <= m_ts].copy()
    if len(fs_m) == 0:
        continue
    # 每只股票取最新的
    latest = fs_m.sort_values('report_date').groupby('ts_code_norm').last().reset_index()
    latest['month'] = m_ts
    pit_roe_data.append(latest[['ts_code_norm', 'month', 'roe', 'eps']])
    if (i+1) % 50 == 0:
        print(f'  {i+1}/{len(months)}')

pit = pd.concat(pit_roe_data, ignore_index=True)
pit['roe'] = pit['roe'].clip(-1, 1)
print(f'  PIT ROE: {len(pit)}行, {pit.ts_code_norm.nunique()}只')

# 5. 合并到因子表, 做IC测试
print('[4] IC测试...')
fn_pit = fn.merge(pit, on=['ts_code_norm', 'month'], how='left')
fn_pit = fn_pit.dropna(subset=['roe'])

# 按年做滚动IC
years = sorted(fn_pit['month'].dt.year.unique())
yearly_ic = []
for yr in years:
    yr_data = fn_pit[fn_pit['month'].dt.year == yr]
    if len(yr_data) < 100:
        continue
    for m in yr_data['month'].unique():
        md = yr_data[yr_data['month'] == m]
        if len(md) < 50:
            continue
        # PIT ROE rank vs fwd ret
        md = md.dropna(subset=['roe', 'fwd_ret_1m']) if 'fwd_ret_1m' in md.columns else md
        # Use price_rev as proxy for next-month return if fwd_ret not available
        target = 'price_rev' if 'fwd_ret_1m' not in md.columns else 'fwd_ret_1m'
        if target not in md.columns:
            continue
        md = md.dropna(subset=[target])
        if len(md) < 50:
            continue
        ic = md['roe'].rank().corr(md[target].rank())
        if not np.isnan(ic):
            yearly_ic.append({'year': yr, 'month': m, 'ic': ic})

if yearly_ic:
    ic_df = pd.DataFrame(yearly_ic)
    avg_ic = ic_df['ic'].mean()
    ic_ir = avg_ic / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0
    ic_t = ic_ir * np.sqrt(len(ic_df))
    print(f'  PIT ROE IC: 均值{avg_ic*100:+.2f}% IR{ic_ir:+.3f} t{ic_t:+.2f} ({len(ic_df)}月)')

    # 年份间稳定性
    yr_avg = ic_df.groupby('year')['ic'].mean()
    for yr, ic_val in yr_avg.items():
        pos = 'POS' if ic_val > 0 else 'NEG'
        print(f'    {yr}: {ic_val*100:+5.2f}% [{pos}]')
else:
    print('  No valid IC data')

# 6. 对比: 直接用report_date的ROE (未来函数版本)
print('\n[5] 对比: report_date ROE (含未来函数)...')
fs['month'] = fs['report_date'].dt.to_period('M')
fs['month'] = fs['month'].dt.to_timestamp()
# 简单填充: 每个月底用最新report_date的roe (这就是未来函数)
fs_monthly = fs.sort_values('report_date').groupby(['ts_code_norm', 'month']).last().reset_index()
fs_monthly = fs_monthly[['ts_code_norm', 'month', 'roe']]
fs_monthly['roe_raw'] = fs_monthly['roe'].clip(-1, 1)

fn_raw = fn.merge(fs_monthly, on=['ts_code_norm', 'month'], how='left')
fn_raw = fn_raw.dropna(subset=['roe_raw'])

raw_ic = []
for yr in years:
    yr_data = fn_raw[fn_raw['month'].dt.year == yr]
    for m in yr_data['month'].unique():
        md = yr_data[yr_data['month'] == m]
        if len(md) < 50: continue
        target = 'price_rev'
        if target not in md.columns: continue
        md = md.dropna(subset=['roe_raw', target])
        if len(md) < 50: continue
        ic = md['roe_raw'].rank().corr(md[target].rank())
        if not np.isnan(ic):
            raw_ic.append(ic)

if raw_ic:
    avg_raw = np.mean(raw_ic)
    print(f'  原始ROE IC: {avg_raw*100:+.2f}% ({len(raw_ic)}月)  <- 含未来函数')
else:
    print(f'  No valid raw IC')

con.close()
print(f'\n耗时: {time.time()-t0:.0f}s')
