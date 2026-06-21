# -*- coding: utf-8 -*-
"""升级#3: 杠杆情绪因子 — margin_detail 融资余额变化率"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("升级#3: 杠杆情绪因子")
print("=" * 60)

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

# 1. 加载margin_detail
print('[1] 加载margin数据...')
md = pd.read_parquet('D:/AgentQuant/our/cache/ts/margin_detail_2015_2026.parquet')
md['trade_date'] = pd.to_datetime(md['trade_date'])
md['ts_code_norm'] = md['ts_code'].apply(norm)
print(f'  margin_detail: {len(md)}行, {md.ts_code_norm.nunique()}只, {md.trade_date.min().date()}~{md.trade_date.max().date()}')

# 2. 月度融资变化率
print('[2] 计算月度融资变化率...')
md['month'] = md['trade_date'].dt.to_period('M')
md['month'] = md['month'].dt.to_timestamp()

# 每月每只股票取首尾融资余额
monthly_margin = md.groupby(['ts_code_norm', 'month']).agg(
    rzye_first=('rzye', 'first'),
    rzye_last=('rzye', 'last'),
    rzmre_sum=('rzmre', 'sum'),
    rzche_sum=('rzche', 'sum'),
).reset_index()

# 融资变化率
monthly_margin['margin_chg'] = (monthly_margin['rzye_last'] - monthly_margin['rzye_first']) / monthly_margin['rzye_first'].clip(lower=1)
monthly_margin['margin_chg'] = monthly_margin['margin_chg'].clip(-1, 1)

# 融资净买入占比 (买入-偿还)/余额
monthly_margin['net_buy_ratio'] = (monthly_margin['rzmre_sum'] - monthly_margin['rzche_sum']) / monthly_margin['rzye_first'].clip(lower=1)
monthly_margin['net_buy_ratio'] = monthly_margin['net_buy_ratio'].clip(-1, 1)

print(f'  月度margin: {len(monthly_margin)}行, {monthly_margin.ts_code_norm.nunique()}只')
print(f'  margin_chg分布: mean={monthly_margin.margin_chg.mean():+.4f} std={monthly_margin.margin_chg.std():.4f}')

# 3. 合并因子表
print('[3] 合并因子表, IC测试...')
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
fn['month'] = fn['trade_date'].dt.to_period('M')
fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

fn_m = fn.merge(monthly_margin[['ts_code_norm','month','margin_chg','net_buy_ratio']],
                on=['ts_code_norm','month'], how='left')
fn_m = fn_m.dropna(subset=['margin_chg'])
print(f'  合并后: {len(fn_m)}行 ({len(fn_m)/len(fn)*100:.0f}%覆盖率)')

# 4. IC测试
print('[4] IC分析...')
months = sorted(fn_m['month'].unique())
ics_margin = []; ics_netbuy = []

for m in months:
    md_data = fn_m[fn_m['month']==m]
    if len(md_data) < 50: continue
    target = 'price_rev'
    if target not in md_data.columns: continue
    md_data = md_data.dropna(subset=['margin_chg', target])
    if len(md_data) < 50: continue

    ic1 = md_data['margin_chg'].rank().corr(md_data[target].rank())
    ic2 = md_data['net_buy_ratio'].rank().corr(md_data[target].rank())
    if not np.isnan(ic1): ics_margin.append(ic1)
    if not np.isnan(ic2): ics_netbuy.append(ic2)

if ics_margin:
    avg = np.mean(ics_margin); t = avg/np.std(ics_margin)*np.sqrt(len(ics_margin))
    print(f'  margin_chg IC: {avg*100:+.2f}% t={t:+.2f} ({len(ics_margin)}月)')
if ics_netbuy:
    avg = np.mean(ics_netbuy); t = avg/np.std(ics_netbuy)*np.sqrt(len(ics_netbuy))
    print(f'  net_buy_ratio IC: {avg*100:+.2f}% t={t:+.2f} ({len(ics_netbuy)}月)')

# 5. 与现有因子相关性
print('[5] 与现有因子相关性...')
base_factors = ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']
for f in base_factors:
    if f in fn_m.columns:
        corr = fn_m[['margin_chg', f]].dropna().corr().iloc[0,1]
        print(f'  margin_chg vs {f}: r={corr:+.3f}')

print(f'\n耗时: {time.time()-t0:.0f}s')
