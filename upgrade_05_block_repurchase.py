# -*- coding: utf-8 -*-
"""升级#5-6: block_trade + repurchase 因子测试 (fast, small data)"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("升级#5-6: 大宗折溢价 + 回购因子")
print("=" * 60)

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

# 加载因子表
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
fn['month'] = fn['trade_date'].dt.to_period('M'); fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

# ============================================================
# 1. Block Trade 折溢价因子
# ============================================================
print('[1] Block Trade 大宗折溢价...')
bt = pd.read_parquet('D:/AgentQuant/our/cache/ts/block_trade_2015_2026.parquet')
bt['trade_date'] = pd.to_datetime(bt['trade_date'])
bt['ts_code_norm'] = bt['ts_code'].apply(norm)
bt['month'] = bt['trade_date'].dt.to_period('M'); bt['month'] = bt['month'].dt.to_timestamp()
print(f'  {len(bt)}笔交易, {bt.ts_code_norm.nunique()}只')

# 获取收盘价来计算折溢价
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
kline_sample = con.execute("""
    SELECT ts_code, trade_date, close FROM kline_daily
    WHERE trade_date >= '2015-01-01' AND trade_date <= '2026-06-21'
""").df()
kline_sample['trade_date'] = pd.to_datetime(kline_sample['trade_date'])
kline_sample['ts_code_norm'] = kline_sample['ts_code'].apply(norm)
con.close()

# 合并收盘价
bt = bt.merge(kline_sample[['ts_code_norm','trade_date','close']], on=['ts_code_norm','trade_date'], how='left')
bt['discount'] = bt['price'] / bt['close'] - 1  # 负=折价
bt['discount'] = bt['discount'].clip(-0.2, 0.1)  # 限制极端值
print(f'  有收盘价的: {bt.discount.notna().sum()}/{len(bt)}')
print(f'  折溢价分布: mean={bt.discount.mean()*100:+.2f}% std={bt.discount.std()*100:.2f}%')

# 月度聚合: 每只股票当月大宗交易平均折溢价
bt_m = bt.groupby(['ts_code_norm','month']).agg(
    bt_discount=('discount', 'mean'),
    bt_count=('vol', 'count'),
    bt_amount=('amount', 'sum')
).reset_index()
print(f'  月度: {len(bt_m)}行')

# 合并+IC
fn_bt = fn.merge(bt_m[['ts_code_norm','month','bt_discount','bt_count']], on=['ts_code_norm','month'], how='left')
fn_bt['bt_discount'] = fn_bt['bt_discount'].fillna(0)  # 无大宗=0折价

months = sorted(fn_bt['month'].unique())[-72:]
ics = []
for m in months:
    md = fn_bt[fn_bt['month']==m]
    if len(md) < 50: continue
    md = md.dropna(subset=['bt_discount','price_rev'])
    if len(md) < 50: continue
    ic = md['bt_discount'].rank().corr(md['price_rev'].rank())
    if not np.isnan(ic): ics.append(ic)

if ics:
    avg = np.mean(ics); t = avg/np.std(ics)*np.sqrt(len(ics))
    print(f'  bt_discount IC: {avg*100:+.2f}% t={t:+.2f} ({len(ics)}月)')
    if abs(avg) > 0.02:
        print(f'  >>> 有意义! 折价越大→{"差" if avg < 0 else "好"}')
    else:
        print(f'  >>> 信号不足, 不加入')

# 相关性
for f in ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']:
    if f in fn_bt.columns:
        corr = fn_bt[['bt_discount',f]].dropna().corr().iloc[0,1]
        print(f'  bt vs {f}: r={corr:+.3f}')

# ============================================================
# 2. Repurchase 回购因子
# ============================================================
print('\n[2] Repurchase 回购因子...')
rp = pd.read_parquet('D:/AgentQuant/our/cache/ts/repurchase_2015_2026.parquet')
rp['ann_date'] = pd.to_datetime(rp['ann_date'])
rp['ts_code_norm'] = rp['ts_code'].apply(norm)
rp['month'] = rp['ann_date'].dt.to_period('M'); rp['month'] = rp['month'].dt.to_timestamp()
print(f'  {len(rp)}条回购, {rp.ts_code_norm.nunique()}只')

# 回购总金额/市值 = 回购强度
rp['amount'] = pd.to_numeric(rp['amount'], errors='coerce').fillna(0)
rp_m = rp.groupby(['ts_code_norm','month']).agg(
    repurchase_cnt=('amount', 'count'),
    repurchase_amt=('amount', 'sum')
).reset_index()
print(f'  月度回购: {len(rp_m)}行')

fn_rp = fn.merge(rp_m[['ts_code_norm','month','repurchase_cnt','repurchase_amt']], on=['ts_code_norm','month'], how='left')
fn_rp['repurchase_cnt'] = fn_rp['repurchase_cnt'].fillna(0)

ics_rp = []
for m in months:
    md = fn_rp[fn_rp['month']==m]
    if len(md) < 50: continue
    md = md.dropna(subset=['repurchase_cnt','price_rev'])
    if len(md) < 50: continue
    ic = md['repurchase_cnt'].rank().corr(md['price_rev'].rank())
    if not np.isnan(ic): ics_rp.append(ic)

if ics_rp:
    avg = np.mean(ics_rp); t = avg/np.std(ics_rp)*np.sqrt(len(ics_rp))
    print(f'  repurchase IC: {avg*100:+.2f}% t={t:+.2f} ({len(ics_rp)}月)')
    if abs(avg) > 0.02:
        print(f'  >>> 有意义!')
    else:
        print(f'  >>> 样本太少, 不加入')

print(f'\n耗时: {time.time()-t0:.0f}s')
