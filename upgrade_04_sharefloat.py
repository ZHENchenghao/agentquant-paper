# -*- coding: utf-8 -*-
"""升级#4: 供给冲击因子 — share_float 限售解禁"""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("升级#4: 供给冲击因子 (限售解禁)")
print("=" * 60)

def norm(c):
    c = str(c).strip()
    if '.' in c: return c.split('.')[1].lower()+c.split('.')[0]
    return c.lower()

# 1. 加载
print('[1] 加载数据...')
sf = pd.read_parquet('D:/AgentQuant/our/cache/ts/share_float_2015_2026.parquet')
print(f'  share_float: {len(sf)}行, columns: {list(sf.columns)}')
print(f'  sample:')
print(sf.head(3).to_string())

# 检查关键列
float_cols = [c for c in sf.columns if 'float' in c.lower() or 'share' in c.lower() or 'vol' in c.lower()]
date_cols = [c for c in sf.columns if 'date' in c.lower() or 'ann' in c.lower()]
print(f'\n  可能的日期列: {date_cols}')
print(f'  可能的股份列: {float_cols}')
print(f'  dtypes:\n{sf.dtypes}')

sf['ann_date'] = pd.to_datetime(sf['ann_date'])
sf['float_date'] = pd.to_datetime(sf['float_date'].astype(str).str.replace('.0','', regex=False), format='%Y%m%d', errors='coerce')
sf['ts_code_norm'] = sf['ts_code'].apply(norm)
print(f'  float_date有效: {sf.float_date.notna().sum()}/{len(sf)}')

# 2. 构建供给冲击因子
# 核心思路: 未来N个月解禁股数/流通股本 → 供给压力
print('\n[2] 构建供给冲击因子...')

# 获取每只股票的流通股本(从因子表或daily_basic)
fn = pd.read_parquet('D:/AgentQuant/our/cache/factors_orig6f_2002.parquet')
fn['trade_date'] = pd.to_datetime(fn['trade_date'])
fn['month'] = fn['trade_date'].dt.to_period('M'); fn['month'] = fn['month'].dt.to_timestamp()
fn['ts_code_norm'] = fn['ts_code'].apply(norm)

# 用daily_basic获取流通市值
db = pd.read_parquet('D:/AgentQuant/our/cache/ts/daily_basic_2005_2026.parquet')
db['trade_date'] = pd.to_datetime(db['trade_date'])
db['ts_code_norm'] = db['ts_code'].apply(norm)
db['month'] = db['trade_date'].dt.to_period('M'); db['month'] = db['month'].dt.to_timestamp()

# 每月底的流通股本取最新
monthly_circ = db.groupby(['ts_code_norm','month']).last().reset_index()
monthly_circ = monthly_circ[['ts_code_norm','month','float_share','circ_mv','total_mv']]
print(f'  月度流通股本: {len(monthly_circ)}行')

# 对每个月底, 计算未来3个月/6个月的解禁压力
# share_float: ann_date=公告日, float_date=解禁日
if 'float_date' in sf.columns:
    sf['float_date'] = pd.to_datetime(sf['float_date'])
    sf['float_share_qty'] = pd.to_numeric(sf.get('float_share', sf.get('float_share_qty', 0)), errors='coerce').fillna(0)
    if 'float_share_qty' not in sf.columns:
        # 使用vol或其他列
        for c in sf.columns:
            if 'vol' in c.lower() or 'share' in c.lower():
                sf['float_share_qty'] = pd.to_numeric(sf[c], errors='coerce').fillna(0)
                print(f'  使用 {c} 作为解禁股数')
                break

    # 构建: 对每个月底, 汇总未来3个月/6个月内将解禁的总股数
    months = sorted(fn['month'].unique())[-72:]  # 最近6年(够测试)
    supply_shock = []
    for m in months:
        m_end = m + pd.DateOffset(months=3)  # 未来3个月窗口
        # 该股票在未来3个月内的解禁量
        future_float = sf[(sf['float_date'] >= m) & (sf['float_date'] <= m_end)].copy()
        if len(future_float) == 0:
            continue
        # 按股票汇总解禁量
        float_sum = future_float.groupby('ts_code_norm')['float_share_qty'].sum().reset_index()
        float_sum['month'] = m
        supply_shock.append(float_sum)

    if supply_shock:
        ss = pd.concat(supply_shock, ignore_index=True)
        # 合并流通股本, 计算解禁比例
        ss = ss.merge(monthly_circ[['ts_code_norm','month','float_share']], on=['ts_code_norm','month'], how='left')
        ss['unlock_ratio'] = ss['float_share_qty'] / ss['float_share'].clip(lower=1)
        ss['unlock_ratio'] = ss['unlock_ratio'].clip(0, 1)
        print(f'  供给冲击因子: {len(ss)}行, unlock_ratio mean={ss.unlock_ratio.mean():.4f} max={ss.unlock_ratio.max():.4f}')

        # 3. 合并因子表并IC测试
        fn_s = fn.merge(ss[['ts_code_norm','month','unlock_ratio']], on=['ts_code_norm','month'], how='left')
        fn_s['unlock_ratio'] = fn_s['unlock_ratio'].fillna(0)  # 无解禁=0压力
        print(f'  合并后: {len(fn_s)}行')

        # IC
        months_test = sorted(fn_s['month'].unique())[-48:]  # 近4年
        ics = []
        for m in months_test:
            md = fn_s[fn_s['month']==m]
            if len(md) < 50: continue
            target = 'price_rev'
            if target not in md.columns: continue
            md = md.dropna(subset=['unlock_ratio', target])
            if len(md) < 50: continue
            ic = md['unlock_ratio'].rank().corr(md[target].rank())
            if not np.isnan(ic): ics.append(ic)

        if ics:
            avg_ic = np.mean(ics); t = avg_ic/np.std(ics)*np.sqrt(len(ics))
            print(f'\n[3] unlock_ratio IC: {avg_ic*100:+.2f}% t={t:+.2f} ({len(ics)}月)')
            if abs(avg_ic) > 0.02:
                print(f'  >>> 有意义! 解禁比例越高→未来收益越{"低" if avg_ic < 0 else "高"}')
            else:
                print(f'  >>> IC不足, 独立信号太弱')

        # 相关性
        print('[4] 与现有因子相关性...')
        for f in ['amihud','max_rev','price_rev','turnover_rev','sr5','vp_corr']:
            if f in fn_s.columns:
                corr = fn_s[['unlock_ratio',f]].dropna().corr().iloc[0,1]
                print(f'  unlock_ratio vs {f}: r={corr:+.3f}')
    else:
        print('  无有效解禁数据!')
else:
    print('  share_float没有float_date列! 列: {list(sf.columns)}')

print(f'\n耗时: {time.time()-t0:.0f}s')
