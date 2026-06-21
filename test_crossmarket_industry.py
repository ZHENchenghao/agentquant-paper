# -*- coding: utf-8 -*-
"""跨市场传导 · WTI/铜/美债→行业方向预测 WF检验
==============================================
测试: 宏观变量变动能否预测行业指数未来方向
方法: 每月末计算宏观变量1月变化 → 对行业打分 → WF IC
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 60)
print("跨市场传导 · 宏观→行业方向预测")
print("=" * 60)

# === 1. 加载数据 ===
con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 行业指数
ind = con.execute("""
    SELECT industry, trade_date, close
    FROM proxy_industry_daily
    WHERE trade_date >= DATE '2010-01-01'
    ORDER BY industry, trade_date
""").df()
ind['trade_date'] = pd.to_datetime(ind['trade_date'])

# 宏观
macro = con.execute("""
    SELECT trade_date, vix, usdcny, m1_growth, m2_growth, spx, gold, wti
    FROM macro_indicators ORDER BY trade_date
""").df()
macro['trade_date'] = pd.to_datetime(macro['trade_date'])
macro = macro.set_index('trade_date')

# FRED
fred = pd.read_csv('D:/AgentQuant/our/cache/macro_fred.csv', parse_dates=['DATE']).set_index('DATE')
fred2 = pd.read_csv('D:/AgentQuant/our/cache/macro_fred_v2.csv', parse_dates=['DATE']).set_index('DATE')
for c in fred2.columns:
    if c not in fred.columns:
        fred[c] = fred2[c]
con.close()

print(f"[1] 行业: {ind['industry'].nunique()}个, 宏观: {len(macro.columns)}变量, FRED: {len(fred.columns)}变量")

# === 2. 月度对齐 ===
ind['month'] = ind['trade_date'].dt.to_period('M')
monthly_ind = ind.groupby(['industry', 'month'])['close'].last().reset_index()
monthly_ind['ret_1m'] = monthly_ind.groupby('industry')['close'].pct_change()
monthly_ind['fwd_ret'] = monthly_ind.groupby('industry')['ret_1m'].shift(-1)
monthly_ind['month'] = monthly_ind['month'].dt.to_timestamp()

# 宏观月度
def to_monthly(series, mdates):
    """把日频对齐到月末"""
    s = series.sort_index().dropna()
    result = {}
    for d in mdates:
        vals = s[s.index <= d]
        if len(vals) > 0:
            result[d] = vals.iloc[-1]
    return pd.Series(result)

mdates = sorted(monthly_ind['month'].unique())
macro_m = {}
for col in ['wti', 'spx', 'gold', 'vix', 'usdcny']:
    if col in macro.columns:
        macro_m[col.upper()] = to_monthly(macro[col], mdates)
for col in ['WTI', 'VIX_fred', 'US10Y', 'DXY', 'T10Y2Y', 'Copper']:
    if col in fred.columns:
        name = col.upper().replace('_FRED', '')
        macro_m[name] = to_monthly(fred[col], mdates)

# 衍生
if 'M1' not in macro_m:
    macro_m['M1M2'] = to_monthly(macro['m1_growth'] - macro['m2_growth'], mdates)

# 构建DataFrame
macro_df = pd.DataFrame(macro_m)
# 变化率
for c in macro_df.columns:
    macro_df[c + '_1m'] = macro_df[c].pct_change()
    macro_df[c + '_3m'] = macro_df[c].pct_change(3)
    # z-score (12月滚动)
    macro_df[c + '_z'] = (macro_df[c] - macro_df[c].rolling(12).mean()) / macro_df[c].rolling(12).std().replace(0, 1)

macro_df = macro_df.dropna(how='all')
print(f"[2] 宏观月度: {len(macro_df)}月, {len(macro_df.columns)}特征")

# === 3. 合并+Walk-Forward ===
macro_df.index = pd.to_datetime(macro_df.index)
monthly_ind['month'] = pd.to_datetime(monthly_ind['month'])
merged = monthly_ind.merge(macro_df, left_on='month', right_index=True, how='inner')
merged = merged.dropna(subset=['fwd_ret'])

# 宏观特征列
macro_cols = [c for c in macro_df.columns if '_1m' in c or '_z' in c or '_3m' in c]
macro_cols = [c for c in macro_cols if c in merged.columns]
print(f"[3] 合并: {len(merged)}行, {len(macro_cols)}宏观特征")

# === 4. WF IC ===
YEARS = sorted(set(d.year for d in merged['month']))
TRAIN_YEARS = 5
FY = YEARS[0] + TRAIN_YEARS + 1

print(f"\n{'='*60}")
print(f"WF IC检验 ({FY}-{YEARS[-1]})")
print(f"{'='*60}")

# 测试每个宏观变量对行业收益的预测力
macro_ic = {}
for col in macro_cols:
    ics = []
    for test_yr in range(FY, YEARS[-1] + 1):
        ts = pd.Timestamp(f'{test_yr}-01-01')
        te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            if len(grp) > 5:
                ic = grp[col].rank().corr(grp['fwd_ret'].rank())
                if not np.isnan(ic):
                    ics.append(ic)
    if ics:
        macro_ic[col] = (np.mean(ics), np.std(ics), len(ics))

# 排序输出
print("\n宏观特征→行业下月收益 IC排名:")
print(f"{'特征':<30s} {'IC均值':>8s} {'IR':>8s} {'月数':>6s}")
print("-" * 55)
for col, (mi, std, n) in sorted(macro_ic.items(), key=lambda x: abs(x[1][0]), reverse=True)[:20]:
    ir = mi / std * np.sqrt(12) if std > 0 else 0
    print(f"{col:<30s} {mi:+8.4f} {ir:+8.2f} {n:>6d}")

# === 5. 传导逻辑验证 ===
print(f"\n{'='*60}")
print("传导逻辑验证(基于理论预期方向)")
print(f"{'='*60}")

# 行业分组映射
IND_GROUPS = {
    '有色金属': ['有色金属'],
    '石油石化': ['石油石化'],
    '煤炭': ['煤炭'],
    '银行': ['银行'],
    '电子': ['电子'],
    '电力设备': ['电力设备'],
    '国防军工': ['国防军工'],
}

# 理论传导
THEORIES = [
    ('WTI_1m', '有色金属', '+', '油价涨→有色跟涨'),
    ('WTI_1m', '煤炭', '+', '油价涨→替代需求→煤价涨'),
    ('WTI_1m', '石油石化', '+', '油价涨→三桶油受益'),
    ('VIX_FRED_1m', '银行', '-', 'VIX涨→避险→银行跌'),
    ('US10Y_1m', '电子', '-', '利率升→杀成长估值'),
    ('US10Y_1m', '电力设备', '-', '利率升→新能源估值承压'),
    ('DXY_1m', '有色金属', '-', '美元涨→大宗跌→有色跌'),
    ('GOLD_1m', '有色金属', '+', '金价涨→黄金股跟涨'),
]

for macro_var, ind_name, expected_dir, logic in THEORIES:
    if macro_var not in merged.columns:
        continue

    ind_data = merged[merged['industry'] == ind_name].dropna(subset=[macro_var, 'fwd_ret'])
    if len(ind_data) < 20:
        print(f"  {macro_var}→{ind_name}: 数据不足({len(ind_data)}月)")
        continue

    # 计算该行业时序IC
    ic_ts = []
    for test_yr in range(FY, YEARS[-1] + 1):
        ts = pd.Timestamp(f'{test_yr}-01-01')
        te = pd.Timestamp(f'{test_yr}-12-31')
        test = ind_data[(ind_data['month'] >= ts) & (ind_data['month'] <= te)]
        if len(test) > 3:
            ic = test[macro_var].corr(test['fwd_ret'])
            if not np.isnan(ic):
                ic_ts.append(ic)

    if ic_ts:
        mi = np.mean(ic_ts)
        direction = '+' if mi > 0 else '-'
        hit = (direction == expected_dir)
        print(f"  {macro_var}→{ind_name}: IC={mi:+.3f} 预期{expected_dir} 实际{direction} {'✅' if hit else '⚠️'} ({logic})")
    else:
        print(f"  {macro_var}→{ind_name}: 无法计算")

# === 6. 简单叠加测试 ===
print(f"\n{'='*60}")
print("宏观叠加·行业选择测试")
print(f"{'='*60}")

# 选最强的3个宏观特征，等权打分选Top5行业
top_macros = [c for c, (mi, std, n) in sorted(macro_ic.items(), key=lambda x: abs(x[1][0]), reverse=True)[:3] if abs(mi) > 0.02]
print(f"入选宏观特征: {top_macros}")

if top_macros:
    # 计算综合z-score
    for c in top_macros:
        merged[c + '_rank'] = merged.groupby('month')[c].rank(pct=True)
    rank_cols = [c + '_rank' for c in top_macros]
    merged['macro_score'] = merged[rank_cols].mean(axis=1)

    # WF: Top5行业等权
    long_ret = []; eq_ret = []
    for test_yr in range(FY, YEARS[-1] + 1):
        ts = pd.Timestamp(f'{test_yr}-01-01')
        te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            if len(grp) > 5:
                top5 = grp.nlargest(5, 'macro_score')
                long_ret.append(top5['fwd_ret'].mean())
                eq_ret.append(grp['fwd_ret'].mean())

    if long_ret:
        long_cum = np.prod(1 + np.array(long_ret))
        eq_cum = np.prod(1 + np.array(eq_ret))
        long_avg = np.mean(long_ret) * 100
        eq_avg = np.mean(eq_ret) * 100
        dh = np.mean(np.array(long_ret) > np.array(eq_ret))

        print(f"\n宏观打分Top5 vs 等权:")
        print(f"  Top5月均: {long_avg:+.2f}%  等权月均: {eq_avg:+.2f}%")
        print(f"  Top5累积: {long_cum-1:+.1%}  等权累积: {eq_cum-1:+.1%}")
        print(f"  跑赢比例: {dh*100:.1f}%")

print(f"\n耗时: {time.time()-t0:.0f}s")
