# -*- coding: utf-8 -*-
"""行业基本面因子 · 盈利动量+加速度+超预期
==========================================
从个股财报→行业聚合: 盈利增速、盈利加速度、利润率变化、超预期比
"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0 = time.time()

print("=" * 70)
print("行业基本面因子")
print("=" * 70)

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# ===== 1. 个股财报 =====
fin = con.execute("""
    SELECT ts_code, report_date, report_type, net_profit, revenue, eps,
           gross_margin, net_margin, roe
    FROM financial_statements
    WHERE net_profit IS NOT NULL AND revenue IS NOT NULL
    ORDER BY ts_code, report_date
""").df()
fin['report_date'] = pd.to_datetime(fin['report_date'])
print(f"[1] 财报: {len(fin)}行, {fin['ts_code'].nunique()}只")

# 行业映射
ind_map = con.execute("SELECT ts_code, ind_name FROM stock_industry").df()
ind_map = ind_map.rename(columns={'ind_name': 'industry'})

# 行业指数
ind_idx = con.execute("""
    SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date >= DATE '2010-01-01' ORDER BY industry, trade_date
""").df()
ind_idx['trade_date'] = pd.to_datetime(ind_idx['trade_date'])
ind_idx['month'] = ind_idx['trade_date'].dt.to_period('M')
ind_m = ind_idx.groupby(['industry', 'month'])['close'].last().reset_index()
ind_m['month'] = ind_m['month'].dt.to_timestamp()
ind_m['ret_1m'] = ind_m.groupby('industry')['close'].pct_change()
ind_m['fwd_ret'] = ind_m.groupby('industry')['ret_1m'].shift(-1)

con.close()

# ===== 2. 个股盈利指标(按报告期) =====
fin = fin.merge(ind_map[['ts_code', 'industry']], on='ts_code', how='inner')
fin = fin.sort_values(['ts_code', 'report_date'])

# 筛选年报+半年报(有完整利润表)
fin_annual = fin[fin['report_type'].isin(['annual', 'Q2'])].copy()

# 计算YoY增速(同比)
fin_annual['np_growth'] = fin_annual.groupby('ts_code')['net_profit'].pct_change(2)  # 同期
fin_annual['rev_growth'] = fin_annual.groupby('ts_code')['revenue'].pct_change(2)

# 盈利加速度: 增速的变化
fin_annual['np_accel'] = fin_annual.groupby('ts_code')['np_growth'].diff(2)

# 利润率变化
fin_annual['margin_chg'] = fin_annual.groupby('ts_code')['net_margin'].diff(2)
fin_annual['roe_chg'] = fin_annual.groupby('ts_code')['roe'].diff(2)

# 过滤极端值
for c in ['np_growth', 'rev_growth', 'np_accel', 'margin_chg', 'roe_chg']:
    fin_annual[c] = fin_annual[c].clip(-5, 5)

# ===== 3. 行业聚合(中位数, 每报告期) =====
# 每报告期→归入最近月份
# 报告日期通常在季度结束后1-4个月公布
# 简单处理: report_date直接归入月份
fin_annual['month'] = fin_annual['report_date'].dt.to_period('M')
fin_annual['month'] = fin_annual['month'].dt.to_timestamp()

ind_factors = fin_annual.groupby(['industry', 'month']).agg(
    np_growth_med=('np_growth', 'median'),
    np_growth_pos=('np_growth', lambda x: np.mean(x > 0)),
    rev_growth_med=('rev_growth', 'median'),
    np_accel_med=('np_accel', 'median'),
    margin_chg_med=('margin_chg', 'median'),
    roe_med=('roe', 'median'),
    roe_chg_med=('roe_chg', 'median'),
    n_stocks=('ts_code', 'nunique'),
).reset_index()

# 因子平滑(财报季之间有空白月→前向填充)
ind_factors = ind_factors.sort_values(['industry', 'month'])
for c in ['np_growth_med', 'np_growth_pos', 'rev_growth_med', 'np_accel_med',
          'margin_chg_med', 'roe_med', 'roe_chg_med']:
    ind_factors[c] = ind_factors.groupby('industry')[c].ffill(limit=6)

print(f"[2] 行业基本面: {len(ind_factors)}行, {ind_factors['industry'].nunique()}行业")
print(f"    时间: {ind_factors['month'].min().date()}~{ind_factors['month'].max().date()}")

# ===== 4. 合并价格数据 =====
merged = ind_m.merge(ind_factors, on=['industry', 'month'], how='left')
# ffill基本面(财报空白期)
for c in ['np_growth_med', 'np_growth_pos', 'rev_growth_med', 'np_accel_med',
          'margin_chg_med', 'roe_med', 'roe_chg_med', 'n_stocks']:
    merged[c] = merged.groupby('industry')[c].ffill(limit=6)

merged = merged.dropna(subset=['fwd_ret', 'np_growth_med'])
print(f"[3] 合并: {len(merged)}行")

# ===== 5. FACTORS =====
FACTORS = {
    'np_growth_med': ('盈利增速(中位)', 1),
    'np_growth_pos': ('盈利正增比', 1),
    'rev_growth_med': ('营收增速(中位)', 1),
    'np_accel_med': ('盈利加速度', 1),
    'margin_chg_med': ('利润率变化', 1),
    'roe_med': ('ROE水平', 1),
    'roe_chg_med': ('ROE变化', 1),
}

# ===== 6. WF IC =====
YEARS = sorted(set(d.year for d in merged['month']))
TRAIN_YEARS = 5; WF_START = YEARS[0] + TRAIN_YEARS + 1

print(f"\n[4] WF IC ({WF_START}-{YEARS[-1]})")
ic_results = {}
for f, (name, expected) in FACTORS.items():
    if f not in merged.columns: continue
    ics = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
        test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
        for m, grp in test.groupby('month'):
            valid = grp.dropna(subset=[f, 'fwd_ret'])
            if len(valid) > 5:
                ic = valid[f].rank().corr(valid['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
    if len(ics) > 10:
        mi = np.mean(ics); std = np.std(ics)
        t = mi/std*np.sqrt(len(ics)) if std>0 else 0
        ir = mi/std*np.sqrt(12) if std>0 else 0
        dir_ok = (mi>0 and expected>0) or (mi<0 and expected<0)
        ic_results[f] = {'name':name, 'ic':mi, 'ir':ir, 't':t, 'dir_ok':dir_ok}

print(f"{'因子':<20s} {'IC':>8s} {'IR':>7s} {'t':>7s} {'方向'}")
print("-" * 50)
for f, r in sorted(ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    print(f"{r['name']:<20s} {r['ic']*100:+7.2f}% {r['ir']:+6.2f} {r['t']:+6.2f} "
          f"{'OK' if r['dir_ok'] else 'XX'}")

# ===== 7. 策略WF: 基本面 vs 价格动量 =====
print(f"\n[5] 策略WF对比")
STRATEGIES = {
    '基本面(盈利+营收)': ['np_growth_med', 'rev_growth_med'],
    '基本面(盈利+加速度)': ['np_growth_med', 'np_accel_med'],
    '基本面+利润率': ['np_growth_med', 'rev_growth_med', 'margin_chg_med'],
    '纯价格动量(1m+12m)': ['ret_1m'],  # 同窗口
    '基本面+动量混合': ['np_growth_med', 'np_accel_med', 'ret_1m'],
}

# 确保ret_1m可用
merged['ret_1m'] = merged.groupby('industry')['fwd_ret'].shift(1)  # 上月收益

for sname, factors in STRATEGIES.items():
    long_r = []; ls_r = []
    for test_yr in range(WF_START, YEARS[-1]+1):
        train_s = pd.Timestamp(f'{test_yr-TRAIN_YEARS}-01-01')
        train_e = pd.Timestamp(f'{test_yr-1}-12-31')
        test_s = pd.Timestamp(f'{test_yr}-01-01'); test_e = pd.Timestamp(f'{test_yr}-12-31')

        train = merged[(merged['month']>=train_s)&(merged['month']<=train_e)]
        test = merged[(merged['month']>=test_s)&(merged['month']<=test_e)]
        if len(test)<30 or len(train)<60: continue

        dirs = {}
        for f in factors:
            if f not in train.columns or f not in test.columns: continue
            ics = []
            for m, grp in train.groupby('month'):
                v = grp.dropna(subset=[f,'fwd_ret'])
                if len(v)>5:
                    ic = v[f].rank().corr(v['fwd_ret'].rank())
                    if not np.isnan(ic): ics.append(ic)
            dirs[f] = 1 if (len(ics)>8 and np.mean(ics)>0) else -1
        if len(dirs)<1: continue

        tc = test.copy()
        for f in factors:
            if f in tc.columns and f in dirs:
                tc[f'{f}_r'] = tc.groupby('month')[f].rank(pct=True)*dirs[f]
        rc = [f'{f}_r' for f in factors if f'{f}_r' in tc.columns]
        if not rc: continue
        tc['score'] = tc[rc].mean(axis=1)

        for m, grp in tc.groupby('month'):
            if len(grp)<10: continue
            n = max(1, len(grp)//4)
            top = grp.nlargest(n,'score')
            long_r.append(top['fwd_ret'].mean()-0.003)
            ls_r.append(top['fwd_ret'].mean() - grp.nsmallest(n,'score')['fwd_ret'].mean())

    if long_r:
        la=np.array(long_r); ls=np.array(ls_r); n=len(la)
        la_cum=np.prod(1+la); la_ann=la_cum**(12/n)-1
        ls_cum=np.prod(1+ls); ls_ann=ls_cum**(12/n)-1
        la_mdd=np.min(np.cumprod(1+la)/np.maximum.accumulate(np.cumprod(1+la))-1)
        ls_sh=ls_ann/(np.std(ls)*np.sqrt(12)) if np.std(ls)>0 else 0
        print(f"  {sname:<24s} 做多{la_ann*100:+5.1f}% MDD{la_mdd*100:+5.0f}%  "
              f"多空{ls_ann*100:+5.1f}% Sh{ls_sh:+5.2f}")

# ===== 8. 分年IC: 基本面因子是否在特定年份有效 =====
print(f"\n[6] 盈利增速IC分年:")
for test_yr in range(WF_START, YEARS[-1]+1):
    ts = pd.Timestamp(f'{test_yr}-01-01'); te = pd.Timestamp(f'{test_yr}-12-31')
    test = merged[(merged['month'] >= ts) & (merged['month'] <= te)]
    yr_ics = []
    for m, grp in test.groupby('month'):
        valid = grp.dropna(subset=['np_growth_med', 'fwd_ret'])
        if len(valid) > 5:
            ic = valid['np_growth_med'].rank().corr(valid['fwd_ret'].rank())
            if not np.isnan(ic): yr_ics.append(ic)
    if yr_ics:
        print(f"  {test_yr}: IC={np.mean(yr_ics)*100:+5.2f}% ({len(yr_ics)}月)")

print(f"\n耗时: {time.time()-t0:.0f}s")
