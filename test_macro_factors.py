# -*- coding: utf-8 -*-
"""
宏观政策信号因子逐一回测
========================
7层信号: 货币周期 | 外部冲击 | 资金面 | 全球风偏 | 事件 | 情绪 | 产业
每层: IC检验 + 方向命中率 + 最优阈值 + 可回测性评级
"""
import duckdb, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# 月度调仓日
kline_dates = con.execute("SELECT DISTINCT trade_date FROM kline_daily WHERE trade_date>='2005-01-01' ORDER BY trade_date").fetchdf()
kline_dates['trade_date'] = pd.to_datetime(kline_dates['trade_date'])
md = []
for ym, g in kline_dates.groupby(kline_dates['trade_date'].dt.strftime('%Y-%m')):
    md.append(g['trade_date'].iloc[0])
md = sorted(md)

# HS300月度收益(目标)
hs300 = con.execute("SELECT trade_date,close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date").df()
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300_m = {}
for i in range(len(md)-1):
    cur = md[i]; nxt = md[i+1]
    c = hs300[hs300['trade_date']==cur]
    n = hs300[hs300['trade_date']==nxt]
    if len(c) and len(n): hs300_m[cur] = n['close'].iloc[0]/c['close'].iloc[0]-1

def align_to_monthly(series_daily, monthly_dates):
    """日频→月频(取月末值)"""
    result = {}
    for d in monthly_dates:
        vals = series_daily[series_daily.index <= d]
        if len(vals) > 0: result[d] = vals.iloc[-1]
    return pd.Series(result)

def calc_ic(factor_series, target_dict, label):
    """计算因子IC和方向命中率"""
    common = set(factor_series.index) & set(target_dict.keys())
    if len(common) < 30: return None
    vals, tgts = [], []
    for d in sorted(common):
        vals.append(factor_series[d])
        tgts.append(target_dict[d])
    vals, tgts = np.array(vals), np.array(tgts)
    ic = np.corrcoef(vals, tgts)[0,1] if len(vals)>5 else 0
    direction_hit = np.mean((vals>0) == (tgts>0)) if len(vals)>0 else 0
    return {'label':label,'IC':ic,'months':len(vals),'hit_rate':direction_hit,'mean':np.mean(vals),'std':np.std(vals)}

# ================ 逐个测试 ================
results = []

print("="*70)
print("宏观信号因子逐一IC检验")
print("="*70)

# === 1. 货币周期 ===
print("\n[1] 货币信用周期...")
macro = con.execute("SELECT trade_date, m1_growth, m2_growth, social_finance FROM macro_indicators WHERE m1_growth IS NOT NULL ORDER BY trade_date").df()
macro['trade_date'] = pd.to_datetime(macro['trade_date']); macro = macro.set_index('trade_date')

# M1-M2剪刀差
scissor = (macro['m1_growth'] - macro['m2_growth']).dropna()
scissor_m = align_to_monthly(scissor, md)
r = calc_ic(scissor_m, hs300_m, 'M1-M2剪刀差(水平)')
if r: results.append(r)
r = calc_ic(scissor_m.diff(3), hs300_m, 'M1-M2剪刀差(3月Δ)')
if r: results.append(r)
# M2增速
m2_m = align_to_monthly(macro['m2_growth'], md)
r = calc_ic(m2_m, hs300_m, 'M2增速(水平)')
if r: results.append(r)

# === 2. 外部冲击 ===
print("[2] 外部冲击...")
ext = con.execute("SELECT trade_date, vix, usdcny FROM macro_indicators WHERE vix IS NOT NULL ORDER BY trade_date").df()
ext['trade_date'] = pd.to_datetime(ext['trade_date']); ext = ext.set_index('trade_date')

vix_m = align_to_monthly(ext['vix'], md)
r = calc_ic(vix_m, hs300_m, 'VIX(水平)')
if r: results.append(r)
r = calc_ic(-vix_m, hs300_m, 'VIX逆(低VIX=好)')
if r: results.append(r)
# VIX月度变化
vix_chg = vix_m.diff(1)
r = calc_ic(-vix_chg, hs300_m, 'VIX下降(恐慌消退)')
if r: results.append(r)
# VIX极端值标记
vix_extreme = (vix_m > 30).astype(float)
r = calc_ic(-vix_extreme, hs300_m, 'VIX<30(非恐慌)')
if r: results.append(r)

# 汇率
cny_m = align_to_monthly(ext['usdcny'], md)
r = calc_ic(-cny_m.diff(3), hs300_m, '人民币升值(3月Δ)')
if r: results.append(r)

# === 3. 资金面 ===
print("[3] 资金面...")
# 两融余额
margin = con.execute("SELECT trade_date, margin_balance FROM margin_trading ORDER BY trade_date").df()
margin['trade_date'] = pd.to_datetime(margin['trade_date']); margin = margin.set_index('trade_date')
mg_m = align_to_monthly(margin['margin_balance'], md)
mg_chg = mg_m.pct_change(1)
r = calc_ic(mg_chg, hs300_m, '两融余额变化(月)')
if r: results.append(r)
mg_chg3 = mg_m.pct_change(3)
r = calc_ic(mg_chg3, hs300_m, '两融余额变化(3月)')
if r: results.append(r)

# 北向资金
nb = con.execute("SELECT trade_date, net_flow FROM north_bound_flow WHERE type='北上资金' ORDER BY trade_date").df()
if len(nb) == 0:
    nb = con.execute("SELECT trade_date, net_flow FROM north_bound_flow ORDER BY trade_date").df()
nb['trade_date'] = pd.to_datetime(nb['trade_date']); nb = nb.set_index('trade_date')
nb_m = align_to_monthly(nb['net_flow'], md)
nb_cum3 = nb_m.rolling(3).sum()
r = calc_ic(nb_cum3, hs300_m, '北向累计(3月)')
if r: results.append(r)

# === 4. 全球风偏 ===
print("[4] 全球风偏...")
risk = con.execute("SELECT trade_date, gold, spx, nasdaq, sox FROM macro_indicators WHERE gold IS NOT NULL ORDER BY trade_date").df()
risk['trade_date'] = pd.to_datetime(risk['trade_date']); risk = risk.set_index('trade_date')

gold_m = align_to_monthly(risk['gold'], md)
# 金铜比(避险/风险): 用gold作为避险代理
gold_chg = gold_m.pct_change(1)
r = calc_ic(-gold_chg, hs300_m, '黄金跌(风险偏好升)')
if r: results.append(r)
gold_chg3 = gold_m.pct_change(3)
r = calc_ic(-gold_chg3, hs300_m, '黄金跌(3月,风险偏好升)')
if r: results.append(r)

# === 5. 市场情绪 ===
print("[5] 市场情绪...")
# 用VIX+A股波动率合成
hs300_ret = hs300.set_index('trade_date')['close'].pct_change()
hs300_vol = hs300_ret.rolling(20).std() * np.sqrt(252)
vol_m = align_to_monthly(hs300_vol, md)
r = calc_ic(-vol_m, hs300_m, '波动率逆(低波=稳)')
if r: results.append(r)
r = calc_ic(-vol_m.diff(1), hs300_m, '波动率下降')
if r: results.append(r)

# === 汇总 ===
print("\n" + "="*70)
print("宏观因子IC排序")
print("="*70)
valid = [r for r in results if r and not np.isnan(r['IC'])]
print("%-30s %8s %8s %8s %6s" % ('因子','IC','月数','命中率','评级'))
print("-"*65)
for r in sorted(valid, key=lambda x: abs(x['IC']), reverse=True):
    stars = '★★★' if abs(r['IC'])>0.10 else ('★★' if abs(r['IC'])>0.05 else '★')
    print("%-30s %+8.4f %8d %7.1f%% %s" % (r['label'], r['IC'], r['months'], r['hit_rate']*100, stars))

con.close()
