# -*- coding: utf-8 -*-
# IVOL修正: 市场模型残差 + 1/Price重测
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()
print('IVOL修正: 市场模型残差标准差')
print('='*60)

con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)

# Step 1: 计算每只股票每天对CSI300的回归残差
# 滚动252日窗口, 逐日逐股回归 → 取残差std作为IVOL
print('[1] 加载数据...')

# 取CSI300日收益
hs300=con.execute('''
    SELECT trade_date, close/LAG(close) OVER(ORDER BY trade_date)-1 AS mkt_ret
    FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2002-01-01'
''').df()
hs300=hs300.dropna()
hs300=hs300.set_index('trade_date')['mkt_ret']

# 取所有股票日收益(抽样以减少计算量)
# 每只股需要252日滚动回归 → 计算量巨大
# 优化：用简化版——取FF3市场模型残差的滚动标准差

# 方法: 直接用SQL窗口函数!
# IVOL_252d = std(ret - beta*mkt_ret) over 252d
# 其中beta用滚动回归估计太慢 → 用滚动corr * std(ret)/std(mkt)近似

print('[2] 计算IVOL (滚动252日市场模型残差标准差)...')
# 使用简化但准确的近似:
# IVOL ≈ std(ret) * sqrt(1 - corr(ret, mkt_ret)^2)
# 这等价于市场模型残差的标准差

all_data=con.execute('''
WITH rets AS (
    SELECT s.ts_code, s.trade_date,
           s.close/LAG(s.close) OVER(PARTITION BY s.ts_code ORDER BY s.trade_date)-1 AS ret,
           x.close/LAG(x.close) OVER(ORDER BY x.trade_date)-1 AS mkt_ret
    FROM kline_daily s
    JOIN kline_daily x ON s.trade_date=x.trade_date AND x.ts_code='sh000300'
    WHERE s.trade_date>='2002-01-01'
),
rolling AS (
    SELECT ts_code, trade_date, ret, mkt_ret,
           STDDEV_SAMP(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS total_vol,
           CORR(ret, mkt_ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS mkt_corr
    FROM rets WHERE ret IS NOT NULL AND mkt_ret IS NOT NULL
)
SELECT ts_code, trade_date,
       total_vol * SQRT(GREATEST(1.0 - mkt_corr*mkt_corr, 0.01)) AS ivol_mkt,
       total_vol,
       mkt_corr,
       1.0/NULLIF(close,0) AS inv_price
FROM rolling
JOIN (SELECT ts_code, trade_date, close FROM kline_daily WHERE trade_date>='2002-01-01') px
  USING (ts_code, trade_date)
WHERE total_vol IS NOT NULL AND mkt_corr IS NOT NULL
''').df()
all_data['trade_date']=pd.to_datetime(all_data['trade_date'])
print(f'  数据: {len(all_data):,}行')

# 目标
target=con.execute('''
    SELECT s.ts_code, s.trade_date,
           (LEAD(s.close,20) OVER(PARTITION BY s.ts_code ORDER BY s.trade_date)/s.close-1)
           -(LEAD(x.close,20) OVER(ORDER BY x.trade_date)/x.close-1) AS excess_ret_20d
    FROM kline_daily s
    JOIN kline_daily x ON s.trade_date=x.trade_date AND x.ts_code='sh000300'
    WHERE s.trade_date>='2002-01-01'
''').df()
target['trade_date']=pd.to_datetime(target['trade_date'])
con.close()

merged=all_data.merge(target,on=['ts_code','trade_date'],how='inner').dropna()
print(f'  合并: {len(merged):,}行\n')

# Step 3: IC测试
print('[3] RankIC...')

def test_factor(col, name, logic):
    daily_ic=merged.groupby('trade_date').apply(
        lambda g: pd.Series({'rankic': g[col].rank().corr(g['excess_ret_20d'].rank()) if len(g)>=30 else np.nan})
    ).reset_index().dropna(subset=['rankic'])
    daily_ic['year']=daily_ic['trade_date'].dt.year

    avg_ic=daily_ic['rankic'].mean(); std_ic=daily_ic['rankic'].std()
    ir=avg_ic/std_ic if std_ic>0 else 0
    pos_pct=(daily_ic['rankic']>0).mean()*100
    neg_yrs=sum(1 for yr in range(2002,2027) if len(daily_ic[daily_ic['year']==yr])>20 and daily_ic[daily_ic['year']==yr]['rankic'].mean()<0)

    # 判定: 期待IVOL为负IC(低IVOL高收益)
    if abs(ir)>0.3 and abs(avg_ic)>0.02:
        v='✅ 强有效'
    elif abs(ir)>0.15 and abs(avg_ic)>0.01:
        v='⚠ 边缘'
    else:
        v='❌ 无效'

    print(f'{name}: IC={avg_ic:+.4f} IR={ir:+.2f} IC>0={pos_pct:.1f}% 负年={neg_yrs} {v}')
    print(f'  {logic}')
    for yr in [2004,2010,2016,2022]:
        d=daily_ic[daily_ic['year']==yr]
        if len(d)>20:
            ic_yr=d['rankic'].mean()
            print(f'  {yr}: {ic_yr:+.4f}',end='')
    print()
    return avg_ic, ir, v

# IVOL (纠正后): 市场模型残差标准差, 预期负IC
test_factor('ivol_mkt', 'IVOL(市场模型)', '低特质波动→高收益(Ang2006)')

# 对比原始总波动
test_factor('total_vol', '总波动(对照)', '总波动≠特质波动')

# 1/Price
test_factor('inv_price', '1/Price', '低价股→低收益')

# 相关性
print('\n[4] 因子相关性...')
valid=merged[['ivol_mkt','total_vol','inv_price','excess_ret_20d']].dropna()
corr=valid[['ivol_mkt','total_vol','inv_price']].corr(method='spearman')
print(corr.to_string())

print(f'\n耗时: {time.time()-t0:.0f}s')
