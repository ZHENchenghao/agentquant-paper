# -*- coding: utf-8 -*-
# 6个遗漏因子单因子IC测试
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
t0=time.time()
con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
print('='*60)
print('遗漏6因子 单因子IC测试')
print('='*60)

# ============================================================
# 1. 计算所有6因子 + 目标
# ============================================================
print('[1] 计算因子值...')

all_factors=con.execute('''
WITH daily AS (
    SELECT ts_code, trade_date, open, high, low, close, vol, amount,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
           open/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS overnight_ret,
           close/open-1 AS intraday_ret,
           LN(GREATEST(amount,1)) AS log_amount
    FROM kline_daily WHERE trade_date>='2002-01-01'
),
-- 滚动计算
roll AS (
    SELECT ts_code, trade_date, ret, overnight_ret, intraday_ret, log_amount, close,
           -- IVOL: FF3残差简化版 = 20日回归残差标准差
           STDDEV_SAMP(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ivol_20d,
           -- 52周高点
           close/MAX(high) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS p52w,
           -- 偏度: 20日收益偏度
           (AVG(ret*ret*ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            - 3*AVG(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            *STDDEV_SAMP(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)*
            STDDEV_SAMP(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
            - POWER(AVG(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),3))
            / NULLIF(POWER(STDDEV_SAMP(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),3),0) AS skew_20d,
           -- 隔夜收益持续性: 5日平均隔夜收益
           AVG(overnight_ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS overnight_5d,
           -- 日内反转: -intraday_ret (当日日内跌→次日弹)
           -intraday_ret AS intraday_rev,
           -- 名义价格: 1/close
           1.0/NULLIF(close,0) AS inv_price,
           -- 成交量: 5日平均对数成交额
           AVG(log_amount) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS log_amount_5d
    FROM daily WHERE ret IS NOT NULL
)
SELECT ts_code, trade_date,
       ivol_20d,
       p52w,
       skew_20d,
       overnight_5d,
       intraday_rev,
       inv_price,
       log_amount_5d
FROM roll
WHERE ivol_20d IS NOT NULL AND p52w IS NOT NULL AND skew_20d IS NOT NULL
''').df()
all_factors['trade_date']=pd.to_datetime(all_factors['trade_date'])
print(f'  因子数据: {len(all_factors):,}行')

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

merged=all_factors.merge(target,on=['ts_code','trade_date'],how='inner').dropna()
print(f'  合并: {len(merged):,}行')

# ============================================================
# 2. 逐个因子RankIC
# ============================================================
print('\n[2] RankIC分析...\n')

FACTOR_MAP={
    'ivol_20d':     ('特质波动(IVOL)',     '低IVOL→高收益(Ang2006)', False),
    'p52w':         ('52周高点距离',        '离高点近→动量继续',       True),
    'skew_20d':     ('偏度',               '负偏度→风险补偿',          False),
    'overnight_5d': ('隔夜收益持续性',       '隔夜涨→日间跌(反转)',    False),
    'intraday_rev': ('日内反转',            '日内跌→次日弹',           True),
    'inv_price':    ('低价股(1/price)',     '低价→彩票→低收益',       False),
    'log_amount_5d':('对数成交额',           '大额成交→流动性好',        False),
}

results=[]
for col, (name, logic, expect_pos) in FACTOR_MAP.items():
    if col not in merged.columns: continue

    daily_ic=merged.groupby('trade_date').apply(
        lambda g: pd.Series({'rankic': g[col].rank().corr(g['excess_ret_20d'].rank()) if len(g)>=30 else np.nan})
    ).reset_index().dropna(subset=['rankic'])
    daily_ic['year']=daily_ic['trade_date'].dt.year

    avg_ic=daily_ic['rankic'].mean(); std_ic=daily_ic['rankic'].std()
    ir=avg_ic/std_ic if std_ic>0 else 0
    pos_pct=(daily_ic['rankic']>0).mean()*100
    neg_yrs=[yr for yr in range(2002,2027) if len(daily_ic[daily_ic['year']==yr])>20 and daily_ic[daily_ic['year']==yr]['rankic'].mean()<0]

    # 判定
    if abs(avg_ic)>0.02 and abs(ir)>0.3 and pos_pct>55:
        verdict='✅ 强有效'
    elif abs(avg_ic)>0.01 and abs(ir)>0.15 and pos_pct>50:
        verdict='⚠ 边缘'
    else:
        verdict='❌ 无效'

    # 如果IC方向与预期相反但绝对值大，也标注
    if expect_pos and avg_ic<-0.02 and abs(ir)>0.3:
        verdict='✅ 强有效(负向)'
    elif (not expect_pos) and avg_ic>0.02 and abs(ir)>0.3:
        verdict='✅ 强有效(正向)'

    results.append({
        'col':col, 'name':name, 'logic':logic,
        'ic':avg_ic, 'ir':ir, 'pos':pos_pct, 'neg_yrs':len(neg_yrs),
        'verdict':verdict, 'neg_yr_list':neg_yrs
    })

    print(f'{name:<16s} IC={avg_ic:+.4f} IR={ir:+.2f} IC>0={pos_pct:.1f}% 负年={len(neg_yrs)} {verdict}')
    print(f'  {logic}')

    # 分年(只显示首尾和异常年)
    for yr in [2002,2008,2015,2020,2025]:
        d=daily_ic[daily_ic['year']==yr]
        if len(d)>20:
            ic_yr=d['rankic'].mean()
            print(f'  {yr}: {ic_yr:+.4f}',end='')
    print()

# ============================================================
# 3. 汇总
# ============================================================
print('\n[3] 汇总')
print('='*60)
effective=[r for r in results if '有效' in r['verdict']]
marginal=[r for r in results if '边缘' in r['verdict']]
invalid=[r for r in results if '无效' in r['verdict']]

print(f'强有效: {len(effective)} | 边缘: {len(marginal)} | 无效: {len(invalid)}')
if effective:
    print('\n有效因子:')
    for r in effective:
        nm=r['name']; ic=r['ic']; ir_val=r['ir']; ps=r['pos']
        print(f'  {nm:<16s} IC={ic:+.4f} IR={ir_val:+.2f} IC>0={ps:.0f}%')
if marginal:
    print('\n边缘因子:')
    for r in marginal:
        nm=r['name']; ic=r['ic']; ir_val=r['ir']
        print(f'  {nm:<16s} IC={ic:+.4f} IR={ir_val:+.2f}')

print(f'\n耗时: {time.time()-t0:.0f}s')
