# -*- coding: utf-8 -*-
"""从日K线自建行业级资金流向 → 申万30行业月度资金指纹"""
import duckdb, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
t0=time.time()

print("="*60)
print("行业级资金流向构建")
print("="*60)

con=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

# 1. 取行业映射和日K线
ind_map=con.execute("""SELECT ts_code, ind_name as industry FROM stock_industry""").df()
print("[1] 行业映射: %d stocks → %d industries"%(len(ind_map),ind_map['industry'].nunique()))

# 2. 日K线+行业标签
kline=con.execute("""SELECT ts_code, trade_date, open, high, low, close, vol, amount
    FROM kline_daily WHERE trade_date>='2010-01-01' ORDER BY ts_code, trade_date""").df()
kline['trade_date']=pd.to_datetime(kline['trade_date'])
# Merge industry
kline=kline.merge(ind_map[['ts_code','industry']],on='ts_code',how='inner')
print("[2] K线: %d rows, %d stocks"%(len(kline),kline['ts_code'].nunique()))

# 3. 个股日频资金代理指标
kline=kline.sort_values(['ts_code','trade_date'])
# 日内买入压力: (收盘-最低)/(最高-最低)
kline['buy_pressure']=(kline['close']-kline['low'])/(kline['high']-kline['low']+0.001)
kline['buy_pressure']=kline['buy_pressure'].clip(0,1)
# 量比: 成交量/20日均量
kline['vol_ma20']=kline.groupby('ts_code')['vol'].transform(lambda x:x.rolling(20).mean())
kline['vol_ratio']=kline['vol']/kline['vol_ma20'].replace(0,1)
# 成交额变化(大资金痕迹)
kline['amt_chg']=kline.groupby('ts_code')['amount'].pct_change()
# 资金流代理: 涨跌幅×成交额
kline['ret']=kline.groupby('ts_code')['close'].pct_change()
kline['money_flow']=kline['ret']*kline['amount']/1e8
# 大单代理: 量比>1.5 且 振幅>3%
kline['big_order']=((kline['vol_ratio']>1.5)&(abs(kline['ret'])>0.03)).astype(int)
kline['big_buy']=((kline['vol_ratio']>1.5)&(kline['ret']>0.03)).astype(int)
kline['big_sell']=((kline['vol_ratio']>1.5)&(kline['ret']<-0.03)).astype(int)

# 4. 汇总到行业日频
daily_ind=kline.groupby(['industry','trade_date']).agg(
    buy_pressure=('buy_pressure','mean'),
    vol_ratio=('vol_ratio','mean'),
    money_flow=('money_flow','sum'),
    big_order_pct=('big_order','mean'),
    big_buy_pct=('big_buy','mean'),
    big_sell_pct=('big_sell','mean'),
    up_pct=('ret',lambda x:(x>0).mean()),      # 上涨比例
    ret_mean=('ret','mean'),
    ret_std=('ret','std'),
    stock_cnt=('ts_code','nunique')
).reset_index()
print("[3] 行业日频: %d rows" % len(daily_ind))

# 5. 月度汇总特征
daily_ind['month']=daily_ind['trade_date'].dt.to_period('M')
monthly_ind=daily_ind.groupby(['industry','month']).agg(
    buy_pressure=('buy_pressure','mean'),
    vol_ratio=('vol_ratio','mean'),
    money_flow=('money_flow','sum'),
    big_order=('big_order_pct','mean'),
    big_buy=('big_buy_pct','mean'),
    big_sell=('big_sell_pct','mean'),
    up_pct=('up_pct','mean'),
    ret_mean=('ret_mean','mean'),
    ret_std=('ret_std','mean'),
    days=('trade_date','nunique')
).reset_index()
# 衍生特征
monthly_ind['money_flow_1m']=monthly_ind.groupby('industry')['money_flow'].shift(1)
monthly_ind['big_net']=monthly_ind['big_buy']-monthly_ind['big_sell']
monthly_ind['flow_divergence']=monthly_ind['money_flow']-monthly_ind['ret_mean']*100  # 流向vs价格背离

monthly_ind['month']=monthly_ind['month'].dt.to_timestamp()
print("[4] 行业月频: %d rows, %d months"%(len(monthly_ind),monthly_ind['month'].nunique()))

# 6. 行业前向收益(目标)
# 计算每个行业下月收益
ind_close=con.execute("""SELECT industry, trade_date, close FROM proxy_industry_daily
    WHERE trade_date>=DATE '2010-01-01' ORDER BY industry,trade_date""").df()
ind_close=ind_close.rename(columns={'stock_code':'idx_code'})
ind_close['trade_date']=pd.to_datetime(ind_close['trade_date'])
ind_close['month']=ind_close['trade_date'].dt.to_period('M')
monthly_close=ind_close.groupby(['industry','month'])['close'].last().reset_index()
monthly_close['fwd_ret']=monthly_close.groupby('industry')['close'].pct_change(-1)  # 下月收益
monthly_close['month']=monthly_close['month'].dt.to_timestamp()

# Merge
final=monthly_ind.merge(monthly_close[['industry','month','fwd_ret']],on=['industry','month'],how='inner')
final=final.dropna(subset=['fwd_ret'])
print("[5] 合并目标: %d rows" % len(final))

# Save
final.to_parquet('D:/AgentQuant/our/cache/industry_flow_monthly.parquet')
print("Saved: cache/industry_flow_monthly.parquet")

# Quick IC test
feat_cols=['buy_pressure','vol_ratio','money_flow','big_order','big_net','up_pct','ret_mean','flow_divergence']
print("\n[6] 快速IC检验:")
for f in feat_cols:
    if f in final.columns:
        ics=[]
        for m,g in final.groupby('month'):
            if len(g)>5:
                ic=g[f].rank().corr(g['fwd_ret'].rank())
                if not np.isnan(ic): ics.append(ic)
        if ics:
            mi=np.mean(ics); ir=mi/np.std(ics)*np.sqrt(12) if np.std(ics)>0 else 0
            print("  %-20s IC=%+.4f IR=%+.2f (%d月)"%(f,mi,ir,len(ics)))

con.close()
print("\n耗时: %.0fs"%(time.time()-t0))
