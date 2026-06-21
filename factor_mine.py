# -*- coding: utf-8 -*-
"""深挖全部因子"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb, pandas as pd, numpy as np
from scipy import stats
c=duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db',read_only=True)

results=[]

def sic(fv,fr,label):
    m=fv.notna()&fr.notna()
    if m.sum()<30: return None,None
    ic,p=stats.spearmanr(fv[m].astype(float),fr[m].astype(float).clip(-0.5,0.5))
    results.append((label,round(ic,4),round(p,4) if p else None,m.sum()))
    print(f'  IC={ic:.4f} N={m.sum()}')
    return ic,p

def sic_simple(fv,fr,label):
    m=fv.notna()&fr.notna()
    if m.sum()<5: return
    ic,_=stats.spearmanr(fv[m].astype(float),fr[m].astype(float))
    results.append((label,round(ic,4),None,m.sum()))
    print(f'  IC={ic:.4f} N={m.sum()}')

# 通用前向收益
fwd=c.execute("""
    SELECT ts_code,(MAX(close)/MIN(close)-1) r FROM kline_daily
    WHERE trade_date>='2026-05-25' GROUP BY ts_code HAVING COUNT(*)>=10
""").df()

fwd10=c.execute("""
    SELECT ts_code,(MAX(close)/MIN(close)-1) r FROM kline_daily
    WHERE trade_date>='2026-06-05' GROUP BY ts_code HAVING COUNT(*)>=5
""").df()

fwd_idx=c.execute("SELECT (MAX(close)/MIN(close)-1) r FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2026-06-01'").fetchone()[0]
print(f'沪深300前瞻收益: {fwd_idx*100:+.2f}%\n')

# ===== 1. PE分位数 =====
print('[1] PE分位数...',end='')
df=c.execute("""
    SELECT pe.ts_code, pe.pct FROM (
        SELECT ts_code,pe_ttm,NTILE(100) OVER(ORDER BY pe_ttm) pct
        FROM valuation_daily WHERE pe_ttm>0 AND pe_ttm<500 AND trade_date='2026-06-12'
    ) pe
""").df()
m=df.merge(fwd,on='ts_code')
sic(m['pct'],m['r'],'PE分位数(低=便宜)')

# ===== 2. ROE分位 =====
print('[2] ROE...',end='')
df=c.execute("""
    SELECT f.ts_code,f.roe FROM financial_statements f
    WHERE f.roe>0 AND f.roe<100 AND f.report_type='annual'
    AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
""").df()
m=df.merge(fwd,on='ts_code')
sic(m['roe'],m['r'],'ROE')

# ===== 3. 毛利率 =====
print('[3] 毛利率...',end='')
df=c.execute("""
    SELECT f.ts_code,f.gross_margin FROM financial_statements f
    WHERE f.gross_margin>0 AND f.gross_margin<100 AND f.report_type='annual'
    AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
""").df()
m=df.merge(fwd,on='ts_code')
sic(m['gross_margin'],m['r'],'毛利率')

# ===== 4. ROE+PE+毛利率组合 =====
print('[4] ROE+PE+毛利率混合...',end='')
df=c.execute("""
    SELECT v.ts_code,v.pe_ttm,f.roe,f.gross_margin
    FROM valuation_daily v
    JOIN financial_statements f ON v.ts_code=f.ts_code
    WHERE v.pe_ttm>0 AND v.pe_ttm<500 AND v.trade_date='2026-06-12'
    AND f.roe>0 AND f.roe<100 AND f.report_type='annual'
    AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
""").df()
m=df.merge(fwd,on='ts_code')
if len(m)>30:
    m['score']=(-m['pe_ttm']).rank(pct=True)*0.4+m['roe'].rank(pct=True)*0.3+m['gross_margin'].rank(pct=True)*0.3
    sic(m['score'],m['r'],'基本面混合')

# ===== 5. 动量 =====
print('[5] 20日动量...',end='')
df=c.execute("""
    SELECT ts_code,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=20 THEN close END),0)-1) mom FROM (
        SELECT ts_code,close,ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC) rn
        FROM kline_daily WHERE trade_date>='2026-05-01'
    ) t WHERE rn<=20 GROUP BY ts_code HAVING COUNT(*)>=15
""").df()
m=df.merge(fwd10,on='ts_code')
sic(m['mom'],m['r'],'20日动量')

# ===== 6. 波动率 =====
print('[6] 波动率(低波)...',end='')
df=c.execute("""
    SELECT ts_code,STDDEV(dr) vol FROM (
        SELECT ts_code,(close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) dr
        FROM kline_daily WHERE trade_date>='2026-05-01'
    ) WHERE dr IS NOT NULL GROUP BY ts_code HAVING COUNT(*)>=15
""").df()
m=df.merge(fwd10,on='ts_code')
sic(-m['vol'],m['r'],'低波动率')

# ===== 7. 换手率 =====
print('[7] 换手率(低换手)...',end='')
df=c.execute("""
    SELECT ts_code,AVG(turnover_rate) turn FROM kline_daily
    WHERE trade_date>='2026-05-01' AND turnover_rate>0 GROUP BY ts_code HAVING COUNT(*)>=10
""").df()
m=df.merge(fwd10,on='ts_code')
sic(-m['turn'],m['r'],'低换手率')

# ===== 8. 行业动量 =====
print('[8] 行业动量...',end='')
df=c.execute("""
    SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1) mom FROM (
        SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
        FROM proxy_industry_daily WHERE trade_date>='2026-05-15'
    ) t WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8
""").df()
ifwd=c.execute("""
    SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=5 THEN close END),0)-1) r FROM (
        SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
        FROM proxy_industry_daily WHERE trade_date>='2026-06-05'
    ) t WHERE rn<=5 GROUP BY industry HAVING COUNT(*)>=4
""").df()
m=df.merge(ifwd,on='industry')
sic(m['mom'],m['r'],'行业动量')

# ===== 9. 地量 =====
print('[9] 地量...',end='')
df=c.execute("""
    SELECT ts_code,vol/NULLIF(avg20,0) vol_ratio FROM (
        SELECT ts_code,vol,AVG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND 1 PRECEDING) avg20
        FROM kline_daily WHERE trade_date>='2026-05-01' AND vol>0
    ) QUALIFY ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC)=1
""").df()
m=df.merge(fwd10,on='ts_code')
sic(-m['vol_ratio'],m['r'],'地量(缩量)')

# ===== 10. VIX择时 =====
print('[10] VIX择时...',end='')
vix=c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL ORDER BY trade_date DESC LIMIT 1").fetchone()[0]
vix20=c.execute("SELECT AVG(vix) FROM macro_indicators WHERE vix IS NOT NULL AND trade_date>='2026-05-01'").fetchone()[0]
score=(20-vix20)/20
# VIX<20→利好, VIX>25→利空
vix_signal = '多' if score>0 else '空'
results.append(('VIX择时(均值回归)',round(score,4),None,20))
print(f'  VIX={vix:.1f} 20日均={vix20:.1f} 信号={vix_signal}')

# ===== 11. 科创/沪深相对强弱 =====
print('[11] 科创/沪深RS...',end='')
kc=c.execute("SELECT close FROM kline_daily WHERE ts_code='sh000688' ORDER BY trade_date DESC LIMIT 1").fetchone()[0]
hs=c.execute("SELECT close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date DESC LIMIT 1").fetchone()[0]
kc20=c.execute("SELECT AVG(close) FROM (SELECT close FROM kline_daily WHERE ts_code='sh000688' ORDER BY trade_date DESC LIMIT 20)").fetchone()[0]
hs20=c.execute("SELECT AVG(close) FROM (SELECT close FROM kline_daily WHERE ts_code='sh000300' ORDER BY trade_date DESC LIMIT 20)").fetchone()[0]
rs_now=kc/hs; rs_20=kc20/hs20
rs_chg=rs_now/rs_20-1
# RS上升→成长风格占优
results.append(('科创/沪深RS变化',round(rs_chg,4),None,2))
print(f'  RS={rs_now:.3f} 20日={rs_20:.3f} Δ={rs_chg*100:+.1f}%')

# ===== 12. 融资逆向 =====
print('[12] 融资逆向...',end='')
df=c.execute("""
    SELECT (margin_balance/LAG(margin_balance) OVER(ORDER BY trade_date)-1)*100 chg FROM margin_trading
    WHERE trade_date>='2026-04-01' QUALIFY chg IS NOT NULL ORDER BY trade_date DESC LIMIT 10
""").df()
avg_margin_chg=df['chg'].mean()
signal='逆向买入(恐慌)' if avg_margin_chg<-1 else ('减仓信号(过热)' if avg_margin_chg>1 else '正常')
results.append(('融资日变',round(avg_margin_chg,2),None,10))
print(f'  日均{avg_margin_chg:+.2f}% → {signal}')

c.close()

# ===== 汇总 =====
print('\n'+'='*55)
print('  全因子IC汇总')
print('='*55)
print(f'  {"因子":20s} {"IC":>8s} {"N":>6s} {"判断":>12s}')
print('  '+'-'*50)
ok=0
for name,ic,p,n in results:
    if ic is None: continue
    j='✅有效' if abs(ic)>0.03 else ('⚠️弱' if abs(ic)>0.01 else '❌')
    if j=='✅有效': ok+=1
    print(f'  {name:20s} {ic:>+8.4f} {n:>6} {j:>12s}')
print('  '+'-'*50)
print(f'  有效因子: {ok}个 (|IC|>0.03)')
print('='*55)
