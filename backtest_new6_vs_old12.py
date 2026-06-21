# -*- coding: utf-8 -*-
"""
新6因子 Walk-Forward 全链回测
==============================
对比: 原始12因子 vs 新6因子 vs 等权合成
管线: OLS中性化 → LightGBM双目标 → 动量拐点 → 月度调仓
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings, os, json
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

t0=time.time()
DB='D:/FreeFinanceData/data/duckdb/finance.db'
CACHE='D:/AgentQuant/our/cache'

# ============================================================
# Step 1: 计算6因子 + 目标
# ============================================================
print('='*60)
print('新6因子 vs 原始12因子 Walk-Forward回测')
print('='*60)

FNEW_FILE=f'{CACHE}/factors_new6_v2.parquet'
if os.path.exists(FNEW_FILE):
    print('[1/4] 加载已有因子...')
    factors=pd.read_parquet(FNEW_FILE)
else:
    print('[1/4] 计算新6因子 (2002-2026)...')
    con=duckdb.connect(DB, read_only=True)

    # 6因子SQL
    big=con.execute("""
    WITH daily AS (
        SELECT ts_code, trade_date, open, high, low, close, vol, amount, turnover_rate,
               close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
               close/LAG(close,5) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_5d,
               LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date),1)) AS log_vol_diff,
               ABS(close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1)/NULLIF(amount,0)*1e10 AS illiq_daily,
               open/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS gap
        FROM kline_daily WHERE trade_date>='2002-01-01'
    ),
    ranked AS (
        SELECT *, PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY ret) AS r_ret,
                  PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY log_vol_diff) AS r_vol
        FROM daily WHERE ret IS NOT NULL
    ),
    roll AS (
        SELECT ts_code, trade_date, ret_5d, gap,
               AVG(illiq_daily) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS amihud,
               AVG(turnover_rate) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS turnover_5d,
               MAX(ret) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS max_ret_5d
        FROM ranked
    )
    SELECT ts_code, trade_date,
           -ret_5d AS sr5, gap, amihud,
           -turnover_5d AS turnover_rev,
           -max_ret_5d AS max_rev
    FROM roll WHERE ret_5d IS NOT NULL AND amihud IS NOT NULL AND turnover_5d IS NOT NULL
    """).df()
    con.close()
    big['trade_date']=pd.to_datetime(big['trade_date'])

    # VP_Corr单独算
    print('  VP_Corr...')
    con2=duckdb.connect(DB, read_only=True)
    raw=con2.execute("""
    WITH daily AS (
        SELECT ts_code, trade_date,
               close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret,
               LN(GREATEST(vol,1))-LN(GREATEST(LAG(vol) OVER(PARTITION BY ts_code ORDER BY trade_date),1)) AS log_vol_diff
        FROM kline_daily WHERE trade_date>='2002-01-01'
    ),
    ranked AS (
        SELECT ts_code, trade_date, ret, log_vol_diff,
               PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY ret) AS rank_ret,
               PERCENT_RANK() OVER(PARTITION BY trade_date ORDER BY log_vol_diff) AS rank_vol
        FROM daily WHERE ret IS NOT NULL AND log_vol_diff IS NOT NULL
    )
    SELECT ts_code, trade_date, rank_ret, rank_vol FROM ranked ORDER BY ts_code, trade_date
    """).df()
    con2.close()

    codes=raw['ts_code'].unique()
    results=[]
    for i in range(0,len(codes),800):
        batch=codes[i:i+800]; bdf=raw[raw['ts_code'].isin(batch)]
        for ts, g in bdf.groupby('ts_code'):
            g=g.set_index('trade_date').sort_index()
            if len(g)<10: continue
            corr=g['rank_ret'].rolling(6,min_periods=5).corr(g['rank_vol'])
            results.append(pd.DataFrame({'ts_code':ts,'trade_date':g.index,'vp_corr_raw':corr.values}))
        if (i+800)%4000==0: print(f'    {i+800}/{len(codes)}...')
    vp=pd.concat(results,ignore_index=True); vp['vp_corr']=-vp['vp_corr_raw']
    vp['trade_date']=pd.to_datetime(vp['trade_date']); del raw,results; gc.collect()

    factors=big.merge(vp[['ts_code','trade_date','vp_corr']],on=['ts_code','trade_date'],how='inner')
    factors=factors.dropna()
    factors['trade_date']=factors['trade_date'].dt.strftime('%Y-%m-%d')
    os.makedirs(CACHE, exist_ok=True)
    factors.to_parquet(FNEW_FILE)
    print(f'  已保存: {FNEW_FILE}')

print(f'  因子: {len(factors):,}行, {factors["ts_code"].nunique():,}只')

# ============================================================
# Step 2: 加载目标+市值+行业
# ============================================================
print('[2/4] 加载目标/市值/行业...')
con=duckdb.connect(DB, read_only=True)
target=con.execute("""
    SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
           (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_20d
    FROM (SELECT ts_code, trade_date, close,
          LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
          FROM kline_daily WHERE trade_date>='2002-01-01') s
    JOIN (SELECT trade_date, close,
          LEAD(close,20) OVER(ORDER BY trade_date) AS fc
          FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='2002-01-01') x
    ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
""").df()

mcap=con.execute("""
    SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
           close*total_share/10000 AS mcap
    FROM kline_daily WHERE trade_date>='2002-01-01'
""").df()

industry=con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn=1
""").df()
con.close()

# 合并
df=factors.merge(target, on=['ts_code','trade_date'], how='inner')
df=df.merge(industry, on='ts_code', how='left'); df['ind_name']=df['ind_name'].fillna('Other')
df=df.merge(mcap, on=['ts_code','trade_date'], how='left')
del factors,target,mcap; gc.collect()

# ============================================================
# Step 3: Walk-Forward 回测
# ============================================================
print('[3/4] Walk-Forward回测...')
print()

# 新6因子
FEATS_NEW=['vp_corr','sr5','amihud','turnover_rev','max_rev','gap']

# 原始12因子(用于对比)
FEATS_OLD=['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
           'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

# Walk-Forward: 2010-2015 train → 2016 test, 2011-2016 train → 2017 test, ...
# 滚动窗口: 6年train / 1年test
TEST_YEARS=list(range(2016,2027))  # 2016-2026 (2026 partial)

def process_fold(df, train_yrs, test_yr, feat_list):
    """单折: 截面中性化 + LightGBM训练 + 预测"""
    tr=df[df['trade_date'].str[:4].isin([str(y) for y in train_yrs])].copy()
    te=df[df['trade_date'].str[:4]==str(test_yr)].copy()

    if len(tr)<50000 or len(te)<5000:
        return None

    # 截面中性化
    for d in [tr,te]:
        d['excess_ret_20d']=d.groupby('trade_date')['excess_ret_20d'].transform(lambda x: x-x.mean())

    # 市值填充
    for d in [tr,te]:
        d['mcap']=d['mcap'].fillna(d.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        d['ln_mcap']=np.log(d['mcap'].clip(lower=1e6))
        d['ln_mcap_sq']=d['ln_mcap']**2

    # OLS中性化
    all_inds=sorted(set(tr['ind_name'].unique())|set(te['ind_name'].unique()))
    ind_map={ind:i for i,ind in enumerate(all_inds)}

    tr_dum=np.zeros((len(tr),len(all_inds)))
    te_dum=np.zeros((len(te),len(all_inds)))
    for i,ind in enumerate(tr['ind_name']):
        if ind in ind_map: tr_dum[i,ind_map[ind]]=1
    for i,ind in enumerate(te['ind_name']):
        if ind in ind_map: te_dum[i,ind_map[ind]]=1

    X_tr=np.column_stack([tr['ln_mcap'].values,tr['ln_mcap_sq'].values,tr_dum])
    X_te=np.column_stack([te['ln_mcap'].values,te['ln_mcap_sq'].values,te_dum])
    y_tr_raw=np.nan_to_num(tr[feat_list].fillna(0).values.astype(float),0)
    y_te_raw=np.nan_to_num(te[feat_list].fillna(0).values.astype(float),0)

    if X_tr.shape[0]>50000:
        idx=np.random.choice(X_tr.shape[0],50000,replace=False)
        Xf,yf=X_tr[idx],y_tr_raw[idx]
    else:
        Xf,yf=X_tr,y_tr_raw

    m=LinearRegression(fit_intercept=False); m.fit(Xf,yf)
    res_tr=y_tr_raw-X_tr@m.coef_.T; res_te=y_te_raw-X_te@m.coef_.T

    for i,c in enumerate(feat_list):
        name=c+'_n'; tr[name]=res_tr[:,i]; te[name]=res_te[:,i]
        mu,std=tr[name].mean(),tr[name].std()
        if std>0: tr[name]=(tr[name]-mu)/std; te[name]=(te[name]-mu)/std

    # LightGBM双目标
    flist=[f+'_n' for f in feat_list if f+'_n' in tr.columns]
    X_tr_f=tr[flist].fillna(0).values.astype(float)
    X_te_f=te[flist].fillna(0).values.astype(float)

    y1=tr['excess_ret_20d'].fillna(0).values
    y2=tr.groupby('trade_date')['excess_ret_20d'].rank(pct=True).fillna(0.5).values

    m1=LGBMRegressor(n_estimators=120,num_leaves=31,max_depth=6,learning_rate=0.03,
                     subsample=0.8,reg_alpha=0.2,reg_lambda=0.2,
                     min_child_samples=50,verbose=-1,n_jobs=-1).fit(X_tr_f,y1)
    m2=LGBMRegressor(n_estimators=120,num_leaves=31,max_depth=6,learning_rate=0.03,
                     subsample=0.8,reg_alpha=0.2,reg_lambda=0.2,
                     min_child_samples=50,verbose=-1,n_jobs=-1).fit(X_tr_f,y2)

    te['pred']=(m1.predict(X_te_f)-y1.mean())/(y1.std() or 1)+(m2.predict(X_te_f)-0.5)/0.3

    # 动量拐点
    te=te.sort_values(['ts_code','trade_date'])
    te['mom_20d']=te.groupby('ts_code')['excess_ret_20d'].transform(
        lambda x: x.rolling(20,min_periods=5).mean())
    te['mom_60d']=te.groupby('ts_code')['excess_ret_20d'].transform(
        lambda x: x.rolling(60,min_periods=5).mean())
    te.loc[(te['mom_20d'].fillna(0)-te['mom_60d'].fillna(0))<0,'pred']*=0.8

    return te[['ts_code','trade_date','pred','excess_ret_20d','mcap']].copy()

# 运行对比
def run_wf(df, feat_list, label):
    """Walk-Forward回测"""
    monthly_results=[]
    yearly_stats=[]

    for test_yr in TEST_YEARS:
        train_yrs=list(range(test_yr-6, test_yr))
        fold=process_fold(df, train_yrs, test_yr, feat_list)
        if fold is None:
            print(f'  {label} {test_yr}: 数据不足,跳过')
            continue

        # 按月选Top-15等权
        fold['ym']=fold['trade_date'].str[:7]
        for ym,g in fold.groupby('ym'):
            if len(g)<30: continue
            g['mcap_r']=g['mcap'].rank(pct=True)
            g_f=g[g['mcap_r']>=0.20]  # 排除最小20%
            top=g_f.nlargest(15,'pred')
            if len(top)<5: continue
            monthly_results.append({
                'year':test_yr, 'month':ym,
                'ret':top['excess_ret_20d'].mean(),
                'n':len(top)
            })

        yr_df=pd.DataFrame([m for m in monthly_results if m['year']==test_yr])
        if len(yr_df)>0:
            ann=yr_df['ret'].mean()*12
            sh=ann/(yr_df['ret'].std()*np.sqrt(12)) if yr_df['ret'].std()>0 else 0
            yearly_stats.append({'year':test_yr,'ann_ret':ann,'sharpe':sh,'n':len(yr_df)})
            print(f'  {label} {test_yr}: 年化={ann*100:+.1f}% Sharpe={sh:+.2f} ({len(yr_df)}月)')

    return monthly_results, yearly_stats

# 测试新6因子
print('--- 新6因子 ---')
m_new, y_new=run_wf(df, FEATS_NEW, 'NEW6')

# 测试原始12因子 (分块加载避免OOM)
print()
print('--- 原始12因子(分块加载) ---')

# 先只加载需要用到的年份(2010+)
df_old=pd.read_parquet(f'{CACHE}/factors_2002.parquet')
df_old['trade_date']=pd.to_datetime(df_old['trade_date'])
df_old=df_old[df_old['trade_date']>='2010-01-01'].copy()
df_old['trade_date']=df_old['trade_date'].astype(str)
print(f'  旧因子2010+: {len(df_old):,}行')

# 分块合并目标数据
con3=duckdb.connect(DB, read_only=True)
# 分年加载目标
target_chunks=[]
for yr in range(2010,2027):
    chunk=con3.execute(f"""
        SELECT s.ts_code, CAST(s.trade_date AS VARCHAR) AS trade_date,
               (s.fc/s.close-1.0) - (x.fc/x.close-1.0) AS excess_ret_20d
        FROM (SELECT ts_code, trade_date, close,
              LEAD(close,20) OVER(PARTITION BY ts_code ORDER BY trade_date) AS fc
              FROM kline_daily WHERE trade_date BETWEEN '{yr}-01-01' AND '{yr}-12-31') s
        JOIN (SELECT trade_date, close,
              LEAD(close,20) OVER(ORDER BY trade_date) AS fc
              FROM kline_daily WHERE ts_code='sh000300'
                AND trade_date BETWEEN '{yr}-01-01' AND '{yr}-12-31') x
        ON s.trade_date=x.trade_date WHERE s.fc IS NOT NULL
    """).df()
    if len(chunk)>0: target_chunks.append(chunk)
    print(f'    目标{yr}: {len(chunk):,}行')

t2=pd.concat(target_chunks,ignore_index=True)
del target_chunks; gc.collect()

# 市值也分年
mc_chunks=[]
for yr in range(2010,2027):
    mc=con3.execute(f"""
        SELECT ts_code, CAST(trade_date AS VARCHAR) AS trade_date,
               close*total_share/10000 AS mcap
        FROM kline_daily WHERE trade_date BETWEEN '{yr}-01-01' AND '{yr}-12-31'
    """).df()
    if len(mc)>0: mc_chunks.append(mc)
mc2=pd.concat(mc_chunks,ignore_index=True)
del mc_chunks; gc.collect()
con3.close()

# 合并
df_old=df_old.merge(t2,on=['ts_code','trade_date'],how='inner')
df_old=df_old.merge(industry,on='ts_code',how='left'); df_old['ind_name']=df_old['ind_name'].fillna('Other')
df_old=df_old.merge(mc2,on=['ts_code','trade_date'],how='left')
del t2,mc2; gc.collect()
print(f'  合并后: {len(df_old):,}行')

m_old, y_old=run_wf(df_old, FEATS_OLD, 'OLD12')

# ============================================================
# Step 4: 对比汇总
# ============================================================
print()
print('='*60)
print('对比汇总')
print('='*60)

def summarize(yearly, label):
    if not yearly: return
    anns=[y['ann_ret'] for y in yearly]
    shs=[y['sharpe'] for y in yearly]
    avg_ann=np.mean(anns); avg_sh=np.mean(shs)
    # 计算累计
    all_rets=[]
    for y in yearly:
        all_rets.extend([y['ann_ret']/12]*y['n'])
    if all_rets:
        cum=np.cumprod(1+np.array(all_rets))
        mdd=np.min(cum/np.maximum.accumulate(cum)-1)
    else:
        mdd=0

    print(f'\n{label}:')
    print(f'  平均年化: {avg_ann*100:+.1f}%')
    print(f'  平均Sharpe: {avg_sh:+.2f}')
    print(f'  MDD: {mdd*100:.1f}%')
    print(f'  分年:')
    for y in sorted(yearly,key=lambda x:x['year']):
        yr=y['year']; ar=y['ann_ret']; sh=y['sharpe']; nm=y['n']
        print(f'    {yr}: {ar*100:+.1f}% Sharpe={sh:+.2f} ({nm}月)')
    return avg_ann, avg_sh, mdd

r_new=summarize(y_new, '新6因子')
r_old=summarize(y_old, '原始12因子')

if r_new and r_old:
    hdr=['指标','原始12','新6','提升']
    print('\n  {:<20s} {:>10s} {:>10s} {:>10s}'.format(*hdr))
    print('  ' + '-'*50)
    print('  {:<20s} {:>+9.1f}% {:>+9.1f}% {:>+9.1f}%'.format('年化收益', r_old[0]*100, r_new[0]*100, (r_new[0]-r_old[0])*100))
    print('  {:<20s} {:>10.2f} {:>10.2f} {:>+9.2f}'.format('Sharpe', r_old[1], r_new[1], r_new[1]-r_old[1]))
    print('  {:<20s} {:>9.1f}% {:>9.1f}% {:>+9.1f}%'.format('MDD', r_old[2]*100, r_new[2]*100, (r_new[2]-r_old[2])*100))

print(f'\n总耗时: {time.time()-t0:.0f}s')
