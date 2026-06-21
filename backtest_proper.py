# -*- coding: utf-8 -*-
"""
Proper Walk-Forward: 月度调仓, 真实持有收益, 无重叠
====================================================
每月第一个交易日选股 → 持有到下月第一个交易日 → 计算持仓期收益
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import duckdb, pandas as pd, numpy as np, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

t0=time.time()
DB='D:/FreeFinanceData/data/duckdb/finance.db'
CACHE='D:/AgentQuant/our/cache'
TOP_N=15; COST=0.0033

print('Proper Walk-Forward Backtest')
print('='*60)

# ============================================================
# 1. 加载因子数据
# ============================================================
print('[1/4] 加载数据...')
fn=pd.read_parquet(f'{CACHE}/factors_new6_v2.parquet')
fn['trade_date']=pd.to_datetime(fn['trade_date'])
FEATS_NEW=['vp_corr','sr5','amihud','turnover_rev','max_rev','gap']

fo=pd.read_parquet(f'{CACHE}/factors_2002.parquet')
fo['trade_date']=pd.to_datetime(fo['trade_date'])
fo=fo[fo['trade_date']>='2010-01-01']
FEATS_OLD=['rsi6','rsi14','boll_pos','boll_width','div_ma20','div_ma60','div_ma120',
           'vol_ratio','ma_score','rsi_extreme','margin_panic','streak5_dn']

# 加载辅助: 每月第一个交易日的open/close
con=duckdb.connect(DB, read_only=True)
kline=con.execute("""
    SELECT ts_code, trade_date, open, high, low, close, vol,
           close*total_share/10000 AS mcap,
           close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1 AS ret_1d,
           GREATEST(vol*close, 1.0) AS amount_proxy
    FROM kline_daily WHERE trade_date>='2010-01-01'
""").df()
kline['trade_date']=pd.to_datetime(kline['trade_date'])

# 目标(训练用)
target=con.execute("""
    SELECT s.ts_code, s.trade_date,
           (LEAD(s.close,20) OVER(PARTITION BY s.ts_code ORDER BY s.trade_date)/s.close-1)
           -(LEAD(x.close,20) OVER(ORDER BY x.trade_date)/x.close-1) AS excess_ret_20d
    FROM kline_daily s
    JOIN kline_daily x ON s.trade_date=x.trade_date AND x.ts_code='sh000300'
    WHERE s.trade_date>='2010-01-01'
""").df()
target['trade_date']=pd.to_datetime(target['trade_date'])

# 行业
industry=con.execute("""
    SELECT ts_code, ind_name FROM (
        SELECT ts_code, ind_name, ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY corr DESC) AS rn
        FROM stock_industry_map) WHERE rn=1
""").df()
con.close()

# 找每月第一个交易日
dates=sorted(kline['trade_date'].unique())
monthly_dates=[]
for ym, g in pd.Series(dates).groupby([d.strftime('%Y-%m') for d in dates]):
    monthly_dates.append(g.iloc[0])
monthly_dates=sorted(monthly_dates)
# 构建调仓日→下个调仓日的价格映射
rd_map={}
con2=duckdb.connect(DB, read_only=True)
for i in range(len(monthly_dates)-1):
    cur=monthly_dates[i]; nxt=monthly_dates[i+1]
    # 取前后两日数据以计算ret_1d
    prev_date=str(cur-pd.Timedelta(days=7))[:10]  # 取前7天确保覆盖
    px2=con2.execute(f"""
        SELECT ts_code, trade_date, close, open, close*total_share/10000 AS mcap
        FROM kline_daily WHERE trade_date IN ('{cur}','{nxt}')
    """).df()
    cur_px=px2[px2['trade_date']==str(cur)[:10]].copy()
    nxt_px=px2[px2['trade_date']==str(nxt)[:10]].copy()
    # Also get prev close for ret_1d
    prev_px=con2.execute(f"""
        SELECT ts_code, close AS prev_close
        FROM kline_daily WHERE trade_date<='{cur}' AND trade_date>='{prev_date}'
        QUALIFY ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC)=1
    """).df()
    cur_px=cur_px.merge(prev_px,on='ts_code',how='left')
    cur_px=cur_px.set_index('ts_code'); nxt_px=nxt_px.set_index('ts_code')
    merged=cur_px.join(nxt_px.rename(columns={'open':'next_open'}),how='inner')
    merged['ret_1d']=merged['close']/merged['prev_close']-1
    merged['fwd_ret']=merged['next_open']/merged['close']-1
    rd_map[cur]=merged
con2.close()
print(f'  调仓月: {len(rd_map)}')

# ============================================================
# 2. 合并数据
# ============================================================
print('[2/4] 合并因子+目标+行业+市值...')
fn=fn.merge(target,on=['ts_code','trade_date'],how='inner')
fn=fn.merge(industry,on='ts_code',how='left'); fn['ind_name']=fn['ind_name'].fillna('Other')
# 加市值
mcap_data=kline[['ts_code','trade_date','mcap']].drop_duplicates()
fn=fn.merge(mcap_data,on=['ts_code','trade_date'],how='left')

fo=fo.merge(target,on=['ts_code','trade_date'],how='inner')
fo=fo.merge(industry,on='ts_code',how='left'); fo['ind_name']=fo['ind_name'].fillna('Other')
fo=fo.merge(mcap_data,on=['ts_code','trade_date'],how='left')

# Trim to essential columns to avoid OOM
keep_cols_new=['ts_code','trade_date']+FEATS_NEW+['excess_ret_20d','ind_name','mcap']
keep_cols_old=['ts_code','trade_date']+FEATS_OLD+['excess_ret_20d','ind_name','mcap']
fn=fn[[c for c in keep_cols_new if c in fn.columns]]
fo=fo[[c for c in keep_cols_old if c in fo.columns]]
print(f'  fn trimmed: {len(fn):,}行 x {len(fn.columns)}列')
print(f'  fo trimmed: {len(fo):,}行 x {len(fo.columns)}列')
del target, mcap_data, kline; gc.collect()

# ============================================================
# 3. Walk-Forward + 月度选股
# ============================================================
print('[3/4] Walk-Forward月度选股...')

def walk_forward_monthly(df, feat_list, label):
    monthly_returns=[]
    monthly_details=[]

    for test_yr in range(2016,2027):
        train_start=f'{test_yr-6}-01-01'
        train_end=f'{test_yr-1}-12-31'

        tr=df[(df['trade_date']>=train_start)&(df['trade_date']<=train_end)].copy()
        if len(tr)<50000:
            print(f'  {label} {test_yr}: 训练数据不足')
            continue

        # OLS中性化
        tr['excess_ret_20d']=tr.groupby('trade_date')['excess_ret_20d'].transform(lambda x:x-x.mean())
        tr['mcap']=tr['mcap'].fillna(tr.groupby('trade_date')['mcap'].transform('median')).fillna(1e6)
        tr['ln_mcap']=np.log(tr['mcap'].clip(lower=1e6)); tr['ln_mcap_sq']=tr['ln_mcap']**2

        all_inds=list(tr['ind_name'].unique()); ind_map={ind:i for i,ind in enumerate(all_inds)}
        tr_dum=np.zeros((len(tr),len(all_inds)))
        for i,ind in enumerate(tr['ind_name']):
            if ind in ind_map: tr_dum[i,ind_map[ind]]=1
        X_tr=np.column_stack([tr['ln_mcap'].values,tr['ln_mcap_sq'].values,tr_dum])
        y_tr_raw=np.nan_to_num(tr[feat_list].fillna(0).values.astype(float),0)

        if X_tr.shape[0]>50000:
            idx=np.random.choice(X_tr.shape[0],50000,replace=False)
            Xf,yf=X_tr[idx],y_tr_raw[idx]
        else: Xf,yf=X_tr,y_tr_raw

        ols=LinearRegression(fit_intercept=False); ols.fit(Xf,yf)
        res_tr=y_tr_raw-X_tr@ols.coef_.T
        for i,c in enumerate(feat_list):
            name=c+'_n'; tr[name]=res_tr[:,i]
            mu,std=tr[name].mean(),tr[name].std()
            if std>0: tr[name]=(tr[name]-mu)/std

        flist=[f+'_n' for f in feat_list if f+'_n' in tr.columns]
        X_tr_f=tr[flist].fillna(0).values.astype(float)
        y1=tr['excess_ret_20d'].fillna(0).values
        y2=tr.groupby('trade_date')['excess_ret_20d'].rank(pct=True).fillna(0.5).values

        m1=LGBMRegressor(n_estimators=120,num_leaves=31,max_depth=6,learning_rate=0.03,
                         subsample=0.8,reg_alpha=0.2,reg_lambda=0.2,
                         min_child_samples=50,verbose=-1,n_jobs=-1).fit(X_tr_f,y1)
        m2=LGBMRegressor(n_estimators=120,num_leaves=31,max_depth=6,learning_rate=0.03,
                         subsample=0.8,reg_alpha=0.2,reg_lambda=0.2,
                         min_child_samples=50,verbose=-1,n_jobs=-1).fit(X_tr_f,y2)

        # 每月选股
        n_months=0
        for rd in monthly_dates:
            if rd.year!=test_yr: continue
            if rd not in rd_map: continue

            te_day=df[df['trade_date']==rd].copy()
            price_data=rd_map[rd]
            if len(te_day)<100: continue

            # 简单截面z-score预测
            te_day['pred']=0
            for f in feat_list:
                if f not in te_day.columns: continue
                mu_f=tr[f].mean(); std_f=tr[f].std()
                if std_f>0:
                    te_day['pred']+=(te_day[f].fillna(mu_f)-mu_f)/std_f
            te_day['pred']=te_day['pred']/len(feat_list)

            # 过滤
            valid_codes=set(price_data.index)
            te_day=te_day[te_day['ts_code'].isin(valid_codes)]
            if len(te_day)<50: continue

            prices=price_data.loc[te_day['ts_code'].values]
            te_day['mcap']=prices['mcap'].values
            te_day['ret_1d']=prices['ret_1d'].values
            te_day['fwd_ret']=prices['fwd_ret'].values

            te_day['mcap_r']=te_day['mcap'].rank(pct=True)
            te_day=te_day[te_day['mcap_r']>=0.20]  # 排最小20%
            te_day=te_day[te_day['ret_1d']<0.095]  # 非涨停
            te_day=te_day[te_day['fwd_ret'].notna()]

            top=te_day.nlargest(TOP_N,'pred')
            if len(top)<5: continue

            month_ret=top['fwd_ret'].mean()-COST
            monthly_returns.append(month_ret)
            monthly_details.append({'year':test_yr,'month':str(rd)[:7],'ret':month_ret,'n':len(top)})
            n_months+=1

        if n_months>0:
            yr_rets=[m['ret'] for m in monthly_details if m['year']==test_yr]
            ann=np.mean(yr_rets)*12
            sh=ann/(np.std(yr_rets)*np.sqrt(12)) if np.std(yr_rets)>0 else 0
            print(f'  {label} {test_yr}: {n_months}月  月均={np.mean(yr_rets)*100:+.2f}%  年化={ann*100:+.1f}%  Sharpe={sh:+.2f}')

    return monthly_returns, monthly_details

mr_new, md_new=walk_forward_monthly(fn, FEATS_NEW, 'NEW6')
print()
mr_old, md_old=walk_forward_monthly(fo, FEATS_OLD, 'OLD12')

# ============================================================
# 4. 绩效对比
# ============================================================
print()
print('='*60)
print('绩效对比')
print('='*60)

def calc_stats(monthly_rets):
    if not monthly_rets: return (0,0,0,0,0)
    r=np.array(monthly_rets)
    ann=np.mean(r)*12
    vol=np.std(r)*np.sqrt(12)
    sh=ann/vol if vol>0 else 0
    cum=np.cumprod(1+r)
    mdd=np.min(cum/np.maximum.accumulate(cum)-1)
    win=(r>0).mean()
    return ann,vol,sh,mdd,win

ann_n,vol_n,sh_n,mdd_n,win_n=calc_stats(mr_new)
ann_o,vol_o,sh_o,mdd_o,win_o=calc_stats(mr_old)

print(f'\n  {"指标":<20s} {"原始12因子":>12s} {"新6因子":>12s} {"提升":>10s}')
print(f'  {"-"*56}')
print(f'  {"年化收益":<20s} {ann_o*100:>+11.1f}% {ann_n*100:>+11.1f}% {(ann_n-ann_o)*100:>+10.1f}%')
print(f'  {"年化波动":<20s} {vol_o*100:>11.1f}% {vol_n*100:>11.1f}% {(vol_n-vol_o)*100:>+10.1f}%')
print(f'  {"Sharpe":<20s} {sh_o:>11.2f} {sh_n:>11.2f} {sh_n-sh_o:>+10.2f}')
print(f'  {"MDD":<20s} {mdd_o*100:>10.1f}% {mdd_n*100:>10.1f}% {(mdd_n-mdd_o)*100:>+10.1f}%')
print(f'  {"月胜率":<20s} {win_o*100:>10.1f}% {win_n*100:>10.1f}% {(win_n-win_o)*100:>+10.1f}%')
print(f'  {"月数":<20s} {len(mr_old):>11d} {len(mr_new):>11d}')

# 分年
print(f'\n  {"年份":<6s} {"旧12年化":>10s} {"新6年化":>10s} {"旧Sharpe":>10s} {"新Sharpe":>10s}')
print(f'  {"-"*50}')
for yr in range(2016,2027):
    ro=[m['ret'] for m in md_old if m['year']==yr]
    rn=[m['ret'] for m in md_new if m['year']==yr]
    if len(ro)>=3 or len(rn)>=3:
        ao=np.mean(ro)*12 if ro else 0; an=np.mean(rn)*12 if rn else 0
        so=ao/(np.std(ro)*np.sqrt(12)) if ro and np.std(ro)>0 else 0
        sn=an/(np.std(rn)*np.sqrt(12)) if rn and np.std(rn)>0 else 0
        print(f'  {yr:<6d} {ao*100:>+9.1f}% {an*100:>+9.1f}% {so:>+9.2f} {sn:>+9.2f}')

# 累计收益曲线
print(f'\n  {"策略":<12s} {"终值(100万起)":>16s} {"总收益":>10s}')
print(f'  {"-"*40}')
if mr_old:
    final_o=1e6*np.prod([1+r for r in mr_old])
    print(f'  {"原始12因子":<12s} {final_o:>16,.0f} {(final_o/1e6-1)*100:>+9.1f}%')
if mr_new:
    final_n=1e6*np.prod([1+r for r in mr_new])
    print(f'  {"新6因子":<12s} {final_n:>16,.0f} {(final_n/1e6-1)*100:>+9.1f}%')

print(f'\n总耗时: {time.time()-t0:.0f}s')
