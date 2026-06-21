# -*- coding: utf-8 -*-
"""
AgentQuant · 哑铃策略回测模拟器
===============================
防守端: 低PE+高ROE (季度调仓, 沪深300池)
进攻端: 行业动量→高波/高换手正交化 (月度调仓, 全A股)
动态: VIX驱动权重调整 + 不对称再平衡
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from scipy import stats
from datetime import date,timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei','Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

DB='D:/FreeFinanceData/data/duckdb/finance.db'

def conn():
    for i in range(20):
        try:
            c=duckdb.connect(DB,read_only=True);c.execute('SELECT 1');return c
        except: import time; time.sleep(min(2**i,30))
    return duckdb.connect(DB,read_only=True)

# ═══════════════════════════════
# 数据加载
# ═══════════════════════════════

def load_hs300_stocks(c):
    """沪深300成分股(简化: 有PE的大市值股票)"""
    df=c.execute("""
        SELECT DISTINCT v.ts_code FROM valuation_daily v
        JOIN kline_daily k ON v.ts_code=k.ts_code AND v.trade_date=k.trade_date
        WHERE v.pe_ttm>0 AND v.pe_ttm<200 AND v.trade_date='2026-06-12'
        AND k.close>0
    """).df()
    return df['ts_code'].tolist()

def load_defensive_pool(c, universe, n=30):
    """防守池: 低PE+高ROE"""
    codes_str=','.join([f"'{x}'" for x in universe[:2000]])
    if not codes_str: return pd.DataFrame()
    df=c.execute(f"""
        SELECT v.ts_code,v.pe_ttm,f.roe,f.gross_margin
        FROM valuation_daily v
        JOIN financial_statements f ON v.ts_code=f.ts_code
        WHERE v.ts_code IN ({codes_str})
        AND v.pe_ttm>0 AND v.pe_ttm<500 AND v.trade_date='2026-06-12'
        AND f.roe>0 AND f.roe<100 AND f.report_type='annual'
        AND f.report_date=(SELECT MAX(report_date) FROM financial_statements WHERE ts_code=f.ts_code AND report_type='annual')
    """).df()
    if len(df)==0: return df
    if len(df)<n: return df
    df['pe_rank']=(-df['pe_ttm']).rank()
    df['roe_rank']=df['roe'].rank()
    df['score']=df['pe_rank']*0.6+df['roe_rank']*0.4
    return df.nlargest(n,'score')

def load_offensive_pool(c, top_ind=3, n_per_ind=10):
    """进攻池: 行业动量→高波/高换手正交化"""
    # 行业动量top3
    ind_mom=c.execute("""
        SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1) mom FROM (
            SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
            FROM proxy_industry_daily WHERE trade_date>='2026-05-15'
        ) WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8 ORDER BY mom DESC LIMIT ?
    """,[top_ind]).df()
    top_industries=ind_mom['industry'].tolist()
    print(f'  强势行业: {top_industries}')

    # 全A股高波/高换手
    vol_turn=c.execute("""
        SELECT ts_code,STDDEV(dr) vol,AVG(turn) turn FROM (
            SELECT ts_code,turnover_rate turn,
                   (close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) dr
            FROM kline_daily WHERE trade_date>='2026-05-01' AND turnover_rate>0
        ) WHERE dr IS NOT NULL GROUP BY ts_code HAVING COUNT(*)>=15
    """).df()

    # 正交化: turn→vol回归, 取残差=净波动率
    from sklearn.linear_model import LinearRegression
    X=vol_turn[['turn']].values
    y=vol_turn['vol'].values
    lr=LinearRegression().fit(X,y)
    vol_turn['net_vol']=y-lr.predict(X)  # 残差
    r2=lr.score(X,y)
    print(f'  正交化: R2={r2:.3f}, 共线={np.sqrt(r2):.3f}')

    # 行业映射(简化: 用关键词)
    # 这里用申万行业指数成分股做映射
    # 实际需要stock_industry映射表, 简化处理: 用全A股
    vol_turn['score']=vol_turn['turn'].rank(pct=True)*0.4+vol_turn['net_vol'].rank(pct=True)*0.6
    return vol_turn.nlargest(n_per_ind*top_ind,'score')

# ═══════════════════════════════
# 回测引擎
# ═══════════════════════════════

def get_monthly_returns(c, stocks, start_date, end_date):
    """获取月度收益"""
    codes_str=','.join([f"'{x}'" for x in stocks[:200]])
    if not codes_str: return pd.DataFrame()
    df=c.execute(f"""
        WITH monthly AS (
            SELECT ts_code,STRFTIME(trade_date,'%Y-%m') mon,close,
                   ROW_NUMBER() OVER(PARTITION BY ts_code,STRFTIME(trade_date,'%Y-%m') ORDER BY trade_date) rn_asc,
                   ROW_NUMBER() OVER(PARTITION BY ts_code,STRFTIME(trade_date,'%Y-%m') ORDER BY trade_date DESC) rn_desc
            FROM kline_daily WHERE ts_code IN ({codes_str})
            AND trade_date>='{start_date}' AND trade_date<='{end_date}'
        )
        SELECT ts_code,mon,
               (MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1) ret
        FROM monthly GROUP BY ts_code,mon HAVING COUNT(*)>=10
    """).df()
    if not df.empty:
        df=df.rename(columns={'mon':'month'})
    return df

def barbell_backtest(c, def_stocks, off_stocks, months=12, def_weight=0.5,
                     def_rebalance=3, off_rebalance=1, friction=0.003):
    """
    哑铃回测
    def_rebalance: 防守端调仓周期(月)
    off_rebalance: 进攻端调仓周期(月)
    friction: 单边交易成本
    """
    # 获取历史月度收益
    start=(date.today()-timedelta(days=months*31)).strftime('%Y-%m-%d')
    end=date.today().strftime('%Y-%m-%d')

    def_ret=get_monthly_returns(c,def_stocks['ts_code'].tolist(),start,end)
    off_ret=get_monthly_returns(c,off_stocks['ts_code'].tolist(),start,end)

    if def_ret.empty or off_ret.empty:
        return None

    # 等权组合月度收益
    def_monthly=def_ret.groupby('month')['ret'].mean()
    off_monthly=off_ret.groupby('month')['ret'].mean()

    # 对齐月份
    months_idx=sorted(set(def_monthly.index)|set(off_monthly.index))

    # VIX动态权重
    vix=c.execute(f"SELECT trade_date,vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date>='{start}' ORDER BY trade_date").df()
    vix_monthly=vix.set_index('trade_date')['vix'].resample('ME').last() if len(vix)>0 else None

    nav_def=1.0; nav_off=1.0
    curve_def=[1.0]; curve_off=[1.0]; curve_comb=[2.0]
    w_def=def_weight

    for m in months_idx:
        r_def=def_monthly.get(m,0)
        r_off=off_monthly.get(m,0)

        # VIX调整
        if vix_monthly is not None:
            v_row=vix_monthly[vix_monthly.index==m]
            if len(v_row)>0:
                v=v_row.iloc[0]
                if pd.notna(v):
                    if v<18: w_def=def_weight-0.15
                    elif v>25: w_def=def_weight+0.25
                    else: w_def=def_weight
                    w_def=np.clip(w_def,0.2,0.8)

        # 调仓摩擦
        if int(m[-2:])%def_rebalance==0: r_def-=friction
        if int(m[-2:])%off_rebalance==0: r_off-=friction

        nav_def*=(1+r_def)
        nav_off*=(1+r_off)

        # 再平衡
        if abs(nav_off/(nav_def+nav_off)-w_def)>0.15:
            total=nav_def+nav_off
            nav_def=total*(1-w_def)
            nav_off=total*w_def

        nav_combined=nav_def+nav_off
        curve_def.append(nav_def)
        curve_off.append(nav_off)
        curve_comb.append(nav_combined)

    return {
        'def_ret':(nav_def-1)*100,'off_ret':(nav_off-1)*100,
        'comb_ret':(nav_combined/2-1)*100,
        'curve_def':curve_def,'curve_off':curve_off,'curve_comb':curve_comb,
        'months':months_idx
    }

# ═══════════════════════════════
# 可视化
# ═══════════════════════════════

def plot_barbell(result, def_label='防守(低PE+ROE)', off_label='进攻(高波+高换手)'):
    fig,axes=plt.subplots(2,1,figsize=(14,10))

    # 资金曲线
    ax=axes[0]
    months=result['months']
    x=range(len(months)+1)
    ax.plot(x,result['curve_def'],'b-',linewidth=2,label=def_label)
    ax.plot(x,result['curve_off'],'r-',linewidth=2,label=off_label)
    ax.plot(x,result['curve_comb'],'k-',linewidth=3,label='哑铃组合')
    ax.axhline(y=1.0,color='gray',linestyle='--',alpha=0.5)
    ax.legend(loc='upper left')
    ax.set_title('A股哑铃策略回测',fontsize=16,fontweight='bold')
    ax.set_ylabel('净值')
    ax.grid(True,alpha=0.3)

    # 标注收益
    dr=result['def_ret'];ore=result['off_ret'];cr=result['comb_ret']
    ax.text(0.02,0.95,f'防守: {dr:+.1f}% | 进攻: {ore:+.1f}% | 组合: {cr:+.1f}%',
            transform=ax.transAxes,fontsize=11,bbox=dict(boxstyle='round',facecolor='wheat'))

    # 回撤
    ax2=axes[1]
    dd_def=np.array(result['curve_def'])/np.maximum.accumulate(result['curve_def'])-1
    dd_off=np.array(result['curve_off'])/np.maximum.accumulate(result['curve_off'])-1
    dd_comb=np.array(result['curve_comb'])/np.maximum.accumulate(result['curve_comb'])-1
    ax2.fill_between(x,0,dd_comb*100,alpha=0.3,color='gray',label='哑铃回撤')
    ax2.plot(x,dd_def*100,'b--',alpha=0.7,label='防守回撤')
    ax2.plot(x,dd_off*100,'r--',alpha=0.7,label='进攻回撤')
    mdd_comb=dd_comb.min()*100
    mdd_def=dd_def.min()*100
    mdd_off=dd_off.min()*100
    ax2.set_title(f'回撤曲线 (最大: 防守{mdd_def:.1f}% 进攻{mdd_off:.1f}% 哑铃{mdd_comb:.1f}%)')
    ax2.set_ylabel('回撤 %')
    ax2.set_xlabel('月份')
    ax2.legend()
    ax2.grid(True,alpha=0.3)

    plt.tight_layout()
    out_path=r'D:\AgentQuant\our\barbell_backtest.png'
    plt.savefig(out_path,dpi=120,bbox_inches='tight')
    print(f'\n图表已保存: {out_path}')
    return out_path

# ═══════════════════════════════
# 主入口
# ═══════════════════════════════

def main(def_w=0.5,friction=0.003):
    c=conn()
    print('='*55)
    print('  A股哑铃策略回测模拟器')
    print(f'  防守:{def_w*100:.0f}% 进攻:{(1-def_w)*100:.0f}% 摩擦:{friction*100:.1f}%')
    print('='*55)

    # 防守池
    print('\n[防守池] 低PE+高ROE...')
    hs300=load_hs300_stocks(c)
    def_pool=load_defensive_pool(c,hs300,30)
    pe_m=def_pool['pe_ttm'].mean(); roe_m=def_pool['roe'].mean()
    print(f'  选入{len(def_pool)}只 PE均值{pe_m:.0f} ROE均值{roe_m:.1f}%')
    top5=def_pool['ts_code'].head().tolist()
    print(f'  前5: {top5}')

    # 进攻池
    print('\n[进攻池] 行业动量→高波/高换手正交化...')
    off_pool=load_offensive_pool(c,3,10)
    print(f'  选入{len(off_pool)}只')

    # 回测
    print('\n[回测]')
    result=barbell_backtest(c,def_pool,off_pool,months=12,
                           def_weight=def_w,off_rebalance=1,friction=friction)
    if result:
        dr=result['def_ret']; ore=result['off_ret']; cr=result['comb_ret']
        print(f'  防守: {dr:+.1f}%')
        print(f'  进攻: {ore:+.1f}%')
        print(f'  哑铃: {cr:+.1f}%')
        plot_barbell(result)
    else:
        print('  数据不足')

    c.close()
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--def-weight',type=float,default=0.5)
    p.add_argument('--friction',type=float,default=0.003)
    args=p.parse_args()
    main(args.def_weight,args.friction)
