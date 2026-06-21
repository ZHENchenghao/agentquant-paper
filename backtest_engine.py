# -*- coding: utf-8 -*-
"""
AgentQuant · 哑铃滚动回测引擎
==============================
66个月滚动截面, PIT财务, 交易日对齐, 换手扣费, VIX风控
输出: 三张图(滚动r/条件相关性/风控对比)

用法: python backtest_engine.py
"""
import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB='D:/FreeFinanceData/data/duckdb/finance.db'
sys.path.insert(0,'D:/AgentQuant/our')
from factor_pipeline import (
    get_clean_universe, get_pit_report_type, get_pit_financials,
    calc_defense_score, calc_offense_score
)

def cdb(): return duckdb.connect(DB,read_only=True)

# ═══════════════════════════════════
# 补丁1: 交易日历
# ═══════════════════════════════════

def get_month_end(c, year, month):
    """真实月末交易日"""
    target=date(year,month,28)+timedelta(days=4)
    target=target.replace(day=1)-timedelta(days=1)
    r=c.execute("SELECT MAX(trade_date) FROM kline_daily WHERE trade_date<=?",
                [target.isoformat()]).fetchone()
    return r[0] if r and r[0] else None

# ═══════════════════════════════════
# 补丁2: 换手摩擦
# ═══════════════════════════════════

def calc_friction(old, new, nav, rate, sell_tax=0.0005):
    """按实际换手扣费"""
    old_s=set(old); new_s=set(new)
    sell_n=len(old_s-new_s); buy_n=len(new_s-old_s)
    if len(new)==0: return 0
    turn_ratio=(sell_n+buy_n)/len(new)
    cost=nav*turn_ratio*rate
    if sell_n>0: cost+=nav*(sell_n/len(new))*sell_tax
    return cost

# ═══════════════════════════════════
# VIX风控
# ═══════════════════════════════════

def market_trend_filter(c, trade_date):
    """大盘趋势开关: 沪深300<20MA → 熊市, 封印进攻"""
    r=c.execute("""
        SELECT AVG(close) ma20, MAX(close) close_now FROM kline_daily
        WHERE ts_code='sh000300' AND trade_date<=? AND trade_date>=?
    """,[trade_date.isoformat(),(trade_date-timedelta(days=30)).isoformat()]).fetchone()
    if not r or not r[0]: return False
    close_now, ma20 = r[1], r[0]

    # 5日前MA
    r2=c.execute("""
        SELECT AVG(close) FROM kline_daily WHERE ts_code='sh000300'
        AND trade_date<=? AND trade_date>=?
    """,[(trade_date-timedelta(days=5)).isoformat(),(trade_date-timedelta(days=35)).isoformat()]).fetchone()
    ma5d_ago = r2[0] if r2 and r2[0] else ma20

    # 双条件: 收盘<MA 且 MA向下
    below_ma = close_now < ma20
    ma_declining = ma20 < ma5d_ago
    return below_ma and ma_declining


def vix_weight(c, trade_date, base_w=0.5):
    """VIX动态权重: 滚动250日分位数阈值"""
    r=c.execute("SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
                [trade_date.isoformat()]).fetchone()
    vix=r[0] if r else 20

    # 250日滚动分位数
    pct=c.execute("""
        SELECT PERCENTILE_CONT(0.85) WITHIN GROUP (ORDER BY vix),
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vix)
        FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? AND trade_date>=?
    """,[trade_date.isoformat(),(trade_date-timedelta(days=365)).isoformat()]).fetchone()
    p85=pct[0] if pct and pct[0] else 28
    p95=pct[1] if pct and pct[1] else 32

    # 21天持续超过分位数
    vix_hi_days=c.execute("""
        SELECT COUNT(*) FROM macro_indicators
        WHERE vix>? AND trade_date<=? AND trade_date>=?
    """,[p95,trade_date.isoformat(),(trade_date-timedelta(days=21)).isoformat()]).fetchone()[0]

    if vix_hi_days>=21:
        return 0.10, 0.00, f'FULL_DEFENSE_21D(P95={p95:.1f})'

    if vix>p95: w_def=0.80; w_off=0.20; state=f'CRISIS(P95={p95:.1f})'
    elif vix>p85: w_def=0.65; w_off=0.35; state=f'DEFENSE(P85={p85:.1f})'
    elif vix<p85*0.6: w_def=0.35; w_off=0.65; state=f'ATTACK'
    else: w_def=0.50; w_off=0.50; state='NEUTRAL'
    return w_def, w_off, state

# ═══════════════════════════════════
# 持仓收益
# ═══════════════════════════════════

def holdings_return(c, holdings, start_date, end_date):
    """等权持仓区间收益 (首日收盘→末日收盘)"""
    if not holdings: return 0
    codes_str=','.join([f"'{x}'" for x in holdings[:200]])
    if not codes_str: return 0
    df=c.execute(f"""
        WITH ranked AS (
            SELECT ts_code,close,ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date) rn_asc,
                   ROW_NUMBER() OVER(PARTITION BY ts_code ORDER BY trade_date DESC) rn_desc
            FROM kline_daily
            WHERE ts_code IN ({codes_str}) AND trade_date>='{start_date.isoformat()}'
            AND trade_date<='{end_date.isoformat()}' AND close>0
        )
        SELECT ts_code,(MAX(CASE WHEN rn_desc=1 THEN close END)/NULLIF(MAX(CASE WHEN rn_asc=1 THEN close END),0)-1) ret
        FROM ranked WHERE rn_asc=1 OR rn_desc=1 GROUP BY ts_code HAVING COUNT(*)=2
    """).df()
    if df.empty: return 0
    return df['ret'].mean()

# ═══════════════════════════════════
# 主循环
# ═══════════════════════════════════

def run_backtest(start_year=2021, end_year=2026, base_w=0.5):
    c=cdb()
    results={'dates':[],'nav_def':[],'nav_off':[],'nav_comb':[],
             'r_def':[],'r_off':[],'vix_vals':[],'states':[],
             'def_holdings':[],'off_holdings':[]}

    nav_def=1.0; nav_off=1.0
    def_hold=[]; off_hold=[]

    # 找到起始月
    first_date=get_month_end(c,start_year,1)
    if not first_date: return results

    months=[]
    for y in range(start_year,end_year+1):
        for m in range(1,13):
            dt=get_month_end(c,y,m)
            if dt and dt>=first_date and dt<=date.today(): months.append(dt)

    print(f'回测: {len(months)}个月, {months[0]}~{months[-1]}')

    for i,rebal_date in enumerate(months):
        if i==len(months)-1: break
        next_date=months[i+1]

        # VIX权重
        w_def,w_off,state=vix_weight(c,rebal_date,base_w)

        # 大盘趋势开关: 熊市封印进攻端
        bear_market=market_trend_filter(c,rebal_date)
        if bear_market:
            w_off=0.0; w_def=1.0
            state='MKT_BEAR'
        results['vix_vals'].append(c.execute(
            "SELECT vix FROM macro_indicators WHERE vix IS NOT NULL AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
            [rebal_date.isoformat()]).fetchone()[0] or 20)
        results['states'].append(state)

        # PIT财务截面
        u=get_clean_universe(c,rebal_date)

        # 防守端: 季末调仓
        is_quarter_end=rebal_date.month in (3,6,9,12)
        if is_quarter_end or i==0:
            def_score=calc_defense_score(c,rebal_date,u)
            new_def=def_score.head(30)['ts_code'].tolist() if not def_score.empty else []
            if new_def:
                cost=calc_friction(def_hold,new_def,nav_def,0.003)
                nav_def-=cost
            def_hold=new_def

        # 进攻端: 每月调仓
        off_score=calc_offense_score(c,rebal_date,u)
        new_off=off_score.head(30)['ts_code'].tolist() if not off_score.empty else []
        if new_off:
            cost=calc_friction(off_hold,new_off,nav_off,0.004)
            nav_off-=cost
        off_hold=new_off

        # VIX>p95 21天: 全退守
        if 'FULL_DEFENSE' in state:
            days=(next_date-rebal_date).days
            nav_off*=(1+0.02*days/365)  # 现金年化2%
            off_hold=[]

        # 大盘熊市: 进攻资金归入防守
        if bear_market:
            r_def=holdings_return(c,def_hold,rebal_date,next_date) if def_hold else 0
            r_off=r_def  # 进攻资金跟防守走
        else:
            r_def=holdings_return(c,def_hold,rebal_date,next_date) if def_hold else 0
            r_off=holdings_return(c,off_hold,rebal_date,next_date) if off_hold else 0

        # 全退守: 10%死仓+90%现金年化2%
        if 'FULL_DEFENSE' in state:
            days=(next_date-rebal_date).days
            r_def=r_def*0.1+(0.02*days/365)*0.9
            r_off=0.02*days/365

        nav_def*=(1+r_def); nav_off*=(1+r_off)

        # 再平衡
        actual_w=nav_off/(nav_def+nav_off) if (nav_def+nav_off)>0 else 0.5
        if abs(actual_w-w_off)>0.15:
            total=nav_def+nav_off
            nav_def=total*(1-w_off); nav_off=total*w_off

        # 记录
        nav_comb=nav_def+nav_off
        results['dates'].append(rebal_date)
        results['nav_def'].append(nav_def); results['nav_off'].append(nav_off)
        results['nav_comb'].append(nav_comb)
        results['r_def'].append(r_def); results['r_off'].append(r_off)
        results['def_holdings'].append(def_hold[:5]); results['off_holdings'].append(off_hold[:5])

        if (i+1)%12==0: print(f'[{i+1}/{len(months)}] {rebal_date} 净值={nav_comb:.3f}')

    c.close()
    return results

# ═══════════════════════════════
# 三张图
# ═══════════════════════════════

def plot_results(r):
    if not r['dates']: print('无数据'); return
    dates=r['dates']; n=len(dates)
    r_def=np.array(r['r_def']); r_off=np.array(r['r_off'])
    nav_def=np.array(r['nav_def']); nav_off=np.array(r['nav_off'])
    nav_comb=np.array(r['nav_comb'])

    fig,axes=plt.subplots(2,2,figsize=(16,12))

    # 图1: 12月滚动相关性
    ax=axes[0,0]
    roll_r=[]
    for i in range(11,n):
        rr=np.corrcoef(r_def[i-11:i+1],r_off[i-11:i+1])[0,1]
        roll_r.append(rr)
    ax.plot(dates[11:],roll_r,'b-',linewidth=1.5)
    ax.axhline(0,color='gray',ls='--',alpha=0.5)
    ax.axhline(-0.1,color='green',ls='--',alpha=0.3,label='理想线 r=-0.1')
    ax.axhline(0.3,color='red',ls='--',alpha=0.3,label='警戒线 r=0.3')
    # 标注VIX>35点
    for i in range(11,n):
        if r['states'][i]=='FULL_DEFENSE_21D':
            ax.axvline(dates[i],color='orange',alpha=0.3,linewidth=1)
    ax.legend(fontsize=8); ax.set_title('图1: 12月滚动相关性'); ax.grid(alpha=0.3)

    # 图2: 条件相关性
    ax=axes[0,1]
    bins=5
    q_idx=np.argsort(r_off)
    q_size=len(q_idx)//bins
    grp_def=[r_def[q_idx[i*q_size:(i+1)*q_size]].mean()*100 for i in range(bins)]
    grp_off=[r_off[q_idx[i*q_size:(i+1)*q_size]].mean()*100 for i in range(bins)]
    x_labels=[f'进攻最差\n20%',f'进攻较差\n20%','进攻中等\n20%',f'进攻较好\n20%',f'进攻最好\n20%']
    ax.bar(range(bins),grp_def,alpha=0.7,color='blue',label='防守端均值')
    ax.bar(range(bins),grp_off,alpha=0.5,color='red',label='进攻端均值')
    ax.set_xticks(range(bins)); ax.set_xticklabels(x_labels,fontsize=7)
    ax.axhline(0,color='gray',ls='--')
    ax.legend(fontsize=8); ax.set_title('图2: 条件相关性(非对称对冲)'); ax.grid(alpha=0.3)

    # 图3: 净值曲线
    ax=axes[1,0]
    ax.plot(dates,nav_def/nav_def[0],'b-',linewidth=1.5,label='防守端')
    ax.plot(dates,nav_off/nav_off[0],'r-',linewidth=1.5,label='进攻端')
    ax.plot(dates,nav_comb/nav_comb[0],'k-',linewidth=2.5,label='哑铃组合')
    for i in range(n):
        if r['states'][i]=='FULL_DEFENSE_21D':
            ax.axvline(dates[i],color='orange',alpha=0.2,linewidth=2)
    ax.legend(fontsize=8); ax.set_title('图3: 净值曲线(橙色=全退守)'); ax.grid(alpha=0.3)

    # 图4: 回撤
    ax=axes[1,1]
    peak=np.maximum.accumulate(nav_comb)
    dd=(nav_comb/peak-1)*100
    ax.fill_between(dates,dd,0,alpha=0.3,color='gray')
    ax.plot(dates,dd,'k-',linewidth=1)
    ax.set_title('哑铃回撤 (MDD={:.1f}%)'.format(dd.min())); ax.grid(alpha=0.3)

    # 指标
    total_months=n
    ann_ret=(nav_comb[-1]/nav_comb[0])**(12/total_months)-1
    ann_vol=np.std(np.diff(nav_comb)/nav_comb[:-1])*np.sqrt(12)
    sharpe=ann_ret/max(ann_vol,0.01)
    mdd=dd.min()
    calm=ann_ret/max(abs(mdd/100),0.01)
    global_r=np.corrcoef(r_def,r_off)[0,1]

    info_text=(f'年化: {ann_ret*100:.1f}%  波动: {ann_vol*100:.1f}%  Sharpe: {sharpe:.2f}\n'
               f'MDD: {mdd:.1f}%  Calmar: {calm:.2f}  全局r: {global_r:.3f}\n'
               f'防守: {(nav_def[-1]-1)*100:.1f}%  进攻: {(nav_off[-1]-1)*100:.1f}%')
    fig.text(0.5,0.01,info_text,ha='center',fontsize=10,bbox=dict(boxstyle='round',facecolor='wheat'))

    plt.tight_layout(rect=[0,0.06,1,1])
    out='D:/AgentQuant/our/rolling_backtest_result.png'
    plt.savefig(out,dpi=150,bbox_inches='tight')
    print(f'\n图表: {out}')
    print(info_text)

# ═══════════════════════════════
# 主入口
# ═══════════════════════════════

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=2021)
    p.add_argument('--end',type=int,default=2026)
    p.add_argument('--base-w',type=float,default=0.5)
    args=p.parse_args()

    print('='*55)
    print('  哑铃滚动回测引擎')
    print(f'  {args.start}-{args.end}')
    print('='*55)

    r=run_backtest(args.start,args.end,args.base_w)
    plot_results(r)
    print('DONE')
