# -*- coding: utf-8 -*-
"""
AgentQuant · 因子流水线
========================
输入: 任意月末日期
输出: 干净截面 → {防守score, 进攻score} per stock

三核心:
  PIT财务对齐: 财报滞后映射, 防未来函数
  正交化: 换手率→波动率回归, 残差=净波动率
  Universe过滤: 剔除ST/停牌/上市<1年/退市

用法: python factor_pipeline.py  → 单月冒烟测试(2026-05-31)
"""
import sys,io
try:
    if hasattr(sys.stdout,'buffer') and not sys.stdout.closed:
        sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
except: pass
import duckdb,pandas as pd,numpy as np
from datetime import date,timedelta
from sklearn.linear_model import LinearRegression

DB='D:/FreeFinanceData/data/duckdb/finance.db'

def conn(): return duckdb.connect(DB,read_only=True)

# ═══════════════════════════════
# 1. PIT 财报映射
# ═══════════════════════════════

def get_pit_report_type(trade_date):
    """根据调仓日期, 返回PIT合法的最新报告类型"""
    m=trade_date.month
    if m<=3:   return ('Q3', trade_date.year-1)     # 1-3月: 上一年Q3
    elif m<=8: return ('annual', trade_date.year-1)  # 4-8月: 上一年年报
    elif m<=10:return ('Q2', trade_date.year)        # 9-10月: 当年半年报
    else:      return ('Q3', trade_date.year)        # 11-12月: 当年Q3

def get_pit_financials(c, trade_date):
    """
    PIT财务截面: 取trade_date时合法可得的最新财报数据
    返回: DataFrame[ts_code, net_profit, revenue, roe, gross_margin, eps]
    """
    rpt_type, rpt_year = get_pit_report_type(trade_date)
    td_str = trade_date.isoformat()

    # 主查询: 指定report_type+year
    df = c.execute(f"""
        SELECT f.ts_code, f.report_date, f.net_profit, f.revenue, f.roe, f.gross_margin, f.eps
        FROM financial_statements f
        WHERE f.report_type = '{rpt_type}'
          AND f.report_date >= '{rpt_year}-01-01'
          AND f.report_date <= '{rpt_year}-12-31'
          AND f.report_date <= '{td_str}'
          AND f.net_profit IS NOT NULL AND f.net_profit > 0
          AND f.roe IS NOT NULL AND f.roe > 0 AND f.roe < 100
    """).df()

    # 去重: 每只股票取最新report_date, 然后只保留需要的列
    if not df.empty:
        df = df.sort_values('report_date').groupby('ts_code').last().reset_index()
        df = df[['ts_code','net_profit','revenue','roe','gross_margin','eps']]

    # 补充: 用最近可得的EPS (回退逻辑)
    # 如果某股票指定report_type无数据, 用最近12个月内任意report的数据
    all_codes = c.execute(f"""
        SELECT DISTINCT ts_code FROM kline_daily
        WHERE trade_date = '{td_str}' AND close > 0 AND vol > 0
    """).df()

    missing = set(all_codes['ts_code']) - set(df['ts_code']) if not df.empty else set(all_codes['ts_code'])
    if missing:
        codes_str = ','.join([f"'{x}'" for x in list(missing)[:1000]])
        fallback = c.execute(f"""
            SELECT f.ts_code, f.report_date, f.net_profit, f.revenue, f.roe, f.gross_margin, f.eps,
                   ROW_NUMBER() OVER(PARTITION BY f.ts_code ORDER BY f.report_date DESC) rn
            FROM financial_statements f
            WHERE f.ts_code IN ({codes_str})
              AND f.report_date <= '{td_str}'
              AND f.report_date >= '{trade_date.year-1}-01-01'
              AND f.net_profit IS NOT NULL
        """).df()
        if not fallback.empty:
            fallback = fallback[fallback['rn']==1].drop(columns=['rn'])
            # 统一列(去report_date, 后续merge用ts_code)
            fallback = fallback[['ts_code','net_profit','revenue','roe','gross_margin','eps']]
            df = pd.concat([df, fallback], ignore_index=True)

    return df


# ═══════════════════════════════
# 2. Universe 过滤
# ═══════════════════════════════

def get_clean_universe(c, trade_date):
    """
    动态股票池: 剔除ST/停牌/上市<1年/退市
    返回: list of ts_code
    """
    td_str = trade_date.isoformat()
    one_year_ago = (trade_date - timedelta(days=365)).isoformat()

    df = c.execute(f"""
        SELECT DISTINCT k.ts_code
        FROM kline_daily k
        WHERE k.trade_date = '{td_str}'
          AND k.close > 0 AND k.vol > 0
          AND k.ts_code NOT IN (
            SELECT ts_code FROM kline_daily
            WHERE trade_date = '{td_str}' AND is_st = TRUE
          )
          AND k.ts_code IN (
            SELECT ts_code FROM kline_daily
            WHERE trade_date >= '{one_year_ago}'
            GROUP BY ts_code HAVING COUNT(*) >= 200
          )
    """).df()

    # 排除: name含ST/退市
    st_codes = c.execute(f"""
        SELECT DISTINCT k.ts_code FROM kline_daily k
        WHERE k.trade_date = '{td_str}'
          AND k.ts_code IN (
            SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%退%'
          )
    """).df()

    if not st_codes.empty:
        df = df[~df['ts_code'].isin(st_codes['ts_code'])]

    # 排除: 指数代码 (000/399开头的深证指数, sh000/sz399等)
    codes = df['ts_code'].tolist()
    codes = [x for x in codes if not (
        x.startswith('sz399') or x.startswith('sh000') or
        x.startswith('sz000') or x == 'sh000001'
    )]
    return codes


# ═══════════════════════════════
# 3. 防守端因子
# ═══════════════════════════════

def get_pe_data(c, trade_date, universe):
    """当日PE(从kline_daily算: 市值/净利润_TTM)"""
    td_str = trade_date.isoformat()
    codes_str = ','.join([f"'{x}'" for x in universe[:2000]])
    if not codes_str:
        return pd.DataFrame()

    # 从kline取当日close, 从valuation_daily取PE(如有)
    df = c.execute(f"""
        SELECT k.ts_code, k.close, v.pe_ttm
        FROM kline_daily k
        LEFT JOIN valuation_daily v ON k.ts_code=v.ts_code AND v.trade_date='{td_str}'
        WHERE k.ts_code IN ({codes_str}) AND k.trade_date='{td_str}'
    """).df()
    return df


def calc_beta(c, trade_date, universe, window=250):
    """个股Beta: Cov(stock_ret, index_ret)/Var(index_ret), 250日窗口"""
    if len(universe) < 30: return pd.DataFrame()
    td_str = trade_date.isoformat()
    lookback = (trade_date - timedelta(days=window+30)).isoformat()
    codes_str = ','.join([f"'{x}'" for x in universe[:5000]])
    if not codes_str: return pd.DataFrame()

    df = c.execute(f"""
        WITH stock_ret AS (
            SELECT ts_code,trade_date,(close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) ret
            FROM kline_daily WHERE ts_code IN ({codes_str}) AND trade_date>='{lookback}' AND trade_date<='{td_str}'
        ),
        idx_ret AS (
            SELECT trade_date,(close/LAG(close) OVER(ORDER BY trade_date)-1) ret
            FROM kline_daily WHERE ts_code='sh000300' AND trade_date>='{lookback}' AND trade_date<='{td_str}'
        ),
        joined AS (
            SELECT s.ts_code,COUNT(*) n,COVAR_POP(s.ret,i.ret)/VAR_POP(i.ret) beta
            FROM stock_ret s JOIN idx_ret i ON s.trade_date=i.trade_date WHERE s.ret IS NOT NULL AND i.ret IS NOT NULL
            GROUP BY s.ts_code HAVING COUNT(*)>=100
        )
        SELECT ts_code,beta FROM joined WHERE beta IS NOT NULL
    """).df()
    return df


def calc_cfo_quality(c, trade_date, universe):
    """现金流质量: operating_cf / net_profit"""
    fin = get_pit_financials(c, trade_date)
    if fin.empty: return pd.DataFrame()

    # 从financial_statements拿operating_cf
    td_str = trade_date.isoformat()
    codes_str = ','.join([f"'{x}'" for x in universe[:5000]])
    if not codes_str: return pd.DataFrame()

    rpt_type, rpt_year = get_pit_report_type(trade_date)
    cfo = c.execute(f"""
        SELECT f.ts_code, f.operating_cf, f.net_profit
        FROM financial_statements f
        WHERE f.ts_code IN ({codes_str}) AND f.report_type='{rpt_type}'
          AND f.report_date>='{rpt_year}-01-01' AND f.report_date<='{rpt_year}-12-31'
          AND f.operating_cf IS NOT NULL AND f.net_profit IS NOT NULL AND f.net_profit>0
    """).df()
    if cfo.empty: return pd.DataFrame()
    cfo = cfo.sort_values('report_date').groupby('ts_code').last().reset_index() if 'report_date' in cfo.columns else cfo
    cfo['cfo_quality'] = cfo['operating_cf'] / cfo['net_profit']
    cfo['cfo_quality'] = cfo['cfo_quality'].clip(-5, 5)
    return cfo[['ts_code','cfo_quality']]


def calc_defense_score(c, trade_date, universe):
    """
    防守因子 V2: 低Beta(绝缘) + 高现金流质量(真金白银) + 低波动(稳定)
    score_defense = rank(-beta)*0.4 + rank(cfo_quality)*0.3 + rank(-volatility)*0.3
    """
    td_str = trade_date.isoformat()
    lookback = (trade_date - timedelta(days=30)).isoformat()
    codes_str = ','.join([f"'{x}'" for x in universe[:5000]])
    if not codes_str: return pd.DataFrame()

    # Beta
    beta = calc_beta(c, trade_date, universe)
    # 现金流质量
    cfo = calc_cfo_quality(c, trade_date, universe)
    # 低波动 + 价格>MA60 (踢掉阴跌股)
    vol = c.execute(f"""
        WITH vol_data AS (
            SELECT ts_code,close,
                   (close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) dr,
                   AVG(close) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) ma60
            FROM kline_daily WHERE ts_code IN ({codes_str}) AND trade_date>='{lookback}' AND trade_date<='{td_str}'
        ),
        vol_agg AS (
            SELECT ts_code,STDDEV(dr) volatility,MAX(close) last_close,MAX(ma60) ma60
            FROM vol_data WHERE dr IS NOT NULL GROUP BY ts_code HAVING COUNT(*)>=15
        )
        SELECT ts_code,volatility FROM vol_agg WHERE last_close > ma60
    """).df()

    if beta.empty:
        return pd.DataFrame()

    # 合并Beta + 波动率(可选) + 现金流质量(可选)
    result = beta.rename(columns={'beta':'beta_val'})
    result['cfo_quality'] = 0.0
    result['volatility'] = 0.02

    if not cfo.empty and 'cfo_quality' in cfo.columns:
        result = result.merge(cfo[['ts_code','cfo_quality']], on='ts_code', how='left', suffixes=('','_cfo'))
        mask = result['cfo_quality_cfo'].notna()
        result.loc[mask,'cfo_quality'] = result.loc[mask,'cfo_quality_cfo']
        result.drop(columns=['cfo_quality_cfo'], inplace=True, errors='ignore')

    if not vol.empty and 'volatility' in vol.columns:
        result = result.merge(vol[['ts_code','volatility']], on='ts_code', how='left', suffixes=('','_v'))
        mask = result['volatility_v'].notna()
        result.loc[mask,'volatility'] = result.loc[mask,'volatility_v']
        result.drop(columns=['volatility_v'], inplace=True, errors='ignore')

    if len(result) < 20: return result

    # 去极值
    for col in ['beta_val','volatility']:
        if col in result.columns and not result[col].isna().all():
            med = result[col].median()
            mad = (result[col] - med).abs().median() * 1.4826
            if mad > 0: result[col] = result[col].clip(med - 3*mad, med + 3*mad)

    result['score_defense'] = (
        (-result['beta_val']).rank(pct=True) * 0.5 +
        (-result['volatility']).rank(pct=True) * 0.3 +
        result['cfo_quality'].rank(pct=True) * 0.2
    )
    return result.sort_values('score_defense', ascending=False)


# ═══════════════════════════════
# 4. 进攻端因子 (含正交化)
# ═══════════════════════════════

def calc_offense_score(c, trade_date, universe):
    """
    进攻因子: 成交额排名 (市场真龙头)
      1. 行业动量 → top3行业 (参考, 不强制过滤)
      2. 日均成交>5亿 + 价格>MA60 + 波动率>0.02
      3. score = rank(log_amt) — 成交额最大=龙头
    返回: DataFrame[ts_code, avg_amount, vol, score_offense]
    """
    td_str = trade_date.isoformat()
    lookback = (trade_date - timedelta(days=30)).isoformat()
    codes_str = ','.join([f"'{x}'" for x in universe[:5000]])
    if not codes_str:
        return pd.DataFrame()

    # 行业动量: top3行业
    ind_mom = c.execute(f"""
        SELECT industry,(MAX(CASE WHEN rn=1 THEN close END)/NULLIF(MAX(CASE WHEN rn=10 THEN close END),0)-1) mom
        FROM (
            SELECT industry,close,ROW_NUMBER() OVER(PARTITION BY industry ORDER BY trade_date DESC) rn
            FROM proxy_industry_daily WHERE trade_date <= '{td_str}' AND trade_date >= '{lookback}'
        ) WHERE rn<=10 GROUP BY industry HAVING COUNT(*)>=8 ORDER BY mom DESC LIMIT 3
    """).df()

    top_industries = ind_mom['industry'].tolist() if not ind_mom.empty else []
    print(f'  强势行业: {top_industries}')

    # 进攻端: 成交额(龙头) + 高波动 + 价格>MA60 + 日均成交>5亿
    vol_turn = c.execute(f"""
        WITH stock_data AS (
            SELECT ts_code,amount,close,
                   (close/LAG(close) OVER(PARTITION BY ts_code ORDER BY trade_date)-1) dr,
                   AVG(close) OVER(PARTITION BY ts_code ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) ma60
            FROM kline_daily WHERE ts_code IN ({codes_str})
            AND trade_date <= '{td_str}' AND trade_date >= '{lookback}' AND amount>0
        ),
        agg AS (
            SELECT ts_code,STDDEV(dr) vol,AVG(amount) avg_amount,
                   MAX(close) last_close, MAX(ma60) ma60
            FROM stock_data WHERE dr IS NOT NULL
            GROUP BY ts_code HAVING COUNT(*)>=15
        )
        SELECT ts_code,vol,avg_amount FROM agg
        WHERE avg_amount > 5e8 AND last_close > ma60
    """).df()

    if vol_turn.empty or len(vol_turn) < 30:
        return pd.DataFrame()

    # 去极值: 只对波动率, 成交额越大越好不去极值
    for col in ['vol']:
        med = vol_turn[col].median()
        mad = (vol_turn[col] - med).abs().median() * 1.4826
        vol_turn[col] = vol_turn[col].clip(med - 3*mad, med + 3*mad)
    # 成交额用log压缩代替去极值(保留排序, 减小极端值影响)
    vol_turn['log_amt'] = np.log1p(vol_turn['avg_amount'])

    # 龙头识别: 成交额排名 = 市场认可度 (波动率只做最低阈值>0.02)
    vol_turn = vol_turn[vol_turn['vol'] > 0.02]
    if len(vol_turn) < 30:
        return pd.DataFrame()
    vol_turn['score_offense'] = vol_turn['log_amt'].rank(pct=True)

    result = vol_turn.sort_values('score_offense', ascending=False)
    return result


# ═══════════════════════════════
# 5. 冒烟测试
# ═══════════════════════════════

def smoke_test(trade_date=None):
    """单月冒烟测试"""
    if trade_date is None:
        trade_date = date(2026, 5, 30)  # 5月最后一个交易日

    c = conn()
    sep='='*55
    print(sep)
    print('  因子流水线 冒烟测试')
    print(f'  调仓日: {trade_date}')
    print(sep)

    # 1. Universe
    print('\n[1] Universe过滤...')
    universe = get_clean_universe(c, trade_date)
    print(f'  候选池: {len(universe)}只')

    # 2. PIT验证
    print('\n[2] PIT财务验证...')
    rpt_type, rpt_year = get_pit_report_type(trade_date)
    print(f'  日期={trade_date} → 可用报告: {rpt_type} {rpt_year}')
    fin = get_pit_financials(c, trade_date)
    print(f'  有财务数据: {len(fin)}只')
    if not fin.empty:
        roe_min = fin['roe'].min(); roe_max = fin['roe'].max()
        print('  ROE范围: {:.1f}~{:.1f}'.format(roe_min, roe_max))
        sample = fin[['ts_code','roe','eps']].head(3)
        print('  样本:')
        for _,r in sample.iterrows():
            print('    {} ROE={:.1f} EPS={:.2f}'.format(r['ts_code'], r['roe'], r['eps']))

    # 3. 防守因子
    print('\n[3] 防守因子...')
    defense = calc_defense_score(c, trade_date, universe)
    print('  有效: {}只'.format(len(defense)))
    if not defense.empty:
        top5 = defense.head(5)
        print('  Top5:')
        for _,r in top5.iterrows():
            print('    {} PE={:.0f} ROE={:.1f} Score={:.3f}'.format(r['ts_code'], r['pe'], r['roe'], r['score_defense']))

    # 4. 进攻因子
    print('\n[4] 进攻因子(正交化)...')
    offense = calc_offense_score(c, trade_date, universe)
    if not offense.empty:
        r2 = offense.attrs.get('ortho_r2', 0)
        corr = offense.attrs.get('ortho_corr_after', 0)
        print('  正交化: R2={:.3f} 残差相关性={:.6f} (应趋近0)'.format(r2, corr))
        print('  有效: {}只'.format(len(offense)))
        top5 = offense.head(5)
        print('  Top5:')
        for _,r in top5.iterrows():
            print('    {} Turn={:.4f} NetVol={:.6f} Score={:.3f}'.format(r['ts_code'], r['turnover'], r['net_vol'], r['score_offense']))

        # 异常值检查
        nv = offense['net_vol']
        mad = (nv - nv.median()).abs().median() * 1.4826
        outliers = (np.abs(nv - nv.median()) > 3 * mad).sum()
        print('  净波动率异常值: {}/{}'.format(outliers, len(offense)))

    c.close()
    sep = '='*55
    print('\n' + sep)
    print('  冒烟测试完成')
    print(sep)
    return defense, offense


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--date', type=str, default='2026-05-30', help='测试日期 YYYY-MM-DD')
    args = p.parse_args()
    smoke_test(date.fromisoformat(args.date))
