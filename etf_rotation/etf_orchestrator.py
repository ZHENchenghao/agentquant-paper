# -*- coding: utf-8 -*-
"""
A股ETF轮动策略 v1.0 — 主编排器
30行业ETF代理 × 14因子 × 月度轮动 × 2002-2026回测

运行: python etf_orchestrator.py
"""
import sys, io, os, json, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.etf_universe import (
    load_industry_universe, build_return_matrix,
    get_benchmark, get_industry_names
)
from data.multi_asset import build_defense_pool, defense_portfolio_return
from data.factor_builder import FactorBuilder
from strategy.factor_correlation import FactorCorrelation
from strategy.rotation_engine import RotationEngine
from strategy.regime_filter import RegimeFilter
from backtest.backtest_engine import ETFBacktest


def main():
    print('=' * 80)
    print('  A股ETF轮动策略 v3.0 | 30行业×19因子×资金流×MA200择时 | 2002-2026')
    print('=' * 80)

    t0 = time.time()

    # ============================================================
    # Step 1: 数据加载
    # ============================================================
    print('\n[1/7] 加载30行业价格数据 (2002-01-01 ~ 2026-06-19)...')
    price_df = load_industry_universe('2002-01-01', '2026-06-19')
    ret_matrix = build_return_matrix(price_df)
    industry_names = get_industry_names()

    print(f'  行业数: {len(ret_matrix.columns)}')
    print(f'  交易日: {len(ret_matrix)} ({ret_matrix.index[0].date()} ~ {ret_matrix.index[-1].date()})')
    print(f'  覆盖: {len(industry_names)}个申万行业')

    # ============================================================
    # Step 2: 基准
    # ============================================================
    print('\n[2/7] 加载基准 (沪深300)...')
    bench = get_benchmark('2002-01-01', '2026-06-19')
    if not bench.empty:
        bench_ret = bench['bench_ret']
        print(f'  基准交易日: {len(bench_ret)}')
    else:
        # fallback: 全行业等权
        bench_ret = ret_matrix.mean(axis=1)
        print('  ⚠ 使用行业等权作基准')

    # ============================================================
    # Step 2.5: 大盘择时开关 (MA200 + 斜率 + 缓冲)
    # ============================================================
    print('\n[2.5/8] 大盘择时开关 (MA200双条件+3日缓冲)...')
    regime_filter = RegimeFilter(ma_window=200, slope_window=5, buffer_days=3)

    # 从kline_daily直接获取沪深300收盘价 (2002起, 5930行)
    import duckdb
    con = duckdb.connect('D:/FreeFinanceData/data/duckdb/finance.db', read_only=True)
    hs300_df = con.execute("""
        SELECT trade_date, close FROM kline_daily
        WHERE ts_code='sh000300'
          AND trade_date BETWEEN '2002-01-01' AND '2026-06-19'
        ORDER BY trade_date
    """).df()
    con.close()
    hs300_df['trade_date'] = pd.to_datetime(hs300_df['trade_date'])
    hs300_price = hs300_df.set_index('trade_date')['close']
    print(f'  沪深300: {len(hs300_price)}个交易日')

    regime, ma200, slope = regime_filter.compute_regime(hs300_price)

    if regime is not None:
        # 含崩盘跳闸+急刹车的仓位掩码
        active_ret, aligned_regime, fragile_mask = regime_filter.get_position_mask(
            regime, ret_matrix, benchmark_price=hs300_price
        )
        bull_pct = (aligned_regime == 'BULL').mean() * 100
        bear_pct = (aligned_regime == 'BEAR').mean() * 100
        regime_switches = (aligned_regime != aligned_regime.shift(1)).sum()

        # 熊市占比
        bear_years = aligned_regime.groupby(aligned_regime.index.year).apply(
            lambda x: (x == 'BEAR').mean()
        )
        worst_bear_years = bear_years.nlargest(3)

        print(f'  牛市占比: {bull_pct:.1f}% | 熊市占比: {bear_pct:.1f}%')
        print(f'  状态切换: {regime_switches}次')
        print(f'  熊市最严重的年份:')
        for yr, pct in worst_bear_years.items():
            bar = '█' * int(pct / 5)
            print(f'    {yr}: {pct:.0f}% {bar}')
    else:
        print('  ⚠ 择时数据不足，回退到无择时模式')
        active_ret = ret_matrix
        aligned_regime = None

    # ============================================================
    # Step 2.6: 多资产防御池
    # ============================================================
    print('\n[2.6/8] 构建多资产防御池 (国债+黄金+纳指)...')
    defense_rets = build_defense_pool()
    defense_port = defense_portfolio_return(defense_rets,
        weights={'bond': 0.40, 'gold': 0.30, 'nasdaq': 0.30})

    # 构建多资产版收益矩阵: 熊市日=防御组合收益 (替代空仓)
    # 加防御自身动量过滤: 防御组合60日收益为负 → 回现金避险
    defense_momentum = defense_port.rolling(60).mean() * 252  # 年化
    multi_ret = active_ret.copy()
    if aligned_regime is not None:
        bear_dates = active_ret.index[aligned_regime.reindex(active_ret.index) == 'BEAR']
        for d in bear_dates:
            if d in defense_port.index:
                # 检查防御自身动量
                def_mom = defense_momentum.get(d, 0)
                if pd.isna(def_mom) or def_mom > 0:
                    # 防御资产趋势向上 → 持有多资产
                    multi_ret.loc[d] = defense_port.loc[d]
                # 否则 → 空仓(0收益)
                else:
                    multi_ret.loc[d] = 0

        # 防御组合绩效统计
        def_cum = (1 + defense_port).cumprod()
        def_total = (def_cum.iloc[-1] - 1) * 100
        def_ann = defense_port.mean() * 252 * 100
        def_vol = defense_port.std() * np.sqrt(252) * 100
        def_sh = defense_port.mean() / defense_port.std() * np.sqrt(252) if defense_port.std() > 0 else 0
        print(f'  防御组合(40/30/30): 总{def_total:.0f}% | 年化{def_ann:.1f}% | Sharpe{def_sh:.2f} | 波动{def_vol:.1f}%')
    else:
        multi_ret = active_ret

    # ============================================================
    # Step 3: 因子构建
    # ============================================================
    print('\n[3/8] 构建14维因子体系...')
    fb = FactorBuilder()
    factors = fb.build_all_factors(ret_matrix)
    print(f'  因子列表 ({len(factors)}个):')
    for name in fb.factor_names:
        df = factors[name]
        valid_pct = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
        print(f'    {name}: {df.shape[1]}行业, {valid_pct:.0f}%覆盖')

    # 因子标准化
    normed = fb.normalize_factors(factors, method='rank')

    # ============================================================
    # Step 4: 因子关联检测
    # ============================================================
    print('\n[4/8] 多因子关联检测...')
    fc = FactorCorrelation(window_months=12)
    corr_df, crowding = fc.pairwise_correlation(normed)
    ic_df = fc.factor_ic_trend(factors, ret_matrix)
    regime = fc.detect_factor_regime(ic_df)

    print(f'  因子拥挤度: 均值={crowding.mean():.3f}, 当前={crowding.dropna().iloc[-1]:.3f}')
    print(f'  IC序列: {len(ic_df)}期 × {len(ic_df.columns)}因子')

    # 因子族一致性
    group_consistency = fc.factor_group_correlation(ic_df, normed)
    print('  因子族一致性:')
    for gname, ginfo in group_consistency.items():
        print(f'    {gname}: {ginfo["consistency"]:.2f} ({ginfo["n_factors"]}因子: {",".join(ginfo["factors"][:3])})')

    # 因子状态
    print('  因子状态:')
    for factor, info in sorted(regime.items(), key=lambda x: x[1]['ic_mean'], reverse=True)[:5]:
        status = '✅稳定' if info['stable'] else '⚠不稳定'
        print(f'    {factor}: IC={info["ic_mean"]:.4f}, {info["trend"]}, 翻转风险={info["flip_risk"]:.0%} {status}')

    # ============================================================
    # Step 5: 轮动选ETF
    # ============================================================
    print('\n[5/8] 月度轮动选ETF (Top-5)...')
    engine = RotationEngine(top_n=5, rebalance_freq='M')

    # 动态权重
    latest_crowding = crowding.dropna().iloc[-1] if len(crowding.dropna()) > 0 else None
    weights = engine.adjust_weights_by_correlation(latest_crowding)
    print(f'  拥挤度: {latest_crowding:.3f}' if latest_crowding else '  拥挤度: N/A')
    print(f'  动量权重折扣: {weights.get("mom_63d",0)/engine.default_weights["mom_63d"]:.1%}')

    # 两套选股: 无脆牛限制(基线) + 脆牛限制(择时版)
    selections_full, score_df = engine.select(normed, latest_crowding, weights)
    if aligned_regime is not None:
        selections_cautious, _ = engine.select(normed, latest_crowding, weights,
                                                fragile_mask=fragile_mask)
    else:
        selections_cautious = selections_full
    print(f'  调仓次数: {len(selections_full)} (脆牛限制版: {len(selections_cautious)})')

    # 最近6个月持仓
    recent_dates = sorted(selections_cautious.keys())[-6:]
    print('  最近持仓:')
    for d in recent_dates:
        etfs = selections_cautious[d]
        print(f'    {d.date()}: {etfs}')

    # ============================================================
    # Step 6: 回测 (三版本对比)
    # ============================================================
    print('\n[6/8] 回测 2002-2026 (等权 vs 风险平价 vs 波动率目标)...')

    # 用择时后的active_ret计算波动率目标权重
    vol_weights = engine.compute_weights(
        selections_cautious, active_ret,
        vol_target=0.20, vol_lookback=60, vol_floor=0.10
    )
    print(f'  波动率目标仓位均值: {sum(sum(v.values()) for v in vol_weights.values())/max(len(vol_weights),1)*100:.0f}%')

    # 版本A: 无择时+等权 (用全版选股)
    bt_no = ETFBacktest(ret_matrix, bench_ret, top_n=5, transaction_cost=0.001)
    portfolio_no, trades_no, stats_no = bt_no.run(selections_full)

    # 版本B: MA200择时+空仓 (用脆牛限制版选股)
    bt_timing = ETFBacktest(active_ret, bench_ret, top_n=5, transaction_cost=0.001)
    portfolio_timing, trades_timing, stats_timing = bt_timing.run(selections_cautious)

    # 版本C: MA200择时+风险平价 (脆牛)
    bt_rp = ETFBacktest(active_ret, bench_ret, top_n=5, transaction_cost=0.001)
    portfolio_rp, trades_rp, stats_rp = bt_rp.run(selections_cautious, vol_weights)

    # 版本D: MA200择时+多资产防御+脆牛
    bt_multi = ETFBacktest(multi_ret, bench_ret, top_n=5, transaction_cost=0.001)
    portfolio_multi, trades_multi, stats_multi = bt_multi.run(selections_cautious, vol_weights)

    if portfolio_no is None or portfolio_multi is None:
        print('  ❌ 回测失败')
        return

    # ============================================================
    # Step 7: Tear Sheet (对比)
    # ============================================================
    print('\n[7/8] 绩效对比报告')

    print('\n' + '=' * 80)
    print('  ETF轮动 v2.1 (2002-2026) — 等权 vs MA200 vs 波动率目标')
    print('=' * 80)

    print(f'\n  周期: {stats_no.get("start_date","?")} ~ {stats_no.get("end_date","?")}')
    print(f'  调仓: 月度, 持仓5行业, 单边0.1%')

    print(f'\n  {"指标":<15} {"无择时等权":>12} {"MA200+空仓":>12} {"MA200+多资产":>14}')
    print(f'  {"-"*56}')
    for key, label in [
        ('total_return', '总收益'), ('ann_return', '年化收益'),
        ('sharpe', 'Sharpe'), ('max_drawdown', '最大回撤'),
        ('calmar', 'Calmar'), ('monthly_win_rate', '月胜率'),
    ]:
        v_no = stats_no.get(key, 0)
        v_tm = stats_rp.get(key, 0)
        v_multi = stats_multi.get(key, 0)
        print(f'  {label:<15} {v_no:>+11.1f}% {v_tm:>+11.1f}% {v_multi:>+13.1f}%')

    # 分年对比
    print(f'\n  {"年度":<6} {"无择时":>10} {"MA200+空仓":>12} {"MA200+多资产":>14} {"状态":>6}')
    print(f'  {"-"*52}')
    yearly_no = bt_no.yearly_breakdown(portfolio_no)
    yearly_tm = bt_rp.yearly_breakdown(portfolio_rp)
    yearly_multi = bt_multi.yearly_breakdown(portfolio_multi)
    for yr in sorted(yearly_no.keys()):
        r_no = yearly_no[yr]['return']
        r_tm = yearly_tm.get(yr, {}).get('return', 0)
        r_multi = yearly_multi.get(yr, {}).get('return', 0)
        if aligned_regime is not None:
            yr_mask = aligned_regime.index.year == yr
            bear_pct = (aligned_regime[yr_mask] == 'BEAR').mean() if yr_mask.any() else 0
            tag = f'熊{bear_pct:.0%}' if bear_pct > 0.4 else (f'{bear_pct:.0%}' if bear_pct > 0.1 else '牛')
        else:
            tag = '—'
        print(f'  {yr:<6} {r_no:>+9.1f}% {r_tm:>+11.1f}% {r_multi:>+13.1f}% {tag:>6}')

    # 择时统计
    if aligned_regime is not None:
        print(f'\n  择时统计: 牛市{(aligned_regime=="BULL").mean()*100:.0f}% | '
              f'熊市{(aligned_regime=="BEAR").mean()*100:.0f}% | '
              f'切换{regime_switches}次')

    # 保存
    elapsed = time.time() - t0
    print(f'\n[8/8] 总耗时: {elapsed:.1f}s')

    report = {
        'strategy': 'ETF轮动v2.1 (MA200择时+波动率目标)',
        'universe': '30申万行业指数',
        'without_timing': stats_no,
        'with_timing_rp': stats_rp,
        'with_rp': stats_rp,
        'with_multi_asset': stats_multi,
        'yearly_no': yearly_no,
        'yearly_timing_rp': yearly_tm,
        'yearly_multi': yearly_multi,
    }

    os.makedirs('reports', exist_ok=True)
    ts = pd.Timestamp.now().strftime('%Y-%m-%d_%H%M')
    with open(f'reports/etf_rotation_{ts}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f'  报告: reports/etf_rotation_{ts}.json')
    return portfolio_multi, stats_multi, yearly_multi


if __name__ == '__main__':
    portfolio, stats, yearly = main()
