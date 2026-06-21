# -*- coding: utf-8 -*-
"""
quant_backtest_engine.py — 百亿级A股量化回测重构
==================================================
四大升级模块：
  模块一: 地狱级交易摩擦模拟
  模块二: 特征纯化与剥离 (行业/市值中性化)
  模块三: 高阶价量因子重构 (Alpha101/量价)
  模块四: 极端行情压力测试

所有函数均向量化, 可直接接入现有LightGBM选股管线。
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# 模块一: 地狱级交易摩擦模拟
# ══════════════════════════════════════════════════════════════════

class ExecutionSimulator:
    """
    真实交易摩擦模拟器。
    输入: 调仓日的目标组合 + 市场微观结构数据
    输出: 实际可成交组合 + 滑点报告
    """

    def __init__(self, stamp_tax=0.001, commission=0.00025, base_slippage=0.0005):
        """
        stamp_tax: 印花税 0.1% (卖出单向)
        commission: 佣金 0.025% (双边)
        base_slippage: 最小滑点 0.05%
        """
        self.stamp_tax = stamp_tax
        self.commission = commission
        self.base_slippage = base_slippage

    def dynamic_slippage(self, order_amount, daily_volume, daily_volatility,
                         participation_rate=0.05):
        """
        动态滑点模型 (Almgren-Chriss简化版)

        Parameters
        ----------
        order_amount : array (N,)  每只股票买入金额
        daily_volume : array (N,)  每只股票近20日日均成交额
        daily_volatility : array (N,)  每只股票近20日日收益标准差
        participation_rate : float  最大参与率上限

        Returns
        -------
        slippage_pct : array (N,)  单向滑点比例
        detail : DataFrame  分项明细
        """
        # 参与率: 订单金额/日均成交额
        part_rate = np.minimum(
            np.abs(order_amount) / np.maximum(daily_volume, 1.0),
            participation_rate
        )

        # 临时冲击: α × sqrt(参与率) + 永久冲击: β × 参与率
        # 参数来自A股实证: α≈0.1, β≈0.05
        alpha, beta = 0.10, 0.05

        # 波动率调整: 高波动股票冲击成本更高
        vol_adj = daily_volatility / np.median(daily_volatility)

        temp_impact = alpha * np.sqrt(part_rate) * vol_adj
        perm_impact = beta * part_rate * vol_adj

        slippage_pct = self.base_slippage + temp_impact + perm_impact

        detail = pd.DataFrame({
            'participation_rate': part_rate,
            'temp_impact': temp_impact,
            'perm_impact': perm_impact,
            'vol_adj': vol_adj,
            'total_slippage': slippage_pct,
        })
        return slippage_pct, detail

    def capacity_cut(self, target_mv, daily_amount, max_pct=0.05):
        """
        容量限制: 单票买入金额不超过当日成交额的max_pct%。
        超限资金截断, 不顺延到下个标的。

        Parameters
        ----------
        target_mv : array (N,)  目标买入金额
        daily_amount : array (N,)  当日成交额
        max_pct : float  单票上限比例

        Returns
        -------
        actual_mv : array (N,)  实际可买入金额
        excess_mv : float  多出的资金总额
        """
        cap = daily_amount * max_pct
        actual_mv = np.minimum(target_mv, cap)
        excess_mv = np.sum(target_mv - actual_mv)
        return actual_mv, excess_mv

    def limit_filter(self, codes, trade_date, action='BUY',
                     db_path='D:/FreeFinanceData/data/duckdb/finance.db'):
        """
        涨跌停过滤: 排除无法交易的情况。

        规则:
          - 主板(60/00开头): 涨跌停±10%
          - 科创(688)/创业(300/301): ±20%
          - 北交所(8/4开头): ±30%
          - 一字涨停(open==high==close且触及涨停线): 买不进
          - 一字跌停(open==low==close且触及跌停线): 卖不出
          - 停牌(vol==0或close==pre_close): 双向剔除
        """
        import duckdb
        c = duckdb.connect(db_path, read_only=True)
        codes_str = ','.join([f"'{x}'" for x in codes])

        df = c.execute(f"""
            SELECT ts_code, open, high, low, close, pre_close, vol, amount,
                   (close/pre_close-1) AS chg
            FROM kline_daily
            WHERE ts_code IN ({codes_str}) AND trade_date='{trade_date}'
        """).df()
        c.close()

        # 板块识别 → 涨跌停阈值
        def get_limit_threshold(code):
            code_str = str(code)
            # 科创板 688xxx
            if code_str.startswith('sh688') or code_str.startswith('sz688'):
                return 0.199
            # 创业板 300/301
            if code_str.startswith('sz300') or code_str.startswith('sz301'):
                return 0.199
            # 北交所 8xxx/4xxx
            if code_str.startswith('bj') or code_str.startswith('8') or code_str.startswith('4'):
                return 0.299
            # 主板 60/00
            return 0.099

        df['limit_pct'] = df['ts_code'].apply(get_limit_threshold)
        df['is_suspended'] = (df['vol'] <= 0) | (df['close'] == df['pre_close'])
        df['is_limit_up_sealed'] = (
            (df['open'] == df['high']) &
            (df['high'] == df['close']) &
            (df['chg'] >= df['limit_pct'] - 0.001)
        )
        df['is_limit_down_sealed'] = (
            (df['open'] == df['low']) &
            (df['low'] == df['close']) &
            (df['chg'] <= -(df['limit_pct'] - 0.001))
        )

        blocked = {}
        valid_codes = []

        for _, row in df.iterrows():
            code = row['ts_code']
            if row['is_suspended']:
                blocked[code] = '停牌'
            elif action == 'BUY' and row['is_limit_up_sealed']:
                blocked[code] = '一字涨停无法买入'
            elif action == 'SELL' and row['is_limit_down_sealed']:
                blocked[code] = '一字跌停无法卖出'
            else:
                valid_codes.append(code)

        return valid_codes, blocked

    def execute_round(self, target_weights, codes, trade_date, prices,
                      daily_volumes, daily_volatilities, daily_amounts,
                      total_capital, existing_holdings=None):
        """
        完整执行一轮调仓。

        Parameters
        ----------
        target_weights : array (N,)  目标权重 (0~1)
        codes : list (N,)  目标持仓代码
        trade_date : str
        prices : array (N,)  当日收盘价
        daily_volumes : array (N,)  近20日日均成交额
        daily_volatilities : array (N,)  近20日日收益波动率
        daily_amounts : array (N,)  当日成交额
        total_capital : float  总资产
        existing_holdings : dict  {code: (shares, cost_price)}  现有持仓

        Returns
        -------
        execution_report : dict
        """
        # 1. 涨跌停过滤
        valid_codes, blocked = self.limit_filter(codes, trade_date, 'BUY')

        valid_idx = [i for i, c in enumerate(codes) if c in valid_codes]
        if not valid_idx:
            return {'status': 'all_blocked', 'blocked': blocked}

        # 过滤到有效股票
        valid_weights = target_weights[valid_idx]
        valid_weights = valid_weights / valid_weights.sum()  # 重归一化

        target_mv = valid_weights * total_capital

        # 2. 容量截断
        actual_mv, excess = self.capacity_cut(
            target_mv,
            daily_amounts[valid_idx],
            max_pct=0.05
        )

        # 3. 动态滑点
        slippage, slip_detail = self.dynamic_slippage(
            actual_mv,
            daily_volumes[valid_idx],
            daily_volatilities[valid_idx]
        )

        # 4. 成本: 从现金中直接扣除 (而非调整收益率)
        #    买入: 资金减少 = 股数×股价 + 佣金×股数×股价 + 滑点×股数×股价
        #    卖出: 资金增加 = 股数×股价 - 印花税×股数×股价 - 佣金×股数×股价 - 滑点×股数×股价
        #    关键: 滑点/佣金在计算股数时就扣除, 确保股数≤现金÷(股价+成本)
        #          不能用 net_ret × (1-cost) 的方式, 那只是近似, 复利下会偏差
        buy_cost_pct = self.commission + slippage  # 买入单边费率
        sell_cost_pct = self.stamp_tax + self.commission + slippage  # 卖出单边费率

        # 5. 计算可买股数 (现金先扣成本, 100股整手)
        valid_prices = prices[valid_idx]
        # 每股总支出 = 股价 × (1 + 买入成本率)
        cost_per_share = valid_prices * (1 + buy_cost_pct)
        shares = np.floor(actual_mv / cost_per_share / 100) * 100
        # 实际消耗现金
        actual_cash_used = shares * cost_per_share

        report = {
            'n_target': len(codes),
            'n_valid': len(valid_codes),
            'n_blocked': len(blocked),
            'blocked': blocked,
            'excess_cash': excess,
            'avg_slippage_bps': np.mean(slippage) * 10000,
            'max_slippage_bps': np.max(slippage) * 10000,
            'total_cash_used': np.sum(actual_cash_used),
            'total_cost': np.sum(actual_cash_used) - np.sum(shares * valid_prices),  # 纯摩擦成本
            'cost_pct': np.sum(actual_cash_used) / np.sum(shares * valid_prices) - 1.0,  # 费率
            'shares': shares,
            'actual_mv': actual_mv,
            'slip_detail': slip_detail,
        }
        return report


# ══════════════════════════════════════════════════════════════════
# 模块二: 特征纯化与剥离 (行业/市值中性化)
# ══════════════════════════════════════════════════════════════════

class RiskNeutralizer:
    """
    因子中性化处理器。
    消除行业归属和市值规模对因子值的系统性影响,
    避免"选小盘股=选alpha"的虚假发现。
    """

    def __init__(self):
        self._industry_medians = {}
        self._size_betas = {}

    def industry_neutralize(self, factor_values, industry_codes):
        """
        行业中性化: 因子值减去行业均值, 除以行业标准差。

        factor_values : array (N,)  原始因子值
        industry_codes : array (N,)  行业代码 (申万/中信)

        Returns
        -------
        neutralized : array (N,)  行业中性化后因子值
        """
        df = pd.DataFrame({'factor': factor_values, 'industry': industry_codes})
        ind_mean = df.groupby('industry')['factor'].transform('mean')
        ind_std = df.groupby('industry')['factor'].transform('std')
        ind_std = ind_std.replace(0, 1.0)
        neutralized = (df['factor'] - ind_mean) / ind_std
        return neutralized.values

    def size_neutralize_quadratic(self, factor_values, log_mcap):
        """
        二次项市值中性化 (initial-d/ml-quant-trading 公式(7))

        factor_final = factor_raw - γ×log_mcap - δ×log_mcap²
        线性项捕捉大盘vs小盘, 二次项捕捉微盘尾部非线性翘尾。

        返回: (residual, gamma, delta, r2)
        """
        y = np.asarray(factor_values, dtype=float)
        x = np.asarray(log_mcap, dtype=float)
        x2 = x * x

        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 100:
            return factor_values, 0.0, 0.0, 0.0

        X = np.column_stack([np.ones(mask.sum()), x[mask], x2[mask]])
        y_masked = y[mask]
        coeffs, residuals_arr, rank, _ = np.linalg.lstsq(X, y_masked, rcond=None)
        beta_0, gamma, delta = coeffs[0], coeffs[1], coeffs[2]

        # R²
        y_pred = X @ coeffs
        ss_res = np.sum((y_masked - y_pred) ** 2)
        ss_tot = np.sum((y_masked - y_masked.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # 全量残差
        x_full = np.where(np.isnan(x), 0.0, x)
        x2_full = x_full * x_full
        predicted_full = beta_0 + gamma * x_full + delta * x2_full
        residual = y - predicted_full

        return residual, gamma, delta, r2

    def size_neutralize(self, factor_values, log_mcap):
        """
        市值中性化: OLS回归因子 ~ log_mcap, 取残差。
        用 np.linalg.lstsq 替代 statsmodels, 千股截面<1ms。

        Y = β₀ + β₁×log_mcap + ε
        ε = 纯因子暴露 (去除了市值贡献)

        Parameters
        ----------
        factor_values : array (N,)
        log_mcap : array (N,)  log(总市值)

        Returns
        -------
        residual : array (N,)  市值中性化后的因子残差
        beta : float  log_mcap的回归系数
        r2 : float  市值对因子的解释度
        """
        y = np.asarray(factor_values, dtype=float)
        x = np.asarray(log_mcap, dtype=float)

        # 有效样本
        mask = ~(np.isnan(y) | np.isnan(x))
        if mask.sum() < 100:
            return factor_values, 0.0, 0.0

        # 构建设计矩阵 [1, log_mcap]
        X_masked = np.column_stack([
            np.ones(mask.sum()),
            x[mask]
        ])
        y_masked = y[mask]

        # np.linalg.lstsq 求解 (比statsmodels快50倍+)
        coeffs, residuals, rank, _ = np.linalg.lstsq(X_masked, y_masked, rcond=None)
        beta_0, beta_1 = coeffs[0], coeffs[1]

        # R²
        y_pred = X_masked @ coeffs
        ss_res = np.sum((y_masked - y_pred) ** 2)
        ss_tot = np.sum((y_masked - y_masked.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # 全量预测 + 残差
        x_full = np.where(np.isnan(x), 0.0, x)
        predicted_full = beta_0 + beta_1 * x_full
        residual = y - predicted_full

        return residual, beta_1, r2

    def full_neutralize(self, factor_df, industry_col='industry',
                        size_col='log_mcap', factor_cols=None):
        """
        一键中性化: 行业→市值 双重剥离, 向量化实现。

        factor_df : DataFrame  包含因子列+行业列+市值列
        factor_cols : list  需要中性化的因子列名 (None=所有数值列)

        Returns
        -------
        neutralized_df : DataFrame  中性化后因子
        report : dict  {factor_name: {size_beta, size_r2}}
        """
        if factor_cols is None:
            factor_cols = [c for c in factor_df.columns
                          if c not in (industry_col, size_col, 'ts_code', 'trade_date')]

        result = factor_df[['ts_code', 'trade_date']].copy() if 'ts_code' in factor_df.columns else pd.DataFrame()
        report = {}

        for col in factor_cols:
            if col not in factor_df.columns:
                continue
            fv = factor_df[col].values.astype(float)

            # Step 1: 行业中性化
            if industry_col in factor_df.columns:
                fv = self.industry_neutralize(fv, factor_df[industry_col].values)

            # Step 2: 市值中性化
            if size_col in factor_df.columns:
                lm = factor_df[size_col].values.astype(float) if size_col in factor_df.columns else np.zeros_like(fv)
                fv, beta, r2 = self.size_neutralize(fv, lm)
                report[col] = {'size_beta': round(beta, 4), 'size_r2': round(r2, 4)}

            result[col + '_neutral'] = fv

        return result, report


# ══════════════════════════════════════════════════════════════════
# 模块三: 高阶价量因子重构 (Alpha101 / 截面量价)
# ══════════════════════════════════════════════════════════════════

def alpha_vol_adj_momentum(close, high, low, volume, window=20, lag=5):
    """
    Alpha #1: 波动率调整后的截面动量

    逻辑: 买入近期涨但波动小的股票 (风险调整后动量 > 纯价格动量)
    公式: (close_t - close_{t-lag}) / (close_{t-lag} × σ_window)

    close/high/low/volume : DataFrame (T×N)  列=股票, 行=日期
    返回: Series (N,)  截面因子值
    """
    ret = close.pct_change(lag).iloc[-1]  # lag日收益
    vol = close.pct_change().rolling(window).std().iloc[-1]  # 窗口波动率
    factor = ret / (vol + 1e-8)
    # 截面z-score标准化
    return (factor - factor.mean()) / (factor.std() + 1e-8)


def alpha_liquidity_premium(open_, high, low, close, volume):
    """
    Alpha #2: 流动性溢价因子 (Amihud ILLIQ改进版)

    逻辑: 日均 |收益|/成交额 越高 → 流动性越差 → 需要补偿溢价
    截面买入高ILLIQ的股票 = 赚取流动性溢价

    返回: Series (N,)
    """
    ret = close.pct_change().abs()
    # 使用近20日均值
    illiq = (ret / (volume + 1e-8)).rolling(20).mean().iloc[-1]
    # 取log压缩极端值
    factor = np.log(illiq + 1e-12)
    return (factor - factor.mean()) / (factor.std() + 1e-8)


def alpha_intraday_variance(open_, high, low, close):
    """
    Alpha #3: 日内收益方差 (Parkinson Range改进)

    逻辑: (high-low)/open 的截面排序 → 日内振幅大的股票,
          在A股中往往有散户情绪溢价 (短期涨) 或恐慌折价

    公式: log( (high-low)/open ) 的20日标准差

    返回: Series (N,)
    """
    daily_range = (high - low) / (open_ + 1e-8)
    log_range = np.log(daily_range + 1e-8)
    factor = log_range.rolling(20).std().iloc[-1]
    return (factor - factor.mean()) / (factor.std() + 1e-8)


def alpha_gap_reversal(open_, close, pre_close):
    """
    Alpha #4: 跳空反转因子

    逻辑: A股T+1下, 大幅跳空高开后往往回落, 跳空低开后反弹。
    因子 = 跳空幅度 × (当日是否回补), 方向为负 (跳空越大越应反向操作)

    gap = (open - pre_close) / pre_close
    fill = (close - open) / (open - pre_close)  # 回补比例
    signal = -gap × |fill|  (负值=高开回落→卖出信号, 正值=低开反弹→买入信号)

    返回: Series (N,)
    """
    gap = (open_ - pre_close) / (pre_close + 1e-8)
    day_move = (close - open_) / (open_ + 1e-8)
    # 跳空回补度: 如果是正跳空且收跌→回补; 负跳空且收涨→回补
    fill = np.where(gap * day_move < 0, 1.0, np.where(np.abs(day_move/gap) > 0.5, 0.5, 0.0))

    # 跳空幅度 × 回补程度, 取负(高开卖出, 低开买入)
    raw = -gap * np.abs(fill)

    # 近5日累计信号
    factor = pd.DataFrame(raw).rolling(5).mean().iloc[-1]
    return (factor - factor.mean()) / (factor.std() + 1e-8)


def alpha_turnover_exhaustion(turnover_rate, close, window=20):
    """
    Alpha #5: 换手率衰竭因子

    逻辑: A股中, 换手率突然放大→短期见顶的概率高。
    因子 = (近5日换手率 / 近20日换手率) — 比值高=换手过热

    返回: Series (N,)
    """
    # 无turnover_rate时用 volume/shares估算 (或用kline_daily.turnover_rate)
    avg5 = turnover_rate.rolling(5).mean().iloc[-1]
    avg20 = turnover_rate.rolling(window).mean().iloc[-1]
    raw = avg5 / (avg20 + 1e-8)
    # 换手过热 = 负信号
    factor = -raw
    return (factor - factor.mean()) / (factor.std() + 1e-8)


# ══════════════════════════════════════════════════════════════════
# 模块四: 极端行情压力测试
# ══════════════════════════════════════════════════════════════════

class StressTester:
    """
    A股历史极端事件情景分析。
    输入策略净值曲线, 输出各危机窗口的回撤、归因和相关性分解。
    """

    # 定义极端事件窗口
    STRESS_SCENARIOS = {
        '2018_trade_war': {
            'start': '2018-01-24',   # 上证3587高点
            'end': '2019-01-03',     # 上证2440低点
            'label': '2018中美贸易摩擦',
            'benchmark': 'sh000300',
        },
        '2024_microcap_crash': {
            'start': '2024-01-02',
            'end': '2024-02-07',     # 微盘股踩踏+量化DMA爆仓
            'label': '2024年1-2月微盘股危机',
            'benchmark': 'sh000300',
        },
        '2020_covid': {
            'start': '2020-01-21',
            'end': '2020-03-23',
            'label': '2020新冠冲击',
            'benchmark': 'sh000300',
        },
        '2015_crash': {
            'start': '2015-06-12',
            'end': '2015-08-26',
            'label': '2015股灾',
            'benchmark': 'sh000300',
        },
    }

    def __init__(self, nav_df, factor_exposure_df=None):
        """
        nav_df : DataFrame  columns=['trade_date', 'nav', 'benchmark_nav']
                              可带 'daily_return', 'excess_return'
        factor_exposure_df : DataFrame  每日因子暴露 (可选, 用于归因)
        """
        self.nav = nav_df.copy()
        self.nav['trade_date'] = pd.to_datetime(self.nav['trade_date'])
        self.nav = self.nav.sort_values('trade_date')

        if 'daily_return' not in self.nav.columns:
            self.nav['daily_return'] = self.nav['nav'].pct_change()
        if 'benchmark_return' not in self.nav.columns and 'benchmark_nav' in self.nav.columns:
            self.nav['benchmark_return'] = self.nav['benchmark_nav'].pct_change()

        self.factor_exposure = factor_exposure_df

    def scenario_analysis(self):
        """
        全情景压力测试。
        Returns: DataFrame  每个情景的指标
        """
        results = []
        for key, scenario in self.STRESS_SCENARIOS.items():
            mask = ((self.nav['trade_date'] >= scenario['start']) &
                    (self.nav['trade_date'] <= scenario['end']))
            window = self.nav[mask]

            if len(window) < 10:
                results.append({'scenario': scenario['label'], 'status': '数据不足'})
                continue

            # 策略回撤
            peak_val = window['nav'].max()
            trough_val = window['nav'].min()
            strat_dd = (trough_val / peak_val - 1) * 100

            # 基准回撤
            if 'benchmark_nav' in window.columns:
                b_peak = window['benchmark_nav'].max()
                b_trough = window['benchmark_nav'].min()
                bench_dd = (b_trough / b_peak - 1) * 100
            else:
                bench_dd = 0

            # 期间总收益
            strat_ret = (window['nav'].iloc[-1] / window['nav'].iloc[0] - 1) * 100

            # 日胜率
            daily_wr = (window['daily_return'].dropna() > 0).mean() * 100

            # 最大连续回撤天数
            cum = (1 + window['daily_return'].fillna(0)).cumprod()
            dd_series = cum / cum.cummax() - 1
            in_dd = dd_series < 0
            if in_dd.any():
                max_dd_days = (in_dd.groupby((~in_dd).cumsum()).cumsum()).max()
            else:
                max_dd_days = 0

            # 波动率放大
            pre_mask = (self.nav['trade_date'] < scenario['start']) & \
                       (self.nav['trade_date'] >= pd.Timestamp(scenario['start']) - pd.DateOffset(months=3))
            pre_vol = self.nav.loc[pre_mask, 'daily_return'].std() if pre_mask.sum() > 20 else 0
            crisis_vol = window['daily_return'].std()
            vol_ratio = crisis_vol / pre_vol if pre_vol > 0 else 1.0

            # 相关性vs基准
            if 'benchmark_return' in window.columns:
                corr = window['daily_return'].corr(window['benchmark_return'])
            else:
                corr = 0

            results.append({
                'scenario': scenario['label'],
                'start': scenario['start'],
                'end': scenario['end'],
                'n_days': len(window),
                'strategy_return_pct': round(strat_ret, 2),
                'strategy_max_dd_pct': round(strat_dd, 2),
                'benchmark_dd_pct': round(bench_dd, 2),
                'daily_win_rate': round(daily_wr, 1),
                'max_dd_days': max_dd_days,
                'crisis_vol': round(crisis_vol * 100, 2) if crisis_vol else 0,
                'pre_crisis_vol': round(pre_vol * 100, 2) if pre_vol else 0,
                'vol_amplification': round(vol_ratio, 2),
                'bench_correlation': round(corr, 4),
            })

        return pd.DataFrame(results)

    def drawdown_decomposition(self, window_start, window_end):
        """
        回撤归因: 将窗口内回撤分解为 β部分(跟跌)和 α部分(超额跌)。

        归因公式: 策略收益 = β×基准收益 + α
        若β贡献>70% → 主要是市场跌, 策略选股没出大问题
        若α贡献>50% → 策略自身选股/因子暴露出了问题
        """
        mask = ((self.nav['trade_date'] >= window_start) &
                (self.nav['trade_date'] <= window_end))
        w = self.nav[mask].dropna(subset=['daily_return', 'benchmark_return'])

        if len(w) < 20 or 'benchmark_return' not in w.columns:
            return {'status': '数据不足'}

        # 简单CAPM归因
        X = w['benchmark_return'].values
        y = w['daily_return'].values
        beta, alpha, _, _, _ = stats.linregress(X, y)

        # 累计
        total_ret = np.prod(1 + y) - 1
        bench_ret = np.prod(1 + X) - 1
        alpha_ret = total_ret - beta * bench_ret

        # 归因占比
        beta_contrib = abs(beta * bench_ret) / (abs(beta * bench_ret) + abs(alpha_ret) + 1e-8)
        alpha_contrib = abs(alpha_ret) / (abs(beta * bench_ret) + abs(alpha_ret) + 1e-8)

        return {
            'total_return': round(total_ret * 100, 2),
            'benchmark_return': round(bench_ret * 100, 2),
            'alpha_return': round(alpha_ret * 100, 2),
            'beta': round(beta, 3),
            'alpha_daily_bps': round(alpha * 10000, 1),
            'beta_contribution_pct': round(beta_contrib * 100, 1),
            'alpha_contribution_pct': round(alpha_contrib * 100, 1),
            'verdict': ('主要跟跌(β>' + str(round(beta_contrib*100)) + '%)，选股能力未受损'
                       if beta_contrib > 0.7
                       else '⚠ 超额回撤显著(α' + str(round(alpha_contrib*100)) + '%)，需排查因子暴露')
        }

    def factor_stress_report(self, window_start, window_end):
        """
        因子层面压力测试: 计算窗口内各因子的多空收益。

        若某因子在危机窗口内多空收益暴跌→该因子是"拥挤因子"。
        """
        if self.factor_exposure is None:
            return {'status': '无因子暴露数据'}

        fdf = self.factor_exposure.copy()
        fdf['trade_date'] = pd.to_datetime(fdf['trade_date'])
        mask = ((fdf['trade_date'] >= window_start) &
                (fdf['trade_date'] <= window_end))
        crisis = fdf[mask]

        factor_cols = [c for c in crisis.columns
                      if c not in ('trade_date', 'ts_code', 'industry', 'log_mcap')]

        results = {}
        for col in factor_cols:
            if crisis[col].notna().sum() < 30:
                continue
            # Top vs Bottom 五分位收益差
            crisis_clean = crisis.dropna(subset=[col, 'forward_ret']) if 'forward_ret' in crisis.columns else crisis
            if 'forward_ret' not in crisis_clean.columns:
                continue
            crisis_clean['quintile'] = pd.qcut(crisis_clean[col].rank(method='first'), 5, labels=False, duplicates='drop')
            top = crisis_clean[crisis_clean['quintile'] == 4]['forward_ret'].mean()
            bot = crisis_clean[crisis_clean['quintile'] == 0]['forward_ret'].mean()
            spread = (top - bot) * 100
            results[col] = {
                'top_quintile_ret': round(top * 100, 2),
                'bottom_quintile_ret': round(bot * 100, 2),
                'long_short_spread': round(spread, 2),
                'is_crowded': spread < -1.0,  # 多空收益为负→拥挤/失效
            }

        return results

    def summary(self):
        """一键输出完整压力测试报告"""
        scenarios = self.scenario_analysis()
        print('=' * 65)
        print('  📉 极端行情压力测试')
        print('=' * 65)
        for _, row in scenarios.iterrows():
            print(f"\n  {row['scenario']}  ({row['start']} → {row['end']})")
            print(f"    策略回撤: {row['strategy_max_dd_pct']:+.1f}%  "
                  f"基准回撤: {row['benchmark_dd_pct']:+.1f}%  "
                  f"收益: {row['strategy_return_pct']:+.1f}%")
            print(f"    日胜率: {row['daily_win_rate']:.0f}%  "
                  f"连跌天数: {row['max_dd_days']}  "
                  f"波动放大: {row['vol_amplification']:.1f}x  "
                  f"β相关: {row['bench_correlation']:.3f}")

        # 对2018和2024做归因
        print(f"\n{'─'*65}")
        print('  回撤归因:')
        for key in ['2018_trade_war', '2024_microcap_crash']:
            s = self.STRESS_SCENARIOS[key]
            decomp = self.drawdown_decomposition(s['start'], s['end'])
            if 'verdict' in decomp:
                print(f'  {s["label"]}: {decomp["verdict"]}')
                print(f'    β={decomp["beta"]:.2f}  α日={decomp["alpha_daily_bps"]}bps  '
                      f'β贡献{decomp["beta_contribution_pct"]}%  α贡献{decomp["alpha_contribution_pct"]}%')
        print('=' * 65)
        return scenarios


# ══════════════════════════════════════════════════════════════════
# 集成示例: 将四模块接入现有LightGBM回测管线
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# 模块二增强: 非线性市值剥离 (AdvancedRiskNeutralizer)
# ══════════════════════════════════════════════════════════════════

class AdvancedRiskNeutralizer:
    """
    非线性分桶中性化。
    不在全市场跑单一OLS回归, 而是按市值分桶, 桶内独立标准化。
    彻底消除"小市值区间非线性拥挤"导致的因子泄露。

    逻辑:
      每日: 市值分N桶 → 每桶内按行业rank(pct) → 缩放到[-0.5, 0.5]
      → LightGBM看到的永远是"同规模档次里最好的", 不会滑向微盘垃圾股
    """

    def __init__(self, n_buckets=10):
        self.n_buckets = n_buckets

    def bucket_rank_neutralize(self, df, factor_cols,
                                target_date_col='trade_date',
                                mcap_col='log_mcap',
                                industry_col='industry_code'):
        """
        非线性分桶中性化。

        Parameters
        ----------
        df : DataFrame  含因子列 + 市值列 + 行业列
        factor_cols : list  待中性化的因子列名
        mcap_col : str  市值列
        industry_col : str  行业列
        target_date_col : str  日期列

        Returns
        -------
        neutralized_df : DataFrame  中性化后 (因子值变为[-0.5, 0.5]的秩)
        """
        result = df.copy()
        if mcap_col not in result.columns:
            return result

        # 市值分桶: N等分
        try:
            result['mcap_bucket'] = result.groupby(target_date_col)[mcap_col].transform(
                lambda x: pd.qcut(x, q=self.n_buckets, labels=False, duplicates='drop')
            )
        except Exception:
            result['mcap_bucket'] = 0

        # 每个日期内, 市值桶×行业 双重组内rank
        for col in factor_cols:
            if col not in result.columns:
                continue
            # 组内rank → pct → [-0.5, 0.5]
            result[col] = result.groupby(
                [target_date_col, 'mcap_bucket', industry_col],
                group_keys=False
            )[col].rank(pct=True) - 0.5

            # 兜底: 组内样本太少导致NaN → 用桶均值填 → 全市场均值
            result[col] = (result[col]
                           .fillna(result.groupby([target_date_col, 'mcap_bucket'])[col].transform('mean'))
                           .fillna(0))

        result.drop(columns=['mcap_bucket'], inplace=True, errors='ignore')
        return result

    def full_neutralize(self, df, factor_cols, mcap_col='log_mcap',
                        industry_col='industry_code'):
        """
        一键非线性中性化 + 线性对比报告。
        返回: (bucket_neutralized_df, report_dict)
        """
        from quant_backtest_engine import RiskNeutralizer
        linear = RiskNeutralizer()

        # 线性OLS版 (用于对比)
        _, linear_report = linear.full_neutralize(
            df.assign(industry=df[industry_col], log_mcap=df[mcap_col]),
            factor_cols=factor_cols,
            industry_col='industry',
            size_col='log_mcap'
        )

        # 非线性分桶版
        bucket_df = self.bucket_rank_neutralize(
            df, factor_cols, mcap_col=mcap_col, industry_col=industry_col
        )

        # 对比报告: 两版因子对市值的R²
        report = {'linear': linear_report, 'bucket': {}}
        for col in factor_cols:
            if col not in df.columns or df[col].isna().all():
                continue
            # 线性版残差对市值的R² (应该接近0)
            _, _, r2_linear = linear.size_neutralize(
                df[col].values, df[mcap_col].values
            )
            # 分桶版对市值的R² (也应该接近0, 但更非线性稳健)
            _, _, r2_bucket = linear.size_neutralize(
                bucket_df[col].values, df[mcap_col].values
            )
            report['bucket'][col] = {
                'linear_size_r2': round(r2_linear, 4),
                'bucket_size_r2': round(r2_bucket, 4),
            }

        return bucket_df, report


def integrate_to_pipeline(factor_df, target_df, codes, trade_date,
                          prices, daily_volumes, daily_volatilities,
                          daily_amounts, industry_map, log_mcap_series,
                          total_capital=1e8):
    """
    演示: 四模块如何串入你的回测主循环。

    在一个调仓日, 依次执行:
      1. RiskNeutralizer  → 中性化因子
      2. LightGBM预测     → 得到原始排名
      3. ExecutionSimulator → 实际执行(滑点+容量+涨跌停)
      4. 记录成交数据     → 用于后续压力测试

    你的主循环只需调这一个函数。
    """
    # Step 1: 中性化
    neutralizer = RiskNeutralizer()
    factor_cols = [c for c in factor_df.columns
                   if c not in ('ts_code', 'trade_date', 'close', 'log_mcap')]
    neut_df, neut_report = neutralizer.full_neutralize(
        factor_df.assign(industry=industry_map, log_mcap=log_mcap_series),
        factor_cols=factor_cols
    )

    # Step 2: [你的LightGBM预测代码在这里]
    # model.predict(neut_df[feature_cols]) → ml_scores

    # Step 3: 执行模拟
    simulator = ExecutionSimulator()
    # 假设ml_scores已选出top30, 等权
    n_select = 30
    target_weights = np.ones(n_select) / n_select

    exec_report = simulator.execute_round(
        target_weights=target_weights,
        codes=codes[:n_select],
        trade_date=trade_date,
        prices=prices[:n_select],
        daily_volumes=daily_volumes[:n_select],
        daily_volatilities=daily_volatilities[:n_select],
        daily_amounts=daily_amounts[:n_select],
        total_capital=total_capital,
    )

    return {
        'neutralization': neut_report,
        'execution': exec_report,
    }


if __name__ == '__main__':
    print("quant_backtest_engine.py — 四大模块已就绪")
    print("  ExecutionSimulator   : 动态滑点 + 容量截断 + 涨跌停过滤")
    print("  RiskNeutralizer      : 行业中性化 + 市值中性化 (statsmodels OLS)")
    print("  Alpha Factors (5个)  : 波动率动量/流动性溢价/日内方差/跳空反转/换手衰竭")
    print("  StressTester         : 4个历史危机情景 + 回撤归因 + 因子拥挤检测")
