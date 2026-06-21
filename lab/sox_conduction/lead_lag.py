# -*- coding: utf-8 -*-
"""
跨市场领先滞后分析引擎
核心能力: 交叉相关 → Granger因果 → 滚动相关性 → 事件窗口检测 → 传导信号
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


class CrossMarketConduction:
    """跨市场传导检测器

    检测SOX(费城半导体指数) → A股AI(电子+计算机+通信)的领先滞后关系
    输出: 传导方向、领先天数、传导强度、当前传导状态
    """

    def __init__(self, max_lag=10, significance=0.05):
        self.max_lag = max_lag
        self.significance = significance

    def align_returns(self, source_df, target_df):
        """
        对齐两个市场的日收益率序列

        Args:
            source_df: DataFrame with [trade_date, close] for SOX
            target_df: DataFrame with [trade_date, close] for A-share AI

        Returns:
            DataFrame with [trade_date, source_ret, target_ret]
        """
        source = source_df.copy()
        target = target_df.copy()

        source['trade_date'] = pd.to_datetime(source['trade_date'])
        target['trade_date'] = pd.to_datetime(target['trade_date'])

        source = source.sort_values('trade_date')
        target = target.sort_values('trade_date')

        source['ret'] = source['close'].pct_change()
        target['ret'] = target['close'].pct_change()

        # 对齐日期
        merged = source[['trade_date', 'ret']].merge(
            target[['trade_date', 'ret']],
            on='trade_date',
            suffixes=('_source', '_target')
        ).dropna()

        return merged

    def cross_correlation(self, merged_returns, max_lag=None):
        """
        计算交叉相关函数 (CCF)

        Returns:
            ccf: dict mapping lag to correlation
            best_lag: 领先天数 (正=source领先target, 负=target领先source)
        """
        if max_lag is None:
            max_lag = self.max_lag

        source = merged_returns['ret_source'].values
        target = merged_returns['ret_target'].values

        ccf = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(source[:lag], target[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(source[lag:], target[:-lag])[0, 1]
            else:
                corr = np.corrcoef(source, target)[0, 1]
            ccf[lag] = corr if not np.isnan(corr) else 0

        # 最佳领先滞后
        best_lag = max(ccf, key=lambda k: abs(ccf[k]))
        best_corr = ccf[best_lag]

        return {
            'ccf': {k: round(v, 4) for k, v in ccf.items()},
            'best_lag': best_lag,
            'best_correlation': round(best_corr, 4),
            'interpretation': (
                f'SOX领先A股AI {best_lag}天, 相关性{best_corr:.3f}'
                if best_lag > 0 else
                f'A股AI领先SOX {-best_lag}天, 相关性{best_corr:.3f}'
                if best_lag < 0 else
                f'同步, 相关性{best_corr:.3f}'
            ),
        }

    def granger_causality(self, merged_returns, max_lag=None):
        """
        Granger因果检验: SOX是否Granger-cause A股AI

        Returns:
            {best_lag, p_value, is_significant, test_results}
        """
        if max_lag is None:
            max_lag = min(self.max_lag, len(merged_returns) // 20)

        if max_lag < 2:
            return {'error': '数据不足'}

        data = merged_returns[['ret_source', 'ret_target']].dropna()

        # 平稳性检验
        adf_s = adfuller(data['ret_source'].dropna())[1]
        adf_t = adfuller(data['ret_target'].dropna())[1]

        if adf_s > 0.1 or adf_t > 0.1:
            return {
                'error': '序列非平稳',
                'adf_source_p': round(adf_s, 4),
                'adf_target_p': round(adf_t, 4),
            }

        try:
            gc_result = grangercausalitytests(
                data[['ret_target', 'ret_source']],  # target ~ source
                maxlag=max_lag, verbose=False
            )
        except Exception as e:
            return {'error': f'Granger检验失败: {str(e)}'}

        # 找最小p值的最佳滞后阶数
        best_lag = None
        best_p = 1.0
        test_results = []

        for lag, tests in gc_result.items():
            p_value = tests[0]['ssr_ftest'][1]  # F-test p-value
            test_results.append({
                'lag': lag,
                'p_value': round(p_value, 4),
                'f_stat': round(tests[0]['ssr_ftest'][0], 2),
                'significant': p_value < self.significance,
            })
            if p_value < best_p:
                best_p = p_value
                best_lag = lag

        return {
            'status': 'ok',
            'best_lag': best_lag,
            'best_p_value': round(best_p, 4),
            'is_significant': best_p < self.significance,
            'significance_level': self.significance,
            'interpretation': (
                f'SOX Granger-causes A股AI (lag={best_lag}, p={best_p:.4f})'
                if best_p < self.significance else
                f'SOX对A股AI无显著Granger因果关系 (best p={best_p:.4f})'
            ),
            'test_results': test_results,
            'adf_source_p': round(adf_s, 4),
            'adf_target_p': round(adf_t, 4),
        }

    def rolling_correlation(self, merged_returns, window=60):
        """
        滚动相关性分析: 检测传导强度是否在变化

        Returns:
            DataFrame with [date, correlation, z_score]
        """
        merged = merged_returns.copy()
        merged['trade_date'] = pd.to_datetime(merged['trade_date'])

        # 滚动相关性 (source[t-1] vs target[t] → 领先1天的相关)
        merged['source_lag1'] = merged['ret_source'].shift(1)
        merged = merged.dropna(subset=['source_lag1', 'ret_target'])

        rolling_corr = merged['source_lag1'].rolling(window).corr(merged['ret_target'])

        result = pd.DataFrame({
            'trade_date': merged['trade_date'],
            'rolling_corr': rolling_corr.values,
        }).dropna()

        # Z-score (传导加速/减速)
        corr_mean = result['rolling_corr'].mean()
        corr_std = result['rolling_corr'].std()
        if corr_std > 0:
            result['z_score'] = (result['rolling_corr'] - corr_mean) / corr_std

        return result

    def detect_conduction_anomaly(self, rolling_corr_df, threshold=2.0):
        """
        检测传导异常:
        - 传导断裂: 相关性突然降至接近0或负值
        - 传导加速: 相关性急剧上升→可能是事件驱动传导
        """
        if rolling_corr_df.empty or len(rolling_corr_df) < 5:
            return {}

        latest = rolling_corr_df.tail(5)
        current_corr = latest['rolling_corr'].iloc[-1]
        z_score = latest['z_score'].iloc[-1] if 'z_score' in latest.columns else 0

        # 传导状态判定
        if current_corr > 0.3:
            conduction_state = 'strong'
        elif current_corr > 0.1:
            conduction_state = 'normal'
        elif current_corr > -0.1:
            conduction_state = 'weak'
        else:
            conduction_state = 'broken'

        return {
            'current_correlation': round(float(current_corr), 4),
            'z_score': round(float(z_score), 2),
            'conduction_state': conduction_state,
            'anomaly': abs(z_score) > threshold,
            'anomaly_type': (
                'conduction_accelerating' if z_score > threshold else
                'conduction_breaking' if z_score < -threshold else
                'none'
            ),
        }

    def event_window_analysis(self, merged_returns, event_dates, window=(-3, 5)):
        """
        事件窗口分析: SOX大事件 → A股AI反应

        Args:
            merged_returns: 对齐的收益率
            event_dates: list of event dates (SOX暴涨/暴跌日)
            window: (pre_days, post_days)

        Returns:
            event_analysis: 事件前后A股AI平均反应
        """
        if not event_dates:
            return {}

        merged = merged_returns.copy()
        merged['trade_date'] = pd.to_datetime(merged['trade_date'])
        merged = merged.set_index('trade_date')

        pre, post = window
        all_responses = []

        for event_date in event_dates:
            event_date = pd.to_datetime(event_date)
            if event_date not in merged.index:
                continue

            idx = merged.index.get_loc(event_date)
            start = max(0, idx + pre)
            end = min(len(merged), idx + post + 1)

            if end - start < post - pre + 1:
                continue

            window_data = merged.iloc[start:end]
            # 累计A股AI收益
            cum_ret = (1 + window_data['ret_target']).prod() - 1

            all_responses.append({
                'event_date': str(event_date.date()),
                'sox_ret': float(window_data['ret_source'].iloc[-pre]),
                'a_share_cum_ret': float(cum_ret),
                'max_reaction': float(window_data['ret_target'].max()),
                'min_reaction': float(window_data['ret_target'].min()),
            })

        if not all_responses:
            return {'status': 'no_matching_events'}

        responses_df = pd.DataFrame(all_responses)
        return {
            'status': 'ok',
            'n_events': len(all_responses),
            'avg_cum_ret': round(float(responses_df['a_share_cum_ret'].mean()), 4),
            'positive_ratio': round(float((responses_df['a_share_cum_ret'] > 0).mean()), 2),
            'max_cum_ret': round(float(responses_df['a_share_cum_ret'].max()), 4),
            'min_cum_ret': round(float(responses_df['a_share_cum_ret'].min()), 4),
            'recent_events': all_responses[-5:],
        }
