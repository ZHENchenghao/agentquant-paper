# -*- coding: utf-8 -*-
"""
Agent 3: SOX→A股AI传导Agent
SOX数据 → A股AI(电子+计算机+通信) → 领先滞后 → Granger因果 → 传导信号
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from base_agent import BaseAgent
from shared.db import get_conn
from sox_conduction.lead_lag import CrossMarketConduction


class SoxConductionAgent(BaseAgent):
    """SOX→A股AI传导Agent

    每日执行:
    1. 加载SOX日线数据 (global_index_daily)
    2. 构建A股AI综合指数 (电子+计算机+通信行业proxy)
    3. 领先滞后分析 (交叉相关 + Granger因果)
    4. 滚动传导强度检测
    5. SOX极端事件窗口分析
    """

    # A股AI相关申万行业
    AI_INDUSTRIES = ['电子', '计算机', '通信']

    def __init__(self):
        super().__init__(
            name='SoxConduction',
            description='SOX→A股AI传导: 领先滞后 → Granger因果 → 传导强度 → 事件窗口'
        )
        self.conduction = CrossMarketConduction(max_lag=10, significance=0.05)
        self.sox_data = None
        self.ai_composite = None
        self.merged_returns = None

    def validate_data(self):
        """验证SOX和A股AI数据可用"""
        conn = get_conn()
        try:
            # 1. SOX数据
            self.sox_data = conn.execute("""
                SELECT trade_date, close
                FROM global_index_daily
                WHERE index_code = 'SOX'
                  AND trade_date >= DATE '2021-01-01'
                ORDER BY trade_date
            """).df()

            # 2. A股AI行业数据 (proxy_industry_daily)
            ind_list = "', '".join(self.AI_INDUSTRIES)
            ai_industries = conn.execute(f"""
                SELECT trade_date, industry, close
                FROM proxy_industry_daily
                WHERE industry IN ('{ind_list}')
                  AND trade_date >= DATE '2021-01-01'
                ORDER BY trade_date
            """).df()

            if ai_industries.empty:
                self.data_status['ai_industries'] = 'empty in proxy_industry_daily'
                return False

            # 3. 构建A股AI综合指数 (等权平均每日收益)
            ai_pivot = ai_industries.pivot(
                index='trade_date', columns='industry', values='close'
            )
            # 日收益率 → 等权平均 → 累计净值
            ai_returns = ai_pivot.pct_change().mean(axis=1)
            ai_composite = (1 + ai_returns).cumprod()

            self.ai_composite = pd.DataFrame({
                'trade_date': ai_composite.index,
                'close': ai_composite.values,
            })

            self.data_status = {
                'sox': f'{len(self.sox_data)}行, {self.sox_data["trade_date"].min()} ~ {self.sox_data["trade_date"].max()}',
                'ai_composite': f'{len(self.ai_composite)}行, {self.ai_composite["trade_date"].min()} ~ {self.ai_composite["trade_date"].max()}',
                'ai_industries': self.AI_INDUSTRIES,
            }

            print(f"    SOX: {len(self.sox_data)}行 | "
                  f"AI综合: {len(self.ai_composite)}行 | "
                  f"行业: {self.AI_INDUSTRIES}")

            return len(self.sox_data) > 100 and len(self.ai_composite) > 100

        except Exception as e:
            self.data_status['error'] = str(e)
            return False
        finally:
            conn.close()

    def analyze(self):
        """核心分析"""
        # 1. 对齐收益率
        print("    对齐SOX和A股AI收益率...")
        self.merged_returns = self.conduction.align_returns(
            self.sox_data, self.ai_composite
        )
        print(f"    对齐后: {len(self.merged_returns)}个交易日")

        # 2. 交叉相关分析
        print("    交叉相关分析...")
        ccf_result = self.conduction.cross_correlation(self.merged_returns)

        # 3. Granger因果检验
        print("    Granger因果检验...")
        granger_result = self.conduction.granger_causality(self.merged_returns)

        # 4. 滚动相关性
        print("    滚动传导强度...")
        rolling_corr = self.conduction.rolling_correlation(self.merged_returns, window=60)

        # 5. 传导异常检测
        conduction_state = self.conduction.detect_conduction_anomaly(rolling_corr)

        # 6. SOX极端事件检测(最近60天)
        print("    SOX极端事件窗口分析...")
        recent = self.merged_returns.tail(120)  # 120天够找极端事件
        sox_vol = recent['ret_source'].std()
        extreme_dates = recent[
            abs(recent['ret_source']) > 2 * sox_vol
        ].index.tolist()

        event_analysis = {}
        if extreme_dates:
            event_analysis = self.conduction.event_window_analysis(
                self.merged_returns,
                extreme_dates[-10:],  # 最近10个极端事件
                window=(-3, 5)
            )

        # 7. 构建传导信号
        signal = self._build_signal(ccf_result, granger_result, conduction_state)

        # 8. 近期SOX走势
        sox_recent = self.sox_data.tail(5)
        ai_recent = self.ai_composite.tail(5)

        return {
            'ccf': ccf_result,
            'granger': granger_result,
            'conduction_state': conduction_state,
            'event_analysis': event_analysis,
            'signal': signal,
            'recent_performance': {
                'sox_5d': round(float(
                    (sox_recent['close'].iloc[-1] / sox_recent['close'].iloc[0] - 1) * 100
                    if len(sox_recent) >= 2 else 0
                ), 2),
                'ai_5d': round(float(
                    (ai_recent['close'].iloc[-1] / ai_recent['close'].iloc[0] - 1) * 100
                    if len(ai_recent) >= 2 else 0
                ), 2),
                'sox_latest': round(float(sox_recent['close'].iloc[-1]), 2),
                'latest_date': str(self.sox_data['trade_date'].max()),
            },
            'n_aligned_days': len(self.merged_returns),
        }

    def _build_signal(self, ccf_result, granger_result, conduction_state):
        """综合所有分析构建传导信号"""
        signals = []
        weight = 0

        # Granger因果得分
        if granger_result.get('is_significant'):
            weight += 1
            signals.append(
                f"SOX→A股AI Granger因果显著 "
                f"(lag={granger_result['best_lag']}, p={granger_result['best_p_value']})"
            )

        # 交叉相关得分
        best_lag = ccf_result.get('best_lag', 0)
        best_corr = ccf_result.get('best_correlation', 0)
        if best_lag > 0 and abs(best_corr) > 0.2:
            weight += 1
            signals.append(f"SOX领先A股AI {best_lag}天, 相关性{best_corr:.3f}")

        # 传导状态得分
        state = conduction_state.get('conduction_state', 'unknown')
        if state == 'strong':
            weight += 1
            signals.append(f"当前传导强度: 强 (r={conduction_state['current_correlation']:.3f})")
        elif state == 'broken':
            weight -= 1
            signals.append(f"⚠ 传导断裂 (r={conduction_state['current_correlation']:.3f})")

        # 传导异常
        if conduction_state.get('anomaly'):
            anomaly_type = conduction_state.get('anomaly_type', '')
            if anomaly_type == 'conduction_accelerating':
                signals.append('⚠ 传导加速→可能有重大事件驱动')
            elif anomaly_type == 'conduction_breaking':
                weight -= 1
                signals.append('⚠ 传导断裂→SOX对A股AI失去指引作用')

        # 综合判定
        if weight >= 2:
            action = 'FOLLOW_SOX'
            action_desc = 'SOX对A股AI有显著传导，可将SOX作为A股AI方向参考'
        elif weight >= 1:
            action = 'REFERENCE'
            action_desc = 'SOX对A股AI有一定传导，可作为辅助参考'
        else:
            action = 'INDEPENDENT'
            action_desc = 'SOX对A股AI传导弱/断裂，A股AI独立运行'

        return {
            'action': action,
            'action_desc': action_desc,
            'weight': weight,
            'signals': signals,
        }

    def report(self, result):
        """生成报告"""
        alerts = []

        signal = result.get('signal', {})
        conduction_state = result.get('conduction_state', {})

        # 传导断裂告警
        if conduction_state.get('conduction_state') == 'broken':
            alerts.append(
                f"[SOX传导] 传导断裂: 相关性{conduction_state['current_correlation']:.3f}→"
                f"A股AI独立运行，不跟SOX"
            )

        # 传导加速告警
        if conduction_state.get('anomaly_type') == 'conduction_accelerating':
            alerts.append(
                f"[SOX传导] 传导加速: z={conduction_state['z_score']:.1f}→"
                f"关注是否有利空/利好通过SOX传导至A股AI"
            )

        # SOX最近走势
        perf = result.get('recent_performance', {})
        if abs(perf.get('sox_5d', 0)) > 5:
            direction = '涨' if perf['sox_5d'] > 0 else '跌'
            alerts.append(
                f"[SOX] 5日{direction}{abs(perf['sox_5d']):.1f}%→"
                f"根据传导关系(lag={result.get('ccf',{}).get('best_lag','?')}), "
                f"预计A股AI将在{result.get('ccf',{}).get('best_lag','?')}天后反应"
            )

        # Granger结果
        granger = result.get('granger', {})
        granger_sig = '显著' if granger.get('is_significant') else '不显著'

        ccf = result.get('ccf', {})
        summary = (
            f"传导: {signal.get('action','N/A')} | "
            f"Granger: {granger_sig}(lag={granger.get('best_lag','?')}) | "
            f"CCF: lag={ccf.get('best_lag','?')}, r={ccf.get('best_correlation',0):.3f} | "
            f"传导状态: {conduction_state.get('conduction_state','?')} | "
            f"SOX 5d: {perf.get('sox_5d',0):+.1f}%"
        )

        return {
            'status': 'ok',
            'summary': summary,
            'signal': signal,
            'ccf': {k: v for k, v in ccf.items() if k != 'ccf'},
            'granger_summary': {
                'is_significant': granger.get('is_significant'),
                'best_lag': granger.get('best_lag'),
                'best_p_value': granger.get('best_p_value'),
            },
            'conduction_state': conduction_state,
            'event_analysis': result.get('event_analysis', {}),
            'recent_performance': perf,
            'alerts': alerts,
            'data_status': self.data_status,
        }


if __name__ == '__main__':
    agent = SoxConductionAgent()
    result = agent.run()
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
