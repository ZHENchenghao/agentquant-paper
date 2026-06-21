# -*- coding: utf-8 -*-
"""
QuantLab 主编排器 v1.0
并行启动三个Agent → 交叉验证 → 统一日报

执行: python orchestrator.py
"""
import sys
import io
import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from factor_flip.agent import FactorFlipAgent
from northbound_decomp.agent import NorthboundDecompAgent
from sox_conduction.agent import SoxConductionAgent
from shared.report import LabReport


class QuantLabOrchestrator:
    """QuantLab主编排器

    三个Agent并行执行, 汇总后进行交叉验证:
    - 因子翻牌bearish + 北向流出 → 共振下跌信号
    - SOX传导断裂 + 因子翻牌 → 结构性变化确认
    - 北向流入 + SOX传导强 → AI方向确认
    """

    def __init__(self):
        self.agents = {
            'FactorFlip': FactorFlipAgent(),
            'NorthboundDecomp': NorthboundDecompAgent(),
            'SoxConduction': SoxConductionAgent(),
        }
        self.results = {}
        self.report = LabReport("QuantLab 日报")

    def run_all(self, parallel=True):
        """运行所有Agent"""
        print(f"\n{'='*70}")
        print(f"  QuantLab v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  三Agent并行量化研究实验室")
        print(f"{'='*70}")

        start_time = time.time()

        if parallel:
            self._run_parallel()
        else:
            self._run_sequential()

        elapsed = time.time() - start_time
        print(f"\n  全部分析完成 ({elapsed:.1f}s)")

        # 交叉验证
        cross_signals = self._cross_validate()
        self.report.set_meta('cross_validation', cross_signals)
        self.report.set_meta('total_elapsed_seconds', round(elapsed, 1))

        # 汇总
        for name, result in self.results.items():
            self.report.add_agent_result(name, result)

        # 添加交叉告警
        for signal in cross_signals.get('alerts', []):
            self.report.alerts.append(signal)

        return self.report

    def _run_parallel(self):
        """并行执行三个Agent"""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(agent.run): name
                for name, agent in self.agents.items()
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    self.results[name] = result
                except Exception as e:
                    print(f"  [{name}] ❌ 异常: {e}")
                    self.results[name] = {
                        'agent_name': name,
                        'status': 'error',
                        'summary': f'执行异常: {str(e)}',
                        'alerts': [],
                    }

    def _run_sequential(self):
        """串行执行(调试用)"""
        for name, agent in self.agents.items():
            self.results[name] = agent.run()

    def _cross_validate(self):
        """
        交叉验证: 三Agent信号互相印证

        验证逻辑:
        1. 因子翻牌bearish + 北向流出 = 共振下跌 (高置信度)
        2. 因子翻牌bullish + 北向流入 = 共振上涨 (高置信度)
        3. 因子翻牌 + 传导断裂 = 结构性变化 (市场定价逻辑切换)
        4. 北向流入 + SOX传导强 = AI方向确认
        """
        alerts = []
        signals = []

        ff = self.results.get('FactorFlip', {})
        nb = self.results.get('NorthboundDecomp', {})
        sox = self.results.get('SoxConduction', {})

        # 跳过失败Agent
        if ff.get('status') != 'ok' or nb.get('status') != 'ok':
            signals.append({'type': 'cross_validation', 'status': 'partial',
                           'note': '部分Agent异常，交叉验证不完全'})
            return {'signals': signals, 'alerts': alerts}

        # 信号1: 因子 + 北向 共振
        ff_signal = ff.get('signal', {})
        ff_action = ff_signal.get('action', 'NORMAL')

        nb_market = nb.get('market_flow', {})
        nb_div = nb.get('divergence', {})
        nb_consensus = nb_div.get('consensus', '')
        nb_direction = nb_market.get('direction', 'neutral')

        # 因子翻牌做空 + 北向流出
        if ff_action in ('REDUCE', 'CAUTION') and nb_consensus == '三方共振做空→强烈看空':
            alerts.append(
                '🔴 [交叉验证] 因子集体向空翻牌 + 北向/融资/主力三方做空 → 强烈减仓信号'
            )
            signals.append({
                'type': '共振下跌',
                'confidence': 'high',
                'components': ['FactorFlip:REDUCE', 'Northbound:三方做空'],
                'suggested_action': '大幅降仓(50%+), 切换防御',
            })

        # 因子翻牌做多 + 北向流入
        elif ff_action == 'NORMAL' and nb_consensus == '三方共振做多→强烈看多':
            if ff_signal.get('reason', '').startswith('因子集体向多'):
                alerts.append(
                    '🟢 [交叉验证] 因子集体向多翻牌 + 三方共振做多 → 强烈做多信号'
                )
                signals.append({
                    'type': '共振上涨',
                    'confidence': 'high',
                    'components': ['FactorFlip:BULLISH', 'Northbound:三方做多'],
                    'suggested_action': '可加仓进攻方向',
                })

        # 信号2: 因子翻牌 + SOX传导断裂 = 结构性变化
        sox_state = sox.get('conduction_state', {})
        if ff_action in ('REDUCE', 'CAUTION') and sox_state.get('conduction_state') == 'broken':
            alerts.append(
                '🟡 [交叉验证] 因子翻牌 + SOX→A股AI传导断裂 → 市场定价逻辑可能发生结构性变化'
            )
            signals.append({
                'type': '结构性变化',
                'confidence': 'medium',
                'components': ['FactorFlip:bearish', 'SOX:传导断裂'],
                'suggested_action': '观望，等待新因子结构形成',
            })

        # 信号3: 北向流入 + SOX传导强 = AI方向确认
        nb_5d = nb.get('raw_summary', {}).get('nb_total_5d', 0)
        sox_signal = sox.get('signal', {})

        if nb_5d > 50 and sox_signal.get('action') in ('FOLLOW_SOX', 'REFERENCE'):
            alerts.append(
                f'🟢 [交叉验证] 北向5日流入{nb_5d:.0f}亿 + SOX传导{sox_signal["action"]} → AI方向获双重确认'
            )
            signals.append({
                'type': 'AI方向确认',
                'confidence': 'medium',
                'components': ['Northbound:inflow', f'SOX:{sox_signal["action"]}'],
                'suggested_action': 'AI相关方向可适度参与',
            })

        # 信号4: 单Agent高置信度信号
        if ff_action == 'REDUCE' and ff_signal.get('confidence') == 'high':
            alerts.append(
                f'🔴 [因子翻牌] 高置信度做空: {ff_signal["reason"]}'
            )

        if nb_div.get('divergence_signals'):
            for sig in nb_div['divergence_signals']:
                if sig.get('confidence') == 'high':
                    alerts.append(
                        f'🟡 [北向] 高置信度分歧: {sig["signal"]}: {sig["interpretation"]}'
                    )

        if not signals:
            signals.append({
                'type': '无交叉信号',
                'confidence': 'low',
                'note': '三Agent未形成有效交叉验证，各自信号独立',
            })

        return {'signals': signals, 'alerts': alerts, 'n_cross_signals': len(signals)}


def main():
    orch = QuantLabOrchestrator()
    report = orch.run_all(parallel=True)

    # 终端输出
    report.print_summary()

    # 保存报告
    path = report.save()
    print(f"  报告已保存: {path}")

    # 输出交叉验证信号
    cross = report.meta.get('cross_validation', {})
    if cross.get('signals'):
        print(f"\n  交叉验证信号 ({cross.get('n_cross_signals', 0)}条):")
        for s in cross['signals']:
            print(f"    [{s.get('confidence','?')}] {s.get('type','?')}: {s.get('suggested_action','?')}")

    return report


if __name__ == '__main__':
    report = main()
