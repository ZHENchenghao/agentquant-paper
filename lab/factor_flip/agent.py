# -*- coding: utf-8 -*-
"""
Agent 1: 因子翻牌检测Agent
监控全A股因子IC方向变化, 检测集体翻牌 → 触发仓位调整建议
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from base_agent import BaseAgent
from shared.db import get_conn, latest_trade_date
from shared.factors import load_factors, get_factor_columns
from factor_flip.ic_monitor import ICMonitor


class FactorFlipAgent(BaseAgent):
    """因子翻牌检测Agent

    每日执行:
    1. 加载因子数据 → 识别有效因子列
    2. 计算滚动12月IC序列
    3. 检测单因子翻牌 + 集体翻牌
    4. 评估IC稳定性
    5. 生成仓位信号
    """

    def __init__(self):
        super().__init__(
            name='FactorFlip',
            description='因子IC翻牌监控: 滚动IC → 方向翻转 → 集体翻牌 → 仓位信号'
        )
        self.monitor = ICMonitor(
            window_months=12,
            flip_threshold=0.3,
            collective_threshold=0.25,
            min_obs=30,
        )
        self.factor_df = None
        self.factor_cols = []
        self.ic_pivot = None
        self.flips = None
        self.collective_flips = None
        self.stability = None

    def validate_data(self):
        """验证因子数据可用"""
        try:
            # 检查因子缓存
            if not os.path.exists('D:/AgentQuant/our/cache/factors_2002.parquet'):
                self.data_status['factor_cache'] = 'missing'
                return False

            # 加载因子数据 (10年, 月频IC)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')

            raw_factors = load_factors(start_date=start_date, end_date=end_date)
            raw_factors['trade_date'] = pd.to_datetime(raw_factors['trade_date'])

            # 加载target_60d (含excess_ret)
            target = pd.read_parquet('D:/AgentQuant/our/cache/target_60d.parquet')
            target['trade_date'] = pd.to_datetime(target['trade_date'])

            # 因子数据按月取最后一天 (月频对齐)
            raw_factors['ym'] = raw_factors['trade_date'].dt.to_period('M')
            monthly = raw_factors.sort_values('trade_date').groupby(
                ['ts_code', 'ym']
            ).last().reset_index()

            target['ym'] = target['trade_date'].dt.to_period('M')

            # 用 ym + ts_code 合并 (比日期字符串更可靠)
            self.factor_df = monthly.merge(
                target[['ts_code', 'ym', 'excess_ret']],
                on=['ts_code', 'ym'],
                how='inner'
            )
            self.factor_df['trade_date'] = self.factor_df['trade_date'].dt.strftime('%Y-%m-%d')

            self.factor_cols = get_factor_columns(self.factor_df)

            self.data_status = {
                'factor_cache': 'ok',
                'n_factors': len(self.factor_cols),
                'n_rows': len(self.factor_df),
                'date_range': f"{self.factor_df['trade_date'].min()} ~ {self.factor_df['trade_date'].max()}",
                'n_stocks': self.factor_df['ts_code'].nunique(),
                'frequency': 'monthly',
            }

            print(f"    因子数据(月频): {len(self.factor_cols)}因子, "
                  f"{self.factor_df['ts_code'].nunique()}只股票, "
                  f"{len(self.factor_df)}行")

            return len(self.factor_df) > 10000 and len(self.factor_cols) > 5

        except Exception as e:
            self.data_status['error'] = str(e)
            return False

    def analyze(self):
        """核心分析"""
        print(f"    计算 {len(self.factor_cols)} 个因子的月度滚动IC...")

        # 只取有excess_ret的数据
        df = self.factor_df.dropna(subset=['excess_ret']).copy()
        df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 1. 计算滚动IC序列
        self.ic_pivot = self.monitor.compute_factor_ic_series(
            df, self.factor_cols
        )

        if self.ic_pivot.empty:
            return {'error': 'IC计算失败，数据不足'}

        print(f"    IC序列: {len(self.ic_pivot)}期 x {len(self.ic_pivot.columns)}因子")

        # 2. 检测单因子翻牌
        self.flips = self.monitor.detect_flips(self.ic_pivot)
        print(f"    翻牌检测: {len(self.flips)}次 (最近12期)")

        # 3. 检测集体翻牌
        self.collective_flips = self.monitor.detect_collective_flip(
            self.flips, self.ic_pivot
        )
        print(f"    集体翻牌: {len(self.collective_flips)}次")

        # 4. IC稳定性
        self.stability = self.monitor.compute_ic_stability(self.ic_pivot)

        # 5. 生成信号
        signal = self.monitor.generate_signal(
            self.collective_flips, self.stability,
            self.ic_pivot.tail(1) if not self.ic_pivot.empty else None
        )

        # 6. 提取最新一期IC全景
        latest_ic = {}
        if not self.ic_pivot.empty:
            last_row = self.ic_pivot.tail(1)
            for col in last_row.columns:
                val = last_row[col].iloc[0]
                if not pd.isna(val):
                    latest_ic[col] = round(float(val), 4)

        return {
            'signal': signal,
            'latest_ic': latest_ic,
            'n_factors_tracked': len(self.factor_cols),
            'n_ic_periods': len(self.ic_pivot),
            'recent_flips': self.flips.tail(20).to_dict('records') if len(self.flips) > 0 else [],
            'collective_flips': self.collective_flips,
            'stability_low': [
                {'factor': k, **v}
                for k, v in sorted(self.stability.items(),
                                   key=lambda x: x[1].get('stability_score', 1))[:10]
            ],
            'stability_high': [
                {'factor': k, **v}
                for k, v in sorted(self.stability.items(),
                                   key=lambda x: x[1].get('stability_score', 1),
                                   reverse=True)[:5]
            ],
        }

    def report(self, result):
        """生成报告"""
        if 'error' in result:
            return {
                'status': 'error',
                'summary': result['error'],
                'alerts': [],
            }

        signal = result['signal']
        alerts = []

        if signal['action'] in ('REDUCE', 'CAUTION'):
            alerts.append(
                f"[因子翻牌] {signal['action']}: {signal['reason']} "
                f"→ {signal['suggested_action']}"
            )

        # IC方向汇总
        latest_ic = result.get('latest_ic', {})
        pos_count = sum(1 for v in latest_ic.values() if v > 0)
        neg_count = sum(1 for v in latest_ic.values() if v < 0)
        total = len(latest_ic)

        # 低稳定性因子告警
        low_stab = result.get('stability_low', [])
        if low_stab and low_stab[0].get('stability_score', 1) < 0.3:
            alerts.append(
                f"[IC稳定性] {len(low_stab)}个因子稳定性<0.3: "
                f"{', '.join(f['factor'][:15] for f in low_stab[:5])}"
            )

        recent_cf = result.get('collective_flips', [])
        cf_recent = [cf for cf in recent_cf
                     if cf['period'] == self.ic_pivot.index[-1]] if len(recent_cf) > 0 and not self.ic_pivot.empty else []

        summary = (
            f"信号: {signal['action']} | "
            f"跟踪{result['n_factors_tracked']}因子 | "
            f"IC正向{pos_count}/负向{neg_count}/{total} | "
            f"翻牌{len(result['recent_flips'])}次 | "
            f"集体翻牌{len(result['collective_flips'])}次"
        )

        return {
            'status': 'ok',
            'summary': summary,
            'signal': signal,
            'ic_direction': {
                'positive': pos_count,
                'negative': neg_count,
                'total': total,
                'top_positive': sorted(latest_ic.items(), key=lambda x: x[1], reverse=True)[:5],
                'top_negative': sorted(latest_ic.items(), key=lambda x: x[1])[:5],
            },
            'stability': {
                'worst_10': result.get('stability_low', []),
                'best_5': result.get('stability_high', []),
            },
            'collective_flip_detail': cf_recent,
            'recent_flips_sample': result['recent_flips'][:10],
            'data_status': self.data_status,
            'alerts': alerts,
        }


if __name__ == '__main__':
    agent = FactorFlipAgent()
    result = agent.run()
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
