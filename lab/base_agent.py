# -*- coding: utf-8 -*-
"""
QuantLab Agent基类
所有Agent继承此类，获得统一的生命周期管理
"""
import sys
import io
import time
import traceback
from datetime import datetime
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """量化Agent基类

    生命周期: init → validate_data() → analyze() → report()
    每个Agent独立运行，不依赖其他Agent的输出
    """

    def __init__(self, name, description=""):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.errors = []
        self.data_status = {}

    def run(self):
        """标准执行流程"""
        self.start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"  [{self.name}] 启动...")
        print(f"{'='*60}")

        try:
            # Step 1: 数据验证
            print(f"  [{self.name}] Step 1/3: 数据验证...")
            data_ok = self.validate_data()
            if not data_ok:
                print(f"  [{self.name}] ⚠ 数据不足，生成降级报告")
                return self._degraded_report()

            # Step 2: 核心分析
            print(f"  [{self.name}] Step 2/3: 核心分析...")
            result = self.analyze()

            # Step 3: 生成报告
            print(f"  [{self.name}] Step 3/3: 生成报告...")
            report = self.report(result)

            self.end_time = datetime.now()
            elapsed = (self.end_time - self.start_time).total_seconds()
            report['agent_name'] = self.name
            report['elapsed_seconds'] = round(elapsed, 1)
            report['status'] = 'ok'

            print(f"  [{self.name}] ✅ 完成 ({elapsed:.1f}s)")
            return report

        except Exception as e:
            self.errors.append({
                'agent': self.name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
            print(f"  [{self.name}] ❌ 异常: {e}")
            return self._error_report(e)

    @abstractmethod
    def validate_data(self):
        """验证所需数据是否可用。返回 True/False"""
        pass

    @abstractmethod
    def analyze(self):
        """核心分析逻辑。返回分析结果字典"""
        pass

    @abstractmethod
    def report(self, result):
        """将分析结果格式化为报告。返回字典"""
        pass

    def _degraded_report(self):
        """数据不足时的降级报告"""
        return {
            'agent_name': self.name,
            'status': 'degraded',
            'summary': '数据不足，跳过本次分析',
            'alerts': [],
            'data_status': self.data_status,
            'elapsed_seconds': (datetime.now() - self.start_time).total_seconds(),
        }

    def _error_report(self, exception):
        """异常报告"""
        return {
            'agent_name': self.name,
            'status': 'error',
            'summary': f'分析异常: {str(exception)}',
            'alerts': [],
            'errors': self.errors,
            'elapsed_seconds': (datetime.now() - self.start_time).total_seconds(),
        }

    def check_table_exists(self, conn, table_name):
        """检查DuckDB表是否存在"""
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()
        return table_name in [t[0] for t in tables]

    def check_data_freshness(self, conn, table, date_col='trade_date', max_age_days=2):
        """检查数据新鲜度"""
        max_date = conn.execute(f"SELECT MAX({date_col}) FROM {table}").fetchone()[0]
        if max_date is None:
            return False, None, 999
        max_date = str(max_date)
        today = datetime.now().strftime('%Y-%m-%d')
        age = (datetime.strptime(today, '%Y-%m-%d') -
               datetime.strptime(max_date[:10], '%Y-%m-%d')).days
        return age <= max_age_days, max_date, age
