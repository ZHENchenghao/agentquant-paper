# -*- coding: utf-8 -*-
"""
QuantLab 共享层: 统一报告格式
"""
import json
from datetime import datetime


class LabReport:
    """三Agent统一报告容器"""

    def __init__(self, title="QuantLab 日报"):
        self.title = title
        self.generated = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.sections = {}
        self.alerts = []
        self.meta = {}

    def add_agent_result(self, agent_name, result):
        """添加Agent输出"""
        self.sections[agent_name] = result
        if result.get('alerts'):
            self.alerts.extend(result['alerts'])

    def set_meta(self, key, value):
        self.meta[key] = value

    def to_dict(self):
        return {
            'title': self.title,
            'generated': self.generated,
            'meta': self.meta,
            'alerts': self.alerts,
            'sections': self.sections,
        }

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def save(self, path=None):
        if path is None:
            date_str = datetime.now().strftime('%Y-%m-%d')
            path = f'D:/AgentQuant/our/lab/reports/daily/lab_{date_str}.json'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return path

    def print_summary(self):
        """终端摘要输出"""
        print(f"\n{'='*70}")
        print(f"  {self.title} | {self.generated}")
        print(f"{'='*70}")
        for name, result in self.sections.items():
            status = result.get('status', 'unknown')
            print(f"\n  [{name}] 状态: {status}")
            if result.get('summary'):
                print(f"    {result['summary']}")
        if self.alerts:
            print(f"\n  ⚠ 告警 ({len(self.alerts)}条):")
            for a in self.alerts[:5]:
                print(f"    • {a}")
        print(f"\n{'='*70}\n")
