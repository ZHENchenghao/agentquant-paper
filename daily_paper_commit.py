# -*- coding: utf-8 -*-
"""
每日纸交自动执行 + Git存档
用法: python daily_paper_commit.py
"""
import subprocess, sys, os, datetime, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TODAY = datetime.date.today().isoformat()
PF = 'D:/AgentQuant/our/paper_portfolio.json'
LOG = 'D:/AgentQuant/our/paper_daily_log.md'

# 1. 跑纸交选股
print(f'[{TODAY}] 运行纸交引擎...')
ret = subprocess.run([sys.executable, 'D:/AgentQuant/our/paper_trade_ml.py'],
                     cwd='D:/AgentQuant/our', capture_output=True, text=True)
print(ret.stdout[-500:] if ret.stdout else '')
if ret.returncode != 0:
    print(f'纸交失败: {ret.stderr[-300:]}')
    sys.exit(1)

# 2. 追加日志
import json
with open(PF, 'r', encoding='utf-8') as f:
    pf = json.load(f)

positions = pf.get('positions', {})
history = pf.get('history', [])
last_day = history[-1]['date'] if history else 'N/A'
total_mv = pf['cash']
for c, p in positions.items():
    total_mv += p['shares'] * p.get('buy_price', 0)

with open(LOG, 'a', encoding='utf-8') as f:
    f.write(f'| {TODAY} | {len(positions)}只 | {total_mv:,.0f} | {last_day} |\n')

# 3. Git commit
os.chdir('D:/AgentQuant/our')
for cmd in [
    'git add paper_portfolio.json paper_daily_log.md',
    f'git commit -m "纸交 {TODAY} — {len(positions)}只持仓" --allow-empty',
]:
    subprocess.run(cmd, shell=True, capture_output=True)

print(f'[{TODAY}] 完成 — {len(positions)}只持仓, 已commit')
