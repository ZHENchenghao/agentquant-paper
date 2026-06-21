# -*- coding: utf-8 -*-
"""
Master Pipeline: Backfill -> Rebuild Factors -> Rolling Backtest -> Shutdown
2002-2026 full cycle with NLP sentiment
"""
import sys, io, os, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from datetime import datetime

STEPS = [
    ('Step 1: Backfill K-line 2002-2015', 'backfill_kline_2002_sequence.py'),
    ('Step 2: Rebuild Factor Cache 2002+', 'rebuild_factors_2002.py'),
    ('Step 3: Rolling Backtest + NLP', 'rolling_backtest_final.py'),
]

log = []
t0 = time.time()

def log_msg(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = '[%s] %s' % (ts, msg)
    print(line, flush=True)
    log.append(line)

log_msg('Master Pipeline START')

for step_name, script in STEPS:
    log_msg(step_name + '...')
    t1 = time.time()
    result = subprocess.run(['python', script], capture_output=True, text=True, timeout=7200, cwd=os.path.dirname(__file__))
    elapsed = time.time() - t1
    if result.returncode == 0:
        log_msg('  OK (%.0fs)' % elapsed)
        # Print last 5 lines of output
        lines = result.stdout.strip().split('\n')
        for l in lines[-5:]:
            log_msg('    ' + l[:120])
    else:
        log_msg('  FAILED (exit=%d)' % result.returncode)
        log_msg('  STDERR: ' + result.stderr[-200:])
        break

log_msg('Total: %.0f min' % ((time.time()-t0)/60))

# Save log
with open('cache/master_pipeline_log.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log))

# Shutdown
log_msg('Shutting down...')
os.system('shutdown /s /t 60')
