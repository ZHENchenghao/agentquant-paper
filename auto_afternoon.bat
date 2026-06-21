@echo off
cd /d D:\AgentQuant\our
echo === Auto Paper Trading Afternoon %date% %time% === >> auto_trade_log.txt
python tianyan.py daily >> auto_trade_log.txt 2>&1
python -u paper_executor.py >> auto_trade_log.txt 2>&1
echo Done Afternoon >> auto_trade_log.txt
git add paper_portfolio_xiaozhong.json execution_plan_*.json auto_trade_log.txt >> auto_trade_log.txt 2>&1
git commit -m "auto: 收盘纸交 %date%" >> auto_trade_log.txt 2>&1
git push origin master >> auto_trade_log.txt 2>&1
echo === Sleep at %time% === >> auto_trade_log.txt
rundll32.exe powrprof.dll,SetSuspendState 0,1,0
