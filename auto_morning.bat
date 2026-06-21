@echo off
cd /d D:\AgentQuant\our
echo === Auto Paper Trading Morning %date% %time% === >> auto_trade_log.txt
python -u daily_runner.py >> auto_trade_log.txt 2>&1
python -u paper_executor.py >> auto_trade_log.txt 2>&1
echo Done Morning >> auto_trade_log.txt
git add paper_portfolio_xiaozhong.json execution_plan_*.json auto_trade_log.txt >> auto_trade_log.txt 2>&1
git commit -m "auto: 盘前纸交 %date%" >> auto_trade_log.txt 2>&1
git push origin master >> auto_trade_log.txt 2>&1
