@echo off
chcp 65001 >nul
echo ============================================================
echo  AgentQuant 每日全任务 — %date% %time%
echo ============================================================

echo.
echo [1/4] 刷新市场数据...
cd /d C:\Users\Lenovo\Desktop\ceshi\天眼
python tianyan.py daily
if %errorlevel% neq 0 echo ⚠ 数据刷新异常,继续执行...

echo.
echo [2/5] 日检脉象...
cd /d C:\Users\Lenovo\Desktop\ceshi\天眼
python engine\daily_pulse.py > D:\AgentQuant\our\pulse_today.txt 2>&1
type D:\AgentQuant\our\pulse_today.txt
if %errorlevel% neq 0 echo ⚠ 脉象异常,继续执行...

echo.
echo [3/5] ML纸交选股...
cd /d D:\AgentQuant\our
python paper_trade_ml.py
if %errorlevel% neq 0 echo ⚠ 纸交异常,继续执行...

echo.
echo [4/5] Git存档+推送...
echo ## %date% >> pulse_daily_log.md
type pulse_today.txt >> pulse_daily_log.md
echo. >> pulse_daily_log.md
git add paper_portfolio.json paper_daily_log.md pulse_daily_log.md pulse_today.txt
git commit -m "纸交+脉象 %date%"
git push origin master 2>&1
if %errorlevel% neq 0 echo ⚠ Git推送失败,已本地存档

echo.
echo [5/5] 60秒后关机...
echo ============================================================
echo  全部任务完成 — %time%
echo ============================================================
shutdown /s /t 60 /c "AgentQuant每日任务完成, 系统将在60秒后关机"
echo.
echo 取消关机命令: shutdown /a
