$action1 = New-ScheduledTaskAction -Execute "D:\AgentQuant\our\auto_morning.bat"
$trigger1 = New-ScheduledTaskTrigger -Daily -At "08:30AM"
$settings = New-ScheduledTaskSettingsSet -WakeToRun -AllowStartIfOnBatteries
Register-ScheduledTask -TaskName "AgentQuant_Morning" -Action $action1 -Trigger $trigger1 -Settings $settings -Force

$action2 = New-ScheduledTaskAction -Execute "D:\AgentQuant\our\auto_afternoon.bat"
$trigger2 = New-ScheduledTaskTrigger -Daily -At "03:32PM"
Register-ScheduledTask -TaskName "AgentQuant_Afternoon" -Action $action2 -Trigger $trigger2 -Settings $settings -Force

Write-Host "Done: 2 tasks registered"
schtasks /query /tn AgentQuant_Morning /fo LIST 2>$null
schtasks /query /tn AgentQuant_Afternoon /fo LIST 2>$null
