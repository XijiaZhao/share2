# AudioAction 一键关闭脚本
# 只关掉 serve.py (8025) 和 app_server.py (8026) 这两个进程，不会误杀别的 Python。
# 用法：在 PowerShell 里执行  .\stop.ps1

$killed = 0
Get-CimInstance Win32_Process -Filter "name='python.exe'" |
    Where-Object { $_.CommandLine -match 'serve\.py|app_server\.py' } |
    ForEach-Object {
        $tag = if ($_.CommandLine -match 'serve\.py') { "模型服务 (8025)" } else { "测试网页 (8026)" }
        Write-Host "[stop] $tag  PID=$($_.ProcessId)" -ForegroundColor Yellow
        Stop-Process -Id $_.ProcessId -Force
        $killed++
    }

if ($killed -eq 0) {
    Write-Host "没有发现正在运行的 AudioAction 服务。" -ForegroundColor Gray
} else {
    Write-Host "已关闭 $killed 个服务。" -ForegroundColor Green
}
