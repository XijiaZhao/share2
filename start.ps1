# AudioAction 一键启动脚本（后台运行）
# 在后台隐藏窗口启动模型服务 (8025) 和测试网页 (8026)，关掉终端也继续运行。
# 日志写入 logs\ 目录。用法：在 PowerShell 里执行  .\start.ps1

$ErrorActionPreference = "Stop"
$Root    = $PSScriptRoot
$TestApp = Join-Path $Root "testapp"
$LogDir  = Join-Path $Root "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Anaconda 的 OpenMP 冲突修复，缺这个 Python 会直接崩溃
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Test-Port($port) {
    $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

# 后台隐藏启动一个服务，stdout/stderr 重定向到日志文件
function Start-Svc($dir, $pyArgs, $log) {
    Start-Process -FilePath "python" -ArgumentList $pyArgs `
        -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
}

# 1) 模型服务 (8025)
if (Test-Port 8025) {
    Write-Host "[skip] 8025 已在运行" -ForegroundColor Yellow
} else {
    Write-Host "[start] 模型服务 -> http://127.0.0.1:8025 （后台）" -ForegroundColor Green
    Start-Svc $Root "serve.py --config config.yml" (Join-Path $LogDir "serve.log")
}

# 等模型服务起来（最多 ~30 秒），它就绪后测试网页才连得上
Write-Host "[wait] 等待模型服务就绪..." -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8025/healthz" -TimeoutSec 2
        if ($r.status -eq "ok") {
            Write-Host "[ok] 模型服务就绪 (model=$($r.model) device=$($r.device))" -ForegroundColor Green
            $ready = $true
            break
        }
    } catch { }
}
if (-not $ready) {
    Write-Host "[warn] 30 秒内未确认模型服务就绪，请查看 logs\serve.log / serve.log.err" -ForegroundColor Yellow
}

# 2) 测试网页 (8026)
if (Test-Port 8026) {
    Write-Host "[skip] 8026 已在运行" -ForegroundColor Yellow
} else {
    Write-Host "[start] 测试网页 -> http://localhost:8026/ （后台）" -ForegroundColor Green
    Start-Svc $TestApp "app_server.py --config config.yml" (Join-Path $LogDir "app.log")
}

Write-Host ""
Write-Host "完成（后台运行）。浏览器打开： http://localhost:8026/" -ForegroundColor Green
Write-Host "查看日志： logs\serve.log  /  logs\app.log" -ForegroundColor Gray
Write-Host "关闭服务： .\stop.ps1" -ForegroundColor Gray
