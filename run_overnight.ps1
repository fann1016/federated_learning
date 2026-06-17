# 过夜完整实验：FedProx 200轮 + FedAvg 200轮（论文 MNIST 设置）
# 用法：在 PowerShell 里执行
#   cd "D:\论文-联邦学习\联邦学习代码"
#   powershell -ExecutionPolicy Bypass -File .\run_overnight.ps1

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$BasePy = "C:/Users/Administrator/AppData/Roaming/uv/python/cpython-3.12.13-windows-x86_64-none/python.exe"
$VenvPy = Join-Path $ProjectRoot ".venv312\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "experiments\fedprox_mnist\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$MainLog = Join-Path $LogDir "overnight_$Timestamp.log"

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $MainLog -Value $line
}

# ---------- 1. 环境 ----------
if (-not (Test-Path $VenvPy)) {
    Write-Log "创建虚拟环境 .venv312 ..."
    & $BasePy -m venv (Join-Path $ProjectRoot ".venv312")
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    Write-Log "虚拟环境已存在，跳过安装"
}

$env:WANDB_MODE = "disabled"

Write-Log "Python: $VenvPy"
& $VenvPy --version 2>&1 | ForEach-Object { Write-Log $_ }

# ---------- 2. 快速试跑 1 轮（确认能跑） ----------
Write-Log "===== 试跑 1 轮 ====="
& $VenvPy examples\run_fedprox_mnist.py --algorithm FedProx --mu 0.01 --rounds 1 --device cpu 2>&1 |
    Tee-Object -FilePath (Join-Path $LogDir "smoke_$Timestamp.log")
if ($LASTEXITCODE -ne 0) {
    Write-Log "试跑失败，请查看日志: $LogDir\smoke_$Timestamp.log"
    pause
    exit 1
}
Write-Log "试跑成功，开始整夜实验"

# ---------- 3. FedProx 200 轮 ----------
Write-Log "===== FedProx 200 轮 开始 ====="
& $VenvPy examples\run_fedprox_mnist.py --algorithm FedProx --mu 0.01 --rounds 200 --device cpu 2>&1 |
    Tee-Object -FilePath (Join-Path $LogDir "fedprox_200_$Timestamp.log")
Write-Log "===== FedProx 200 轮 结束 (exit=$LASTEXITCODE) ====="

# ---------- 4. FedAvg 基线 200 轮 ----------
Write-Log "===== FedAvg 200 轮 开始 ====="
& $VenvPy examples\run_fedprox_mnist.py --algorithm FedAvg --mu 0 --rounds 200 --device cpu 2>&1 |
    Tee-Object -FilePath (Join-Path $LogDir "fedavg_200_$Timestamp.log")
Write-Log "===== FedAvg 200 轮 结束 (exit=$LASTEXITCODE) ====="

Write-Log "全部完成。主日志: $MainLog"
Write-Log "结果目录: experiments\fedprox_mnist\"
pause
