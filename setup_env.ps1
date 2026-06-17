# 用 uv 的 Python 3.12 创建虚拟环境并安装依赖（只需运行一次）
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$BasePy = "C:/Users/Administrator/AppData/Roaming/uv/python/cpython-3.12.13-windows-x86_64-none/python.exe"
$VenvPy = "$ProjectRoot/.venv312/Scripts/python.exe"

if (-not (Test-Path $BasePy)) {
    Write-Host "找不到 Python: $BasePy" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "创建虚拟环境 .venv312 ..."
& $BasePy -m venv "$ProjectRoot/.venv312"

Write-Host "安装依赖（可能需要几分钟）..."
& $VenvPy -m pip install --upgrade pip
& $VenvPy -m pip install -r "$ProjectRoot/requirements.txt"

Write-Host ""
Write-Host "完成。以后用这个 Python 运行实验：" -ForegroundColor Green
Write-Host $VenvPy
Write-Host ""
pause
