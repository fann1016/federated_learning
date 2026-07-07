$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$OutDir = Join-Path $Root "experiments\waffle_strict_mnist_l5_baseline_iid_seed0"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:WANDB_MODE = "disabled"
$env:MPLBACKEND = "Agg"

$Python = Join-Path $Root ".venv312\Scripts\python.exe"
$Script = Join-Path $Root "examples\run_waffle_mnist_l5_baseline_strict.py"
$ConsoleLog = Join-Path $OutDir "mnist_baseline_strict_console.txt"

& $Python -u $Script 2>&1 | Tee-Object -FilePath $ConsoleLog
