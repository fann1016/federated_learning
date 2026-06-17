# SCAFFOLD paper reproduction — EMNIST, logistic regression (ICML 2020 Sec. 7)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:WANDB_MODE = "disabled"
& .\.venv312\Scripts\python.exe examples\run_paper.py --algorithm SCAFFOLD @args
