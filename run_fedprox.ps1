# FedProx paper reproduction — MNIST, logistic regression
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:WANDB_MODE = "disabled"
& .\.venv312\Scripts\python.exe examples\run_paper.py --algorithm FedProx @args
