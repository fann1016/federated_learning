# FedSpeed paper reproduction — CIFAR-10 Setting I (ICLR 2023)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:WANDB_MODE = "disabled"
& .\.venv312\Scripts\python.exe examples\run_paper.py --algorithm FedSpeed @args
