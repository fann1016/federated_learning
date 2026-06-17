# FedCM paper reproduction — CIFAR-10, ResNet-18-GN, Setting I
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:WANDB_MODE = "disabled"
& .\.venv312\Scripts\python.exe examples\run_fedcm.py @args
