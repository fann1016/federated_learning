"""
Per-paper experimental defaults for federated learning reproduction.

Each algorithm uses the dataset, model, and hyperparameters from its own paper.
Use get_paper_config(algorithm) to obtain the full config dict.
"""

import copy
import os

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

_COMMON = {
    "device": "cpu",
    "seed": 0,
    "show_plots": False,
    "results_path": _BASE_DIR,
    "data_path": _DATA_DIR,
    "optimizer_momentum": 0.0,
    "server_momentum_beta": 0.0,
    "server_nesterov": False,
    "client_nesterov": False,
    "participation_mode": "fixed",
    # lr_decay is applied per communication round: lr_t = lr * lr_decay^t
    "lr_decay_per_round": True,
}

# FedProx — MLSys 2020, Appendix C.1/C.2 (MNIST)
_FEDPROX = {
    **_COMMON,
    "algorithm": "FedProx",
    "paper": "FedProx (MLSys 2020)",
    "dataset_name": "MNIST_FEDPROX",
    "partition_method": "fedprox_mnist",
    "model_name": "mnist_logistic",
    "num_classes": 10,
    "num_clients": 1000,
    "clients_per_round": 10,
    "local_epochs": 20,
    "batch_size": 10,
    "lr": 0.01,
    "lr_decay": 1.0,
    "mu": 0.01,
    "client_train_frac": 0.9,
    "drop_percent": 0.0,
    "weight_decay": 0.0,
    "server_lr": 1.0,
    "paper_rounds": 200,
    "rounds": 200,
    "eval_interval": 1,
    "experiments_subdir": "fedprox_mnist",
    "run_prefix": "mnist_fedprox",
    "wandb_project": "federated-learning-fedprox-paper",
    "summary_csv": os.path.join(_BASE_DIR, "experiments", "mnist_fedprox_summaries.csv"),
}

# FedCM — arXiv:2106.10874, Section 6 + Appendix C
# Setting I: CIFAR-10, 100 clients, 10% participation, Dirichlet-0.6, ResNet-18-GN
# Paper reports results at 4000 rounds; default 500 here for local/dev runs.
# Full reproduction: --rounds 4000
_FEDCM = {
    **_COMMON,
    "algorithm": "FedCM",
    "paper": "FedCM (arXiv:2106.10874) — CIFAR-10 Setting I",
    "dataset_name": "CIFAR10",
    "partition_method": "dirichlet",
    "dirichlet_alpha": 0.6,
    "model_name": "resnet18_gn",
    "num_classes": 10,
    "num_clients": 100,
    "participation_mode": "probabilistic",
    "participation_prob": 0.1,
    "local_epochs": 5,
    "batch_size": 50,
    "lr": 0.1,
    "lr_decay": 0.998,
    "fedcm_alpha": 0.1,
    "weight_decay": 0.001,
    "server_lr": 1.0,
    "paper_rounds": 4000,
    "rounds": 500,
    "eval_interval": 10,
    "experiments_subdir": "fedcm_cifar10",
    "run_prefix": "cifar10_fedcm",
    "wandb_project": "federated-learning-fedcm-paper",
    "summary_csv": os.path.join(_BASE_DIR, "experiments", "cifar10_fedcm_summaries.csv"),
}

# SCAFFOLD — ICML 2020 (Karimireddy et al.), Section 7.1/7.3
# EMNIST byclass, logistic regression, N=100, fixed 20% client sampling per round
# 1 epoch = 5 local SGD steps, batch_frac=0.2, η_g=1, Option II control variates
# similarity_frac: 0.0 = 0% similar (label-sorted), 1.0 = 100% IID (Hsu et al. style)
# Paper tunes η_l per algorithm; override with --lr after grid search if needed
_SCAFFOLD = {
    **_COMMON,
    "algorithm": "SCAFFOLD",
    "paper": "SCAFFOLD (ICML 2020) — EMNIST logistic regression",
    "dataset_name": "EMNIST",
    "emnist_split": "byclass",
    "partition_method": "emnist_similarity",
    "similarity_frac": 0.0,
    "model_name": "emnist_logistic",
    "num_classes": 62,
    "num_clients": 100,
    "clients_per_round": 20,
    "local_steps": 5,
    "batch_frac": 0.2,
    "eval_batch_size": 256,
    "local_epochs": 1,
    "lr": 1.0,
    "lr_decay": 1.0,
    "weight_decay": 0.0,
    "server_lr": 1.0,
    "scaffold_control_option": "II",
    "target_accuracy": 50.0,
    "paper_rounds": 1000,
    "rounds": 200,
    "eval_interval": 5,
    "experiments_subdir": "scaffold_emnist",
    "run_prefix": "emnist_scaffold",
    "wandb_project": "federated-learning-scaffold-paper",
    "summary_csv": os.path.join(_BASE_DIR, "experiments", "emnist_scaffold_summaries.csv"),
}

# FedSpeed — ICLR 2023 (Sun et al.), Section 5.1 Setting I
# CIFAR-10, 100 clients, 10% probabilistic participation, Dirichlet-0.6, ResNet-18-GN
# λ=0.1, SAM ρ=0.1, lr_decay=0.9995/round (FedSpeed-specific, not 0.998)
# Paper Figure 1 uses 1500 rounds; default 200 for local runs
_FEDSPEED = {
    **_COMMON,
    "algorithm": "FedSpeed",
    "paper": "FedSpeed (ICLR 2023) — CIFAR-10 Setting I",
    "dataset_name": "CIFAR10",
    "partition_method": "dirichlet",
    "dirichlet_alpha": 0.6,
    "model_name": "resnet18_gn",
    "num_classes": 10,
    "num_clients": 100,
    "participation_mode": "probabilistic",
    "participation_prob": 0.1,
    "local_epochs": 5,
    "batch_size": 50,
    "lr": 0.1,
    "lr_decay": 0.9995,
    "lamb": 0.1,
    "sam_rho": 0.1,
    "sam_alpha": 1.0,
    "grad_clip_norm": 10.0,
    "weight_decay": 0.001,
    "server_lr": 1.0,
    "paper_rounds": 1500,
    "rounds": 200,
    "eval_interval": 10,
    "experiments_subdir": "fedspeed_cifar10",
    "run_prefix": "cifar10_fedspeed",
    "wandb_project": "federated-learning-fedspeed-paper",
    "summary_csv": os.path.join(_BASE_DIR, "experiments", "cifar10_fedspeed_summaries.csv"),
}

PAPER_CONFIGS = {
    "FedProx": _FEDPROX,
    "FedAvg": {**_FEDPROX, "algorithm": "FedAvg", "mu": 0.0},
    "FedCM": _FEDCM,
    "SCAFFOLD": _SCAFFOLD,
    "FedSpeed": _FEDSPEED,
}


def get_paper_config(algorithm, overrides=None):
    key = str(algorithm)
    if key not in PAPER_CONFIGS:
        supported = ", ".join(sorted(PAPER_CONFIGS.keys()))
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {supported}")

    config = copy.deepcopy(PAPER_CONFIGS[key])
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def list_paper_algorithms():
    return sorted(PAPER_CONFIGS.keys())
