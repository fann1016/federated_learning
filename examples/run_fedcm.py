"""FedCM paper reproduction entry point.

Paper default: CIFAR-10 Setting I, Dirichlet-0.6, ResNet-18-GN, alpha=0.1.
Full paper runs use 4000 communication rounds.
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.federated.trainer import run_paper_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce FedCM: Federated Learning with Client-level Momentum"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--split", choices=["dirichlet", "iid"], default="dirichlet")
    parser.add_argument("--setting", choices=["I", "II"], default="I")
    parser.add_argument("--alpha", type=float, default=None, help="FedCM decay alpha")
    parser.add_argument("--rounds", type=int, default=None, help="Paper uses 4000")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--data-path", default=None, help="Override dataset cache path")
    parser.add_argument("--eval-interval", type=int, default=None, dest="eval_interval")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-decay", type=float, default=None, dest="lr_decay")
    return parser.parse_args()


def build_overrides(args):
    overrides = {}

    if args.dataset == "cifar100":
        overrides.update({
            "dataset_name": "CIFAR100",
            "num_classes": 100,
            "summary_csv": os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "experiments",
                "cifar100_fedcm_summaries.csv",
            ),
            "experiments_subdir": "fedcm_cifar100",
            "run_prefix": "cifar100_fedcm",
        })
    else:
        overrides.update({
            "dataset_name": "CIFAR10",
            "num_classes": 10,
        })

    if args.split == "iid":
        overrides["partition_method"] = "iid"
    else:
        overrides["partition_method"] = "dirichlet"
        overrides["dirichlet_alpha"] = 0.6

    if args.setting == "II":
        overrides["num_clients"] = 500
        overrides["participation_prob"] = 0.02
    else:
        overrides["num_clients"] = 100
        overrides["participation_prob"] = 0.1

    optional = {
        "fedcm_alpha": args.alpha,
        "rounds": args.rounds,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "data_path": args.data_path,
    }
    overrides.update({key: value for key, value in optional.items() if value is not None})

    overrides["device"] = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return overrides


if __name__ == "__main__":
    run_paper_experiment("FedCM", build_overrides(parse_args()))
