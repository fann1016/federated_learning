"""
Reproduce federated learning papers with per-algorithm experimental settings.

Each algorithm loads its own dataset, model, partition, and hyperparameters
from src/config/paper_configs.py.

Examples:
  python examples/run_paper.py --algorithm FedProx --rounds 200 --device cpu
  python examples/run_paper.py --algorithm FedCM --rounds 10 --device cpu
  python examples/run_paper.py --algorithm FedCM  # default 200 rounds (paper: 4000)
  python examples/run_paper.py --algorithm FedCM --rounds 4000  # full paper reproduction
  python examples/run_paper.py --algorithm FedSpeed --rounds 2 --device cpu
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.paper_configs import get_paper_config, list_paper_algorithms
from src.federated.trainer import run_paper_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce FL papers with algorithm-specific settings"
    )
    parser.add_argument(
        "--algorithm",
        choices=list_paper_algorithms(),
        required=True,
        help="Algorithm to reproduce (each uses its own paper config)",
    )
    parser.add_argument("--rounds", type=int, default=None, help="Override communication rounds")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--eval-interval", type=int, default=None, dest="eval_interval")
    parser.add_argument("--mu", type=float, default=None, help="FedProx prox coefficient")
    parser.add_argument(
        "--drop-percent",
        type=float,
        default=None,
        dest="drop_percent",
        help="FedProx/FedAvg system heterogeneity fraction",
    )
    parser.add_argument("--alpha", type=float, default=None, help="FedCM client momentum α")
    parser.add_argument("--lr", type=float, default=None, help="Override local learning rate η_l")
    parser.add_argument(
        "--participation",
        choices=["fixed", "probabilistic"],
        default=None,
        help="Override client selection (FedCM/FedSpeed default: probabilistic)",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=None,
        help="SCAFFOLD EMNIST similarity in [0,1]: 0=sorted non-IID, 1=IID",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model_name (e.g. emnist_logistic, emnist_mlp)",
    )
    return parser.parse_args()


def build_overrides(args):
    overrides = {}
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    else:
        overrides["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    if args.eval_interval is not None:
        overrides["eval_interval"] = args.eval_interval
    if args.mu is not None:
        overrides["mu"] = args.mu
    if args.drop_percent is not None:
        overrides["drop_percent"] = args.drop_percent
    if args.alpha is not None:
        overrides["fedcm_alpha"] = args.alpha
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.participation is not None:
        overrides["participation_mode"] = args.participation
    if args.similarity is not None:
        overrides["similarity_frac"] = args.similarity
    if args.model is not None:
        overrides["model_name"] = args.model
    return overrides


def main():
    args = parse_args()
    overrides = build_overrides(args)
    config = get_paper_config(args.algorithm, overrides)

    print("Paper-aligned configuration:")
    print(f"  Algorithm : {config['algorithm']}")
    print(f"  Paper     : {config.get('paper')}")
    print(f"  Dataset   : {config['dataset_name']}")
    print(f"  Model     : {config['model_name']}")
    part_mode = config.get("participation_mode", "fixed")
    if part_mode == "probabilistic":
        part_desc = f"prob {config.get('participation_prob')}"
    else:
        part_desc = str(config.get("clients_per_round"))
    print(f"  Clients   : {config['num_clients']} (per round: {part_desc}, mode={part_mode})")
    if "similarity_frac" in config:
        print(f"  Similarity: {config['similarity_frac']} (0=sorted, 1=IID)")
    if config.get("target_accuracy") is not None:
        print(f"  Target acc: {config['target_accuracy']}% (paper Table 3 metric)")
    print(f"  Rounds    : {config['rounds']} (paper: {config.get('paper_rounds', 'N/A')})")
    print(f"  Device    : {config['device']}")

    run_paper_experiment(args.algorithm, overrides)


if __name__ == "__main__":
    main()
