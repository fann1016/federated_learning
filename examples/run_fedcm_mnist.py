"""Thin wrapper — FedCM paper reproduction (CIFAR-10). See run_paper.py."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.federated.trainer import run_paper_experiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedCM paper reproduction (CIFAR-10 Setting I)")
    parser.add_argument("--alpha", type=float, default=None, help="Client momentum α (paper: 0.1)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--participation",
        choices=["fixed", "probabilistic"],
        default=None,
    )
    args = parser.parse_args()

    overrides = {}
    if args.alpha is not None:
        overrides["fedcm_alpha"] = args.alpha
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.participation is not None:
        overrides["participation_mode"] = args.participation

    run_paper_experiment("FedCM", overrides)
