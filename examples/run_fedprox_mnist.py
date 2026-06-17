"""Thin wrapper — FedProx paper reproduction (MNIST). See run_paper.py."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.run_paper import build_overrides, parse_args
from src.federated.trainer import run_paper_experiment

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FedProx paper reproduction (MNIST)")
    parser.add_argument("--algorithm", choices=["FedProx", "FedAvg"], default="FedProx")
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--drop-percent", type=float, default=None, dest="drop_percent")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()

    overrides = {}
    if args.rounds is not None:
        overrides["rounds"] = args.rounds
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.mu is not None:
        overrides["mu"] = args.mu
    elif args.algorithm == "FedAvg":
        overrides["mu"] = 0.0
    if args.drop_percent is not None:
        overrides["drop_percent"] = args.drop_percent

    run_paper_experiment(args.algorithm, overrides)
