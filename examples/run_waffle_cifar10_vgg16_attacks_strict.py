import copy
import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.dataloader import DataLoaderManager
from src.data.watermark import PatternWatermarkDataset
from src.federated.server import Server
from src.models.factory import build_model
from src.utils.helpers import append_metrics_row, set_seed


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
RESULTS_DIR = os.path.join(
    BASE_DIR,
    "experiments",
    "waffle_strict_cifar10_vgg16_cwaffle_iid_seed0",
)
DEFAULT_WATERMARK_DIR = os.path.join(
    WORKSPACE_DIR,
    "WAFFLE-main",
    "WAFFLE-main",
    "src",
    "data",
    "datasets",
    "CWAFFLE",
)


CONFIG = {
    "model_name": "waffle_cifar_vgg16_strict",
    "vgg_pretrained": True,
    "cifar_transform": "waffle",
    "cifar_normalize": "imagenet",
    "input_size": 32,
    "lr": 0.0005,
    "attack_lr": 0.01,
    "batch_size": 64,
    "attack_batch_size": 50,
    "num_clients": 100,
    "fraction": 0.1,
    "local_epochs": 1,
    "partition_method": "iid",
    "server_momentum_beta": 0.0,
    "server_lr": 1.0,
    "server_nesterov": False,
    "client_nesterov": False,
    "weight_decay": 0.00005,
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_path": os.path.join(BASE_DIR, "data_clean"),
    "num_classes": 10,
    "watermark_data_path": DEFAULT_WATERMARK_DIR,
    "watermark_batch_size": 50,
    "watermark_normalize": "imagenet",
    "results_path": BASE_DIR,
}


def _latest_model_file(results_dir):
    summary_csv = os.path.join(results_dir, "summary.csv")
    if os.path.exists(summary_csv):
        with open(summary_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in reversed(rows):
            model_file = row.get("model_file")
            if model_file and os.path.exists(model_file):
                return model_file

    candidates = []
    for name in os.listdir(results_dir) if os.path.isdir(results_dir) else []:
        if name.endswith("_final_model.pt"):
            path = os.path.join(results_dir, name)
            candidates.append((os.path.getmtime(path), path))
    if candidates:
        return sorted(candidates)[-1][1]

    raise FileNotFoundError(
        "No CIFAR-10 WAFFLE strict checkpoint found. Run "
        "examples/run_waffle_cifar10_vgg16_strict.py first."
    )


def _load_checkpoint_model(config, model_file):
    model = build_model(
        config["model_name"],
        num_classes=config["num_classes"],
        pretrained=bool(config.get("vgg_pretrained", False)),
    )
    checkpoint = torch.load(model_file, map_location=config["device"])
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model


def _prepare_data(config):
    data_manager = DataLoaderManager(config)
    train_dataset, test_dataset = data_manager.load_data(dataset_name="CIFAR10")
    user_groups = data_manager.partition_data(method="iid")
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    client_loaders = [
        DataLoader(
            Subset(train_dataset, list(user_groups[i])),
            batch_size=int(config["attack_batch_size"]),
            shuffle=True,
        )
        for i in range(config["num_clients"])
    ]
    watermark_dataset = PatternWatermarkDataset(
        config["watermark_data_path"],
        image_size=int(config.get("input_size", 32)),
        grayscale=False,
        normalize=config.get("watermark_normalize", "imagenet"),
    )
    watermark_loader = DataLoader(
        watermark_dataset,
        batch_size=int(config["watermark_batch_size"]),
        shuffle=True,
    )
    return test_loader, watermark_loader, client_loaders


def _evaluate_model(config, model, test_loader, watermark_loader):
    server = Server(model, config)
    clean_loss, clean_acc = server.evaluate_loader(test_loader)
    wm_loss, wm_acc = server.evaluate_loader(watermark_loader)
    return clean_loss, clean_acc, wm_loss, wm_acc


def _finetune_on_adversaries(config, model, client_loaders, n_adversaries):
    model.to(config["device"])
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["attack_lr"]))
    model.train()
    total_loss = 0.0

    for _ in range(int(config["local_epochs"])):
        for client_i in range(int(n_adversaries)):
            loader = client_loaders[client_i]
            for data, target in loader:
                data = data.to(config["device"])
                target = target.to(config["device"])
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() / max(1, len(loader.dataset))

    return total_loss


def _apply_l1_pruning(model, level):
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=float(level))
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=float(level))


def run_finetuning_attack(config, model_file, test_loader, watermark_loader, client_loaders, results_dir):
    rows_path = os.path.join(results_dir, "attack_finetuning_summary.csv")
    adversary_counts = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    repeats = int(config.get("attack_repeats", 4))

    for n_adversaries in adversary_counts:
        for repeat in range(repeats):
            model = _load_checkpoint_model(config, model_file)
            before_clean_loss, before_clean_acc, before_wm_loss, before_wm_acc = _evaluate_model(
                config, model, test_loader, watermark_loader
            )
            train_loss = _finetune_on_adversaries(
                config, model, client_loaders, n_adversaries
            )
            clean_loss, clean_acc, wm_loss, wm_acc = _evaluate_model(
                config, model, test_loader, watermark_loader
            )
            row = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "attack": "finetuning",
                "repeat": repeat,
                "n_adversaries": n_adversaries,
                "local_epochs": config["local_epochs"],
                "attack_lr": config["attack_lr"],
                "train_loss": train_loss,
                "before_test_accuracy": before_clean_acc,
                "before_watermark_accuracy": before_wm_acc,
                "test_accuracy": clean_acc,
                "test_loss": clean_loss,
                "watermark_accuracy": wm_acc,
                "watermark_loss": wm_loss,
                "model_file": model_file,
            }
            append_metrics_row(rows_path, row)
            print(
                "Finetuning "
                f"adv={n_adversaries}, repeat={repeat}: "
                f"clean {before_clean_acc:.2f}->{clean_acc:.2f}, "
                f"wm {before_wm_acc:.2f}->{wm_acc:.2f}"
            )


def run_pruning_attack(config, model_file, test_loader, watermark_loader, client_loaders, results_dir):
    rows_path = os.path.join(results_dir, "attack_pruning_summary.csv")
    pruning_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    repeats = int(config.get("attack_repeats", 4))
    n_adversaries = 1

    for repeat in range(repeats):
        base_model = _load_checkpoint_model(config, model_file)
        before_clean_loss, before_clean_acc, before_wm_loss, before_wm_acc = _evaluate_model(
            config, base_model, test_loader, watermark_loader
        )
        for level in pruning_levels:
            model = copy.deepcopy(base_model)
            _apply_l1_pruning(model, level)
            train_loss = _finetune_on_adversaries(
                config, model, client_loaders, n_adversaries
            )
            clean_loss, clean_acc, wm_loss, wm_acc = _evaluate_model(
                config, model, test_loader, watermark_loader
            )
            row = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "attack": "finepruning",
                "repeat": repeat,
                "n_adversaries": n_adversaries,
                "pruning_level": level,
                "local_epochs": config["local_epochs"],
                "attack_lr": config["attack_lr"],
                "train_loss": train_loss,
                "before_test_accuracy": before_clean_acc,
                "before_watermark_accuracy": before_wm_acc,
                "test_accuracy": clean_acc,
                "test_loss": clean_loss,
                "watermark_accuracy": wm_acc,
                "watermark_loss": wm_loss,
                "model_file": model_file,
            }
            append_metrics_row(rows_path, row)
            print(
                "Finepruning "
                f"level={level:.1f}, repeat={repeat}: "
                f"clean {before_clean_acc:.2f}->{clean_acc:.2f}, "
                f"wm {before_wm_acc:.2f}->{wm_acc:.2f}"
            )


def run_attacks(config_overrides=None):
    config = CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)

    set_seed(int(config["seed"]))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_file = config.get("model_file") or _latest_model_file(RESULTS_DIR)
    print(f"Loading watermarked checkpoint: {model_file}")

    test_loader, watermark_loader, client_loaders = _prepare_data(config)
    run_finetuning_attack(config, model_file, test_loader, watermark_loader, client_loaders, RESULTS_DIR)
    run_pruning_attack(config, model_file, test_loader, watermark_loader, client_loaders, RESULTS_DIR)


if __name__ == "__main__":
    run_attacks()
