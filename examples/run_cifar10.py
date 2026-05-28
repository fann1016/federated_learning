import copy
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb 
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")

from src.data.dataloader import DataLoaderManager
from src.federated.client import Client
from src.federated.server import Server
from src.models.cnn import SimpleCIFAR_CNN
from src.utils.helpers import (
    append_metrics_row,
    get_next_run_number,
    plot_data_distribution,
    set_seed,
)

BASE_CONFIG = {
    "num_clients": 100,
    "fraction": 0.1,
    "local_epochs": 1,
    "batch_size": 64,
    "lr": 0.01,
    "optimizer_momentum": 0.0,
    "weight_decay": 0.0004,
    "rounds": 1000,
    "eval_interval": 10,
    "partition_method": "dirichlet",
    "dirichlet_alpha": 0.5,
    "server_momentum_beta": 0.9,
    "server_lr": 1.0,  # 服务器端更新步长
    "server_nesterov": True, # 服务器端是否使用Nesterov动量
    "client_nesterov": False, 
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_path": BASE_DIR,
    "data_path": DATA_DIR,
    "wandb_project": "federated-learning-cifar10",
    "wandb_name": "cifar10-fedavgm-paper-aligned",
    "seed": 2024,  # 随机种子
    "repeat_id": 0, # 实验重复ID
    "show_plots": False,
    "summary_csv": os.path.join(BASE_DIR, "experiments", "cifar10_summaries.csv"),
}

CONFIG = copy.deepcopy(BASE_CONFIG)


def build_config(config_overrides=None):
    config = copy.deepcopy(BASE_CONFIG)
    if config_overrides:
        config.update(config_overrides)
    return config


def init_wandb(run_name, config):
    api_key = os.getenv("WANDB_API_KEY")
    mode = os.getenv("WANDB_MODE", "online" if api_key else "disabled")
    if mode != "online":
        print(f"W&B logging mode: {mode}")

    wandb.init(
        project=config["wandb_project"],
        name=run_name,
        config=config,
        mode=mode
    )


def run_experiment(config_overrides=None):
    config = build_config(config_overrides)
    set_seed(int(config["seed"]) + int(config.get("repeat_id", 0)))

    print("\n--- Running Federated Learning Example ---")

    algo_name = "FedAvg" if config["server_momentum_beta"] <= 0 else "FedAvgM"
    base_results_path = "baselines" if config["server_momentum_beta"] <= 0 else "experiments"
    results_dir = os.path.join(config["results_path"], base_results_path)
    os.makedirs(results_dir, exist_ok=True)

    run_num = get_next_run_number(results_dir, "cifar10")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = (
        f"{algo_name}_a{config['dirichlet_alpha']}_c{config['fraction']}_"
        f"e{config['local_epochs']}_b{config['server_momentum_beta']}_"
        f"s{int(config['seed']) + int(config.get('repeat_id', 0))}_{datetime.now().strftime('%m%d_%H%M')}"
    )
    log_filename = os.path.join(results_dir, f"{algo_name}_run{run_num}_{timestamp}.txt")

    init_wandb(wandb_run_name, config)

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"ALGORITHM: {algo_name}\n")
        f.write(f"CONFIG SETTINGS: {config}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write("=" * 50 + "\n\n")

    data_manager = DataLoaderManager(config)
    train_dataset, test_dataset = data_manager.load_data(dataset_name="CIFAR10")
    
    # 实现狄利克雷分布的核心
    user_groups = data_manager.partition_data(
        method=config["partition_method"],
        alpha=config.get("dirichlet_alpha", 0.5)
    )

    plot_data_distribution(
        user_groups,
        train_dataset,
        num_classes=10,
        results_path=config["results_path"],
        filename=f"cifar10_distribution_{timestamp}.png"
    )

    global_model = SimpleCIFAR_CNN()
    server = Server(global_model, config)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    clients = [Client(train_dataset, user_groups[i], config) for i in range(config["num_clients"])]

    loss_history = []
    acc_history = []
    eval_rounds = []
    best_accuracy = 0.0
    for round_idx in range(config["rounds"]):
        local_weights, local_losses, local_sizes = [], [], []
        m = max(int(config["fraction"] * config["num_clients"]), 1)
        idxs_users = np.random.choice(range(config["num_clients"]), m, replace=False)

        for idx in idxs_users:
            print(f"  Round {round_idx + 1}, Training Client {idx}...")
            w, loss, size = clients[idx].train(copy.deepcopy(server.global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)
            local_sizes.append(size)

        server.aggregate(local_weights, local_sizes)

        avg_local_loss = sum(local_losses) / len(local_losses)
        wandb.log({
            "round": round_idx + 1,
            "avg_local_loss": avg_local_loss,
            "num_participants": len(idxs_users),
        })

        if (round_idx + 1) % config.get("eval_interval", 1) == 0 or (round_idx + 1) == config["rounds"]:
            test_loss, accuracy = server.evaluate(test_loader)
            best_accuracy = max(best_accuracy, accuracy)
            loss_history.append(test_loss)
            acc_history.append(accuracy)
            eval_rounds.append(round_idx + 1)

            wandb.log({
                "round": round_idx + 1,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "best_test_accuracy": best_accuracy,
                "effective_lr": config["lr"] / max(1e-12, (1.0 - config["server_momentum_beta"])),
            })

            log_msg = (
                f"[{algo_name}] Round {round_idx + 1}/{config['rounds']} - "
                f"Avg Local Loss: {avg_local_loss:.4f}, Test Loss: {test_loss:.4f}, "
                f"Test Acc: {accuracy:.2f}%"
            )
            print(log_msg)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(log_msg + "\n")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(eval_rounds, loss_history)
    plt.title(f"{algo_name} Test Loss (Run {run_num})")
    plt.subplot(1, 2, 2)
    plt.plot(eval_rounds, acc_history)
    plt.title(f"{algo_name} Test Accuracy (Run {run_num})")
    plot_path = os.path.join(results_dir, f"{algo_name}_run{run_num}_{timestamp}_results.png")
    plt.savefig(plot_path)
    if config.get("show_plots", False):
        plt.show()
    plt.close()

    summary = {
        "timestamp": timestamp,
        "algorithm": algo_name,
        "seed": int(config["seed"]) + int(config.get("repeat_id", 0)),
        "num_clients": config["num_clients"],
        "fraction": config["fraction"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "server_momentum_beta": config["server_momentum_beta"],
        "effective_lr": config["lr"] / max(1e-12, (1.0 - config["server_momentum_beta"])),
        "alpha": config["dirichlet_alpha"],
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": acc_history[-1] if acc_history else 0.0,
        "final_test_loss": loss_history[-1] if loss_history else 0.0,
        "log_file": log_filename,
        "plot_file": plot_path,
    }
    append_metrics_row(config["summary_csv"], summary)

    wandb.finish()
    return summary


if __name__ == "__main__":
    run_experiment()
