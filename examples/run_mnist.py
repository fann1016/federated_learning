import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")

from src.data.dataloader import DataLoaderManager
from src.models.factory import build_model
from src.federated.server import Server
from src.federated.client import Client
from src.utils.helpers import append_metrics_row, get_next_run_number

# Configuration for MNIST
CONFIG = {
    "num_clients": 100,
    "fraction": 0.1,
    "local_epochs": 1,
    "batch_size": 10,
    "lr": 0.01,
    "rounds": 250,
    "iid": True,
    "optimizer_momentum": 0.0,
    "weight_decay": 0.0,
    "server_momentum_beta": 0.0,
    "server_lr": 1.0,
    "server_nesterov": False,
    "client_nesterov": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "results_path": "D:\研究生学习\论文-联邦学习\联邦学习代码",
    "data_path": "C:/Users/Fnn/CascadeProjects/federated_learning_cifar10_mnist/data",
    "model_name": "simple_mnist_cnn",
    "num_classes": 10,
    "wandb_project": "federated-learning-mnist",
    "wandb_name": "mnist-fedavg",
    "seed": 0,
    "experiment_group": "baseline",
    "show_plots": False,
}

CONFIG["results_path"] = BASE_DIR
CONFIG["data_path"] = DATA_DIR


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


def build_results_dir(config):
    model_name = str(config.get("model_name", "model")).lower()
    subdir = config.get(
        "output_subdir",
        f"{config.get('experiment_group', 'baseline')}_mnist_{model_name}_fedavg_seed{config.get('seed', 0)}",
    )
    return os.path.join(config["results_path"], "experiments", subdir)


def run_experiment(config_overrides=None):
    config = CONFIG.copy()
    if config_overrides:
        config.update(config_overrides)

    print("\n--- Running Federated Learning on MNIST ---")
    
    # Initialize WandB
    init_wandb(f"{config['wandb_name']}_{datetime.now().strftime('%m%d_%H%M')}", config)
    
    results_dir = build_results_dir(config)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    run_num = get_next_run_number(results_dir, "FedAvg")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(results_dir, f"FedAvg_run{run_num}_{timestamp}.txt")

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write(f"MNIST Federated Learning Run #{run_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Config: {config}\n\n")

    # Data management
    data_manager = DataLoaderManager(config)
    train_dataset, test_dataset = data_manager.load_data(dataset_name='MNIST')
    user_groups = data_manager.partition_data(method="iid" if config["iid"] else "dirichlet")
    
    # Initialize Server and Clients
    global_model = build_model(config["model_name"], num_classes=config["num_classes"])
    server = Server(global_model, config)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    clients = [Client(train_dataset, user_groups[i], config) for i in range(config["num_clients"])]
    
    watermark_loader = None
    watermark_acc_history = []
    watermark_retrain_history = []
    if config.get("watermark_enabled", False):
        watermark_factory = config.get("watermark_loader_factory")
        if watermark_factory is None:
            raise ValueError("watermark_enabled=True requires watermark_loader_factory.")
        watermark_loader = watermark_factory(config)
        pretrain_epochs = int(config.get("watermark_pretrain_epochs", 0))
        if pretrain_epochs > 0:
            wm_epochs, wm_loss, wm_acc = server.retrain_on_loader(
                watermark_loader,
                lr=float(config.get("watermark_lr", config["lr"])),
                max_epochs=pretrain_epochs,
                target_accuracy=float(config.get("watermark_target_accuracy", 98.0)),
                stop_on_target=bool(config.get("watermark_pretrain_stop_on_target", True)),
            )
            msg = (
                f"Watermark pretrain - Epochs: {wm_epochs}/{pretrain_epochs}, "
                f"Watermark Loss: {wm_loss:.4f}, Watermark Acc: {wm_acc:.2f}%"
            )
            print(msg)
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    loss_history, acc_history = [], []
    for round_idx in range(config["rounds"]):
        local_weights, local_losses, local_sizes = [], [], []
        m = max(int(config["fraction"] * config["num_clients"]), 1)
        idxs_users = np.random.choice(range(config["num_clients"]), m, replace=False)
        
        for idx in idxs_users:
            print(f"  Round {round_idx+1}, Training Client {idx}...", end="\r")
            w, loss, size = clients[idx].train(copy.deepcopy(server.global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)
            local_sizes.append(size)
        
        # Aggregate local weights to global model
        server.aggregate(local_weights, local_sizes)

        wm_loss = None
        wm_acc = None
        wm_epochs = 0
        if watermark_loader is not None:
            wm_loss, wm_acc = server.evaluate_loader(watermark_loader)
            if config.get("watermark_retrain", False):
                wm_epochs, wm_loss, wm_acc = server.retrain_on_loader(
                    watermark_loader,
                    lr=float(config.get("watermark_lr", config["lr"])),
                    max_epochs=int(config.get("watermark_retrain_max_epochs", 100)),
                    target_accuracy=float(config.get("watermark_target_accuracy", 98.0)),
                    stop_on_target=True,
                )
            watermark_acc_history.append(wm_acc)
            watermark_retrain_history.append(wm_epochs)
        
        test_loss, accuracy = server.evaluate(test_loader)
        loss_history.append(test_loss)
        acc_history.append(accuracy)
        
        avg_local_loss = sum(local_losses)/len(local_losses)
        
        # Log to WandB
        wandb_row = {
            "round": round_idx + 1,
            "avg_local_loss": avg_local_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
        }
        if watermark_loader is not None:
            wandb_row.update(
                {
                    "watermark_loss": wm_loss,
                    "watermark_accuracy": wm_acc,
                    "watermark_retrain_epochs": wm_epochs,
                }
            )
        wandb.log(wandb_row)
        
        log_msg = f"Round {round_idx+1}/{config['rounds']} - Avg Local Loss: {avg_local_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.2f}%"
        if watermark_loader is not None:
            log_msg += (
                f", Watermark Loss: {wm_loss:.4f}, Watermark Acc: {wm_acc:.2f}%, "
                f"Watermark Retrain Epochs: {wm_epochs}"
            )
        print(log_msg)
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    # Plotting
    if watermark_loader is not None:
        plt.figure(figsize=(12, 8))
        rows, cols = 2, 2
    else:
        plt.figure(figsize=(12, 4))
        rows, cols = 1, 2

    plt.subplot(rows, cols, 1)
    plt.plot(range(1, config["rounds"]+1), loss_history)
    plt.title(f'Test Loss - MNIST (Run {run_num})')
    plt.subplot(rows, cols, 2)
    plt.plot(range(1, config["rounds"]+1), acc_history)
    plt.title(f'Test Accuracy - MNIST (Run {run_num})')
    if watermark_loader is not None:
        plt.subplot(rows, cols, 3)
        plt.plot(range(1, config["rounds"] + 1), watermark_acc_history)
        plt.title(f'Watermark Accuracy - MNIST (Run {run_num})')
        plt.subplot(rows, cols, 4)
        plt.plot(range(1, config["rounds"] + 1), watermark_retrain_history)
        plt.title(f'Watermark Retrain Epochs - MNIST (Run {run_num})')
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"FedAvg_run{run_num}_{timestamp}_results.png")
    plt.savefig(plot_path)
    if config.get("show_plots", False):
        plt.show()
    plt.close()

    summary = {
        "timestamp": timestamp,
        "algorithm": "FedAvg",
        "dataset": "MNIST",
        "model": config["model_name"],
        "seed": config.get("seed", 0),
        "num_clients": config["num_clients"],
        "fraction": config["fraction"],
        "local_epochs": config["local_epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "rounds": config["rounds"],
        "best_test_accuracy": max(acc_history) if acc_history else 0.0,
        "final_test_accuracy": acc_history[-1] if acc_history else 0.0,
        "final_test_loss": loss_history[-1] if loss_history else 0.0,
        "log_file": log_filename,
        "plot_file": plot_path,
    }
    if watermark_loader is not None:
        summary.update(
            {
                "watermark_set_size": len(watermark_loader.dataset),
                "best_watermark_accuracy": max(watermark_acc_history) if watermark_acc_history else 0.0,
                "final_watermark_accuracy": watermark_acc_history[-1] if watermark_acc_history else 0.0,
                "total_watermark_retrain_epochs": sum(watermark_retrain_history),
            }
        )
    append_metrics_row(os.path.join(results_dir, "summary.csv"), summary)
    
    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    run_experiment()
