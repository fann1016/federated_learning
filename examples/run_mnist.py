import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import wandb

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, "data")

from src.data.dataloader import DataLoaderManager
from src.models.cnn import SimpleMNIST_CNN
from src.federated.server import Server
from src.federated.client import Client
from src.utils.helpers import get_next_run_number

# Configuration for MNIST
CONFIG = {
    "num_clients": 10,
    "fraction": 0.1,
    "local_epochs": 10,
    "batch_size": 10,
    "lr": 0.01,
    "rounds": 10,
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
    "wandb_project": "federated-learning-mnist",
    "wandb_name": "mnist-fedavg"
}

CONFIG["results_path"] = BASE_DIR
CONFIG["data_path"] = DATA_DIR


def init_wandb(run_name):
    api_key = os.getenv("WANDB_API_KEY")
    mode = os.getenv("WANDB_MODE", "online" if api_key else "disabled")
    if mode != "online":
        print(f"W&B logging mode: {mode}")

    wandb.init(
        project=CONFIG["wandb_project"],
        name=run_name,
        config=CONFIG,
        mode=mode
    )

def run_experiment():
    print("\n--- Running Federated Learning on MNIST ---")
    
    # Initialize WandB
    init_wandb(f"{CONFIG['wandb_name']}_{datetime.now().strftime('%m%d_%H%M')}")
    
    run_num = get_next_run_number(CONFIG["results_path"], "mnist")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(CONFIG["results_path"], f"mnist_run{run_num}_{timestamp}.txt")
    
    if not os.path.exists(CONFIG["results_path"]):
        os.makedirs(CONFIG["results_path"])

    with open(log_filename, "w") as f:
        f.write(f"MNIST Federated Learning Run #{run_num}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Config: {CONFIG}\n\n")

    # Data management
    data_manager = DataLoaderManager(CONFIG)
    train_dataset, test_dataset = data_manager.load_data(dataset_name='MNIST')
    user_groups = data_manager.partition_data(method="iid" if CONFIG["iid"] else "dirichlet")
    
    # Initialize Server and Clients
    global_model = SimpleMNIST_CNN()
    server = Server(global_model, CONFIG)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    clients = [Client(train_dataset, user_groups[i], CONFIG) for i in range(CONFIG["num_clients"])]
    
    loss_history, acc_history = [], []
    for round_idx in range(CONFIG["rounds"]):
        local_weights, local_losses, local_sizes = [], [], []
        m = max(int(CONFIG["fraction"] * CONFIG["num_clients"]), 1)
        idxs_users = np.random.choice(range(CONFIG["num_clients"]), m, replace=False)
        
        for idx in idxs_users:
            print(f"  Round {round_idx+1}, Training Client {idx}...", end="\r")
            w, loss, size = clients[idx].train(copy.deepcopy(server.global_model))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(loss)
            local_sizes.append(size)
        
        # Aggregate local weights to global model
        server.aggregate(local_weights, local_sizes)
        
        test_loss, accuracy = server.evaluate(test_loader)
        loss_history.append(test_loss)
        acc_history.append(accuracy)
        
        avg_local_loss = sum(local_losses)/len(local_losses)
        
        # Log to WandB
        wandb.log({
            "round": round_idx + 1,
            "avg_local_loss": avg_local_loss,
            "test_loss": test_loss,
            "test_accuracy": accuracy
        })
        
        log_msg = f"Round {round_idx+1}/{CONFIG['rounds']} - Avg Local Loss: {avg_local_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {accuracy:.2f}%"
        print(log_msg)
        with open(log_filename, "a") as f:
            f.write(log_msg + "\n")

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, CONFIG["rounds"]+1), loss_history)
    plt.title(f'Test Loss - MNIST (Run {run_num})')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, CONFIG["rounds"]+1), acc_history)
    plt.title(f'Test Accuracy - MNIST (Run {run_num})')
    plt.savefig(os.path.join(CONFIG["results_path"], f"mnist_run{run_num}_{timestamp}_results.png"))
    plt.show()
    
    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    run_experiment()
