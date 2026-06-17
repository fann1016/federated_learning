"""Unified federated training loop driven by per-paper configs."""

import copy
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.config.paper_configs import get_paper_config
from src.data.dataloader import DataLoaderManager
from src.federated.client import Client
from src.federated.server import Server
from src.models.factory import build_model
from src.utils.helpers import append_metrics_row, get_next_run_number, set_seed


def _init_wandb(run_name, config):
    api_key = os.getenv("WANDB_API_KEY")
    mode = os.getenv("WANDB_MODE", "online" if api_key else "disabled")
    if mode != "online":
        print(f"W&B logging mode: {mode}")
    wandb.init(project=config["wandb_project"], name=run_name, config=config, mode=mode)


def _select_clients(config, round_idx):
    num_clients = config["num_clients"]
    seed = int(config["seed"]) + round_idx
    rng = np.random.RandomState(seed)

    if config.get("participation_mode") == "probabilistic":
        prob = float(config["participation_prob"])
        selected = np.where(rng.rand(num_clients) < prob)[0]
        if len(selected) == 0:
            selected = rng.choice(num_clients, 1, replace=False)
        return selected

    return rng.choice(num_clients, config["clients_per_round"], replace=False)


def _eval_batch_size(config):
    if config.get("batch_size") is not None:
        return int(config["batch_size"])
    return int(config.get("eval_batch_size", 128))


def _partition_kwargs(config):
    method = config["partition_method"]
    if method == "dirichlet":
        return {"method": method, "alpha": float(config.get("dirichlet_alpha", 0.6))}
    if method == "emnist_similarity":
        return {
            "method": method,
            "similarity_frac": float(config.get("similarity_frac", 0.0)),
        }
    return {"method": method}


def _train_round_fedcm(server, clients, idxs_users, config, round_idx):
    client_config = {**config, "_current_round": round_idx}
    local_weights, local_losses, local_sizes, local_steps = [], [], [], []
    global_state = copy.deepcopy(server.global_model.state_dict())
    server_delta = server.get_server_delta()

    for idx in idxs_users:
        client_model = copy.deepcopy(server.global_model)
        w, loss, size, steps = clients[idx].train(
            client_model,
            global_state=global_state,
            server_delta=server_delta,
            round_idx=round_idx,
            client_config=client_config,
        )
        local_weights.append(copy.deepcopy(w))
        local_losses.append(loss)
        local_sizes.append(size)
        local_steps.append(steps)

    server.aggregate_fedcm(local_weights, local_sizes, local_steps, global_state, round_idx)
    return local_losses


def _train_round_fedspeed(server, clients, idxs_users, config, round_idx):
    client_config = {**config, "_current_round": round_idx}
    local_losses, local_weights = [], []
    global_state = copy.deepcopy(server.global_model.state_dict())
    global_vec = server.global_param_vector()

    for idx in idxs_users:
        dual_correction = server.get_fedspeed_dual_correction(idx, global_vec)
        client_model = copy.deepcopy(server.global_model)
        w, loss, _size, local_update = clients[idx].train(
            client_model,
            global_state=global_state,
            dual_correction=dual_correction,
            round_idx=round_idx,
            client_config=client_config,
        )
        server.update_fedspeed_h(idx, local_update)
        local_weights.append(copy.deepcopy(w))
        local_losses.append(loss)

    server.aggregate_fedspeed(local_weights)
    return local_losses


def _train_round_scaffold(server, clients, idxs_users, config, round_idx):
    client_config = {**config, "_current_round": round_idx}
    local_weights, local_losses, local_sizes, control_deltas = [], [], [], []
    global_state = copy.deepcopy(server.global_model.state_dict())
    global_control = server.get_global_control()

    for idx in idxs_users:
        client_model = copy.deepcopy(server.global_model)
        w, loss, size, delta_c = clients[idx].train(
            client_model,
            global_state=global_state,
            global_control=global_control,
            round_idx=round_idx,
            client_config=client_config,
        )
        local_weights.append(copy.deepcopy(w))
        local_losses.append(loss)
        local_sizes.append(size)
        control_deltas.append(delta_c)

    server.aggregate_scaffold(
        local_weights, local_sizes, control_deltas, config["num_clients"]
    )
    return local_losses


def _train_round_default(server, clients, idxs_users, config, round_idx):
    client_config = {**config, "_current_round": round_idx}
    local_weights, local_losses, local_sizes = [], [], []
    global_state = copy.deepcopy(server.global_model.state_dict())
    algorithm = str(config.get("algorithm", "")).lower()
    selected_users = list(idxs_users)

    drop_percent = float(config.get("drop_percent", 0.0))
    if drop_percent > 0.0 and selected_users:
        rng = np.random.RandomState(round_idx)
        active_count = max(
            1, int(round(len(selected_users) * (1.0 - drop_percent)))
        )
        active_users = set(
            rng.choice(selected_users, active_count, replace=False).tolist()
        )
    else:
        rng = None
        active_users = set(selected_users)

    for idx in selected_users:
        client_model = copy.deepcopy(server.global_model)
        per_client_config = client_config
        if (
            algorithm == "fedprox"
            and idx not in active_users
            and int(config.get("local_epochs", 1)) > 1
        ):
            per_client_config = {**client_config}
            per_client_config["local_epochs"] = int(
                rng.randint(1, int(config["local_epochs"]))
            )
        elif algorithm == "fedavg" and idx not in active_users:
            continue

        w, loss, size = clients[idx].train(
            client_model,
            global_state=global_state,
            round_idx=round_idx,
            client_config=per_client_config,
        )
        local_weights.append(copy.deepcopy(w))
        local_losses.append(loss)
        local_sizes.append(size)

    server.aggregate(local_weights, local_sizes)
    return local_losses


def _train_round(config, server, clients, idxs_users, round_idx):
    algo = config["algorithm"].lower()
    if algo == "fedcm":
        return _train_round_fedcm(server, clients, idxs_users, config, round_idx)
    if algo == "scaffold":
        return _train_round_scaffold(server, clients, idxs_users, config, round_idx)
    if algo == "fedspeed":
        return _train_round_fedspeed(server, clients, idxs_users, config, round_idx)
    return _train_round_default(server, clients, idxs_users, config, round_idx)


def run_paper_experiment(algorithm, overrides=None):
    config = get_paper_config(algorithm, overrides)
    set_seed(int(config["seed"]))

    algo_name = config["algorithm"]
    if algo_name.lower() == "fedprox" and float(config.get("mu", 0.0)) <= 0.0:
        algo_name = "FedAvg"

    print(f"\n--- Reproducing {algo_name} ---")
    print(f"Paper: {config.get('paper', 'N/A')}")
    print(f"Dataset: {config['dataset_name']} | Model: {config['model_name']}")

    results_dir = os.path.join(
        config["results_path"], "experiments", config["experiments_subdir"]
    )
    os.makedirs(results_dir, exist_ok=True)

    run_num = get_next_run_number(results_dir, config["run_prefix"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run_name = f"{algo_name}_run{run_num}_{timestamp}"
    log_filename = os.path.join(results_dir, f"{algo_name}_run{run_num}_{timestamp}.txt")

    _init_wandb(wandb_run_name, config)

    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ALGORITHM: {algo_name}\n")
        f.write(f"PAPER: {config.get('paper', '')}\n")
        f.write(f"CONFIG: {config}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

    data_manager = DataLoaderManager(config)
    train_dataset, test_dataset = data_manager.load_data(dataset_name=config["dataset_name"])
    user_groups = data_manager.partition_data(**_partition_kwargs(config))
    train_dataset = data_manager.train_dataset
    test_dataset = data_manager.test_dataset

    global_model = build_model(config["model_name"], num_classes=config["num_classes"])
    server = Server(global_model, config)
    test_loader = DataLoader(
        test_dataset, batch_size=_eval_batch_size(config), shuffle=False
    )
    clients = [
        Client(train_dataset, user_groups[i], config) for i in range(config["num_clients"])
    ]

    loss_history, acc_history, eval_rounds = [], [], []
    best_accuracy = 0.0
    target_accuracy = config.get("target_accuracy")
    rounds_to_target = None

    for round_idx in range(config["rounds"]):
        idxs_users = _select_clients(config, round_idx)
        local_losses = _train_round(config, server, clients, idxs_users, round_idx)

        avg_local_loss = sum(local_losses) / len(local_losses)
        wandb.log({
            "round": round_idx + 1,
            "avg_local_loss": avg_local_loss,
            "num_participants": len(idxs_users),
        })

        if (round_idx + 1) % config.get("eval_interval", 1) == 0 or (round_idx + 1) == config["rounds"]:
            test_loss, accuracy = server.evaluate(test_loader)
            best_accuracy = max(best_accuracy, accuracy)
            if (
                rounds_to_target is None
                and target_accuracy is not None
                and accuracy >= float(target_accuracy)
            ):
                rounds_to_target = round_idx + 1
            loss_history.append(test_loss)
            acc_history.append(accuracy)
            eval_rounds.append(round_idx + 1)

            wandb.log({
                "round": round_idx + 1,
                "test_loss": test_loss,
                "test_accuracy": accuracy,
                "best_test_accuracy": best_accuracy,
            })

            log_msg = (
                f"[{algo_name}] Round {round_idx + 1}/{config['rounds']} - "
                f"Participants: {len(idxs_users)}, "
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
    plt.xlabel("Round")
    plt.subplot(1, 2, 2)
    plt.plot(eval_rounds, acc_history)
    plt.title(f"{algo_name} Test Accuracy (Run {run_num})")
    plt.xlabel("Round")
    plot_path = os.path.join(results_dir, f"{algo_name}_run{run_num}_{timestamp}_results.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    if config.get("show_plots", False):
        plt.show()
    plt.close()

    summary = {
        "timestamp": timestamp,
        "algorithm": algo_name,
        "paper": config.get("paper", ""),
        "dataset": config["dataset_name"],
        "model": config["model_name"],
        "seed": config["seed"],
        "num_clients": config["num_clients"],
        "participation_mode": config.get("participation_mode"),
        "participation_prob": config.get("participation_prob"),
        "drop_percent": config.get("drop_percent"),
        "clients_per_round": config.get("clients_per_round"),
        "local_epochs": config.get("local_epochs"),
        "local_steps": config.get("local_steps"),
        "batch_size": config.get("batch_size"),
        "batch_frac": config.get("batch_frac"),
        "similarity_frac": config.get("similarity_frac"),
        "lr": config["lr"],
        "lr_decay": config.get("lr_decay", 1.0),
        "rounds": config["rounds"],
        "best_test_accuracy": best_accuracy,
        "final_test_accuracy": acc_history[-1] if acc_history else 0.0,
        "final_test_loss": loss_history[-1] if loss_history else 0.0,
        "rounds_to_target_accuracy": rounds_to_target,
        "target_accuracy": target_accuracy,
        "log_file": log_filename,
        "plot_file": plot_path,
    }
    if "mu" in config:
        summary["mu"] = config["mu"]
    if "fedcm_alpha" in config:
        summary["fedcm_alpha"] = config["fedcm_alpha"]
    if "lamb" in config:
        summary["lamb"] = config["lamb"]
    if "sam_rho" in config:
        summary["sam_rho"] = config["sam_rho"]

    append_metrics_row(config["summary_csv"], summary)
    wandb.finish()
    return summary
