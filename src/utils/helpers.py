import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

def get_next_run_number(results_path, dataset_name):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    files = os.listdir(results_path)
    run_numbers = []
    for f in files:
        if f.startswith(f"{dataset_name}_run") and f.endswith(".txt"):
            try:
                num = int(f.split("_run")[1].split("_")[0])
                run_numbers.append(num)
            except:
                continue
    return max(run_numbers) + 1 if run_numbers else 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def append_metrics_row(csv_path, row):
    parent_dir = os.path.dirname(csv_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def plot_data_distribution(dict_users, dataset, num_classes, results_path, filename):
    """
    Plots the data distribution across clients, similar to Figure 1 in the paper.
    """
    num_clients = len(dict_users)
    data_counts = np.zeros((num_clients, num_classes))
    
    for i, idxs in dict_users.items():
        for idx in idxs:
            label = dataset.targets[idx]
            data_counts[i][label] += 1
            
    plt.figure(figsize=(10, 6))
    classes = [str(i) for i in range(num_classes)]
    bottom = np.zeros(num_clients)
    
    for k in range(num_classes):
        plt.bar(range(num_clients), data_counts[:, k], bottom=bottom, label=classes[k])
        bottom += data_counts[:, k]
        
    plt.xlabel('Client ID')
    plt.ylabel('Number of Samples')
    plt.title('Data Distribution across Clients (Non-IID)')
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    plt.savefig(os.path.join(results_path, filename))
    plt.close()
    return data_counts
