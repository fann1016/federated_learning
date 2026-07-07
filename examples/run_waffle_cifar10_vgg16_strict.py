import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.run_cifar10 import run_experiment


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_WATERMARK_DIR = os.path.join(
    WORKSPACE_DIR,
    "WAFFLE-main",
    "WAFFLE-main",
    "src",
    "data",
    "datasets",
    "CWAFFLE",
)


if __name__ == "__main__":
    run_experiment(
        {
            "model_name": "waffle_cifar_vgg16_strict",
            "vgg_pretrained": True,
            "cifar_transform": "waffle",
            "cifar_normalize": "imagenet",
            "input_size": 32,
            "lr": 0.0005,
            "batch_size": 64,
            "num_clients": 100,
            "fraction": 0.1,
            "local_epochs": 1,
            "rounds": 250,
            "eval_interval": 1,
            "data_path": os.path.join(BASE_DIR, "data_clean"),
            "partition_method": "iid",
            "server_momentum_beta": 0.0,
            "server_lr": 1.0,
            "server_nesterov": False,
            "client_nesterov": False,
            "weight_decay": 0.00005,
            "seed": 0,
            "experiment_group": "waffle_strict",
            "output_subdir": "waffle_strict_cifar10_vgg16_cwaffle_iid_seed0",
            "wandb_name": "cifar10-waffle-vgg16-strict",
            "watermark_enabled": True,
            "watermark_data_path": DEFAULT_WATERMARK_DIR,
            "watermark_batch_size": 50,
            "watermark_normalize": "imagenet",
            "watermark_lr": 0.0005,
            "watermark_pretrain_epochs": 30,
            "watermark_pretrain_stop_on_target": False,
            "watermark_retrain": True,
            "watermark_retrain_max_epochs": 100,
            "watermark_target_accuracy": 98.0,
        }
    )
