import os
import sys

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.run_mnist import run_experiment
from src.data.watermark import PatternWatermarkDataset


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_WATERMARK_DIR = os.path.join(
    WORKSPACE_DIR,
    "WAFFLE-main",
    "WAFFLE-main",
    "src",
    "data",
    "datasets",
    "MWAFFLE",
)


def build_watermark_loader(config):
    dataset = PatternWatermarkDataset(
        config.get("watermark_data_path", DEFAULT_WATERMARK_DIR),
        image_size=28,
        grayscale=True,
        normalize="waffle",
    )
    return DataLoader(
        dataset,
        batch_size=int(config.get("watermark_batch_size", 50)),
        shuffle=True,
    )


if __name__ == "__main__":
    run_experiment(
        {
            "model_name": "waffle_mnist_l5_strict",
            "loss_function": "nll",
            "mnist_normalize": "waffle",
            "lr": 0.1,
            "batch_size": 50,
            "num_clients": 100,
            "fraction": 0.1,
            "local_epochs": 1,
            "rounds": 250,
            "iid": True,
            "experiment_group": "waffle_strict",
            "output_subdir": "waffle_strict_mnist_l5_pattern_iid_seed0",
            "watermark_enabled": True,
            "watermark_loader_factory": build_watermark_loader,
            "watermark_lr": 0.1,
            "watermark_pretrain_epochs": 25,
            "watermark_pretrain_stop_on_target": False,
            "watermark_retrain": True,
            "watermark_retrain_max_epochs": 100,
            "watermark_target_accuracy": 98.0,
            "watermark_normalize": "waffle",
        }
    )
