import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.run_mnist import run_experiment


if __name__ == "__main__":
    run_experiment(
        {
            "model_name": "mnist_l5",
            "lr": 0.1,
            "batch_size": 50,
            "num_clients": 100,
            "fraction": 0.1,
            "local_epochs": 1,
            "rounds": 250,
            "iid": True,
            "experiment_group": "baseline",
            "output_subdir": "baseline_mnist_l5_fedavg_iid_seed0",
        }
    )
