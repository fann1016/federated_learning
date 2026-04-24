import itertools

from run_cifar10 import run_experiment


FIGURE4_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
FIGURE45_ALPHAS = [0.05, 0.10, 0.20, 0.50, 1.0, 10.0, 100.0]
REPORTING_FRACTIONS = [0.05, 0.10, 0.20, 0.40]
LOCAL_EPOCHS = [1, 5]
SERVER_BETAS = [0.0, 0.7, 0.9, 0.97, 0.99, 0.997]
REPEATS = 5


def run_figure4_grid():
    for fraction, local_epochs, alpha, lr, repeat_id in itertools.product(
        REPORTING_FRACTIONS,
        LOCAL_EPOCHS,
        FIGURE45_ALPHAS,
        FIGURE4_LRS,
        range(REPEATS),
    ):
        summary = run_experiment({
            "fraction": fraction,
            "local_epochs": local_epochs,
            "dirichlet_alpha": alpha,
            "lr": lr,
            "server_momentum_beta": 0.0,
            "server_nesterov": False,
            "repeat_id": repeat_id,
        })
        print(summary)


def run_figure5_grid():
    for fraction, local_epochs, alpha, beta, repeat_id in itertools.product(
        [0.05, 0.40],
        [1, 5],
        FIGURE45_ALPHAS,
        SERVER_BETAS,
        range(REPEATS),
    ):
        summary = run_experiment({
            "fraction": fraction,
            "local_epochs": local_epochs,
            "dirichlet_alpha": alpha,
            "server_momentum_beta": beta,
            "server_nesterov": beta > 0,
            "repeat_id": repeat_id,
        })
        print(summary)


if __name__ == "__main__":
    # Choose the sweep block that matches the paper figure you want to reproduce.
    run_figure5_grid()
