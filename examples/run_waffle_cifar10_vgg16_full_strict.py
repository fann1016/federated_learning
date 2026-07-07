import os
import subprocess
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_step(script_name):
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MPLBACKEND", "Agg")
    script_path = os.path.join(BASE_DIR, "examples", script_name)
    print(f"\n=== Running {script_name} ===")
    subprocess.run([sys.executable, "-u", script_path], cwd=BASE_DIR, env=env, check=True)


if __name__ == "__main__":
    run_step("run_waffle_cifar10_vgg16_strict.py")
    run_step("run_waffle_cifar10_vgg16_attacks_strict.py")
