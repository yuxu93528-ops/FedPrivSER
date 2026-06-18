from __future__ import annotations

try:
    from .dataset import prepare_data
    from .train_common import run_single_training
except ImportError:
    from dataset import prepare_data
    from train_common import run_single_training


def run() -> dict:
    prepared = prepare_data()
    return run_single_training(method="Centralized", prepared=prepared, epochs=50)


if __name__ == "__main__":
    print(run())
