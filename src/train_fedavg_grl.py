from __future__ import annotations

try:
    from . import config
    from .train_fedavg import run_federated
except ImportError:
    import config
    from train_fedavg import run_federated


def run(alpha: float = config.ALPHA_GRL) -> dict:
    return run_federated(method=f"FedAvg+GRL(alpha={alpha})", sigma=-1.0, use_grl=True, alpha_grl=alpha)


if __name__ == "__main__":
    print(run())
