from __future__ import annotations

try:
    from . import config
    from .train_fedavg import run_federated
except ImportError:
    import config
    from train_fedavg import run_federated


def run(sigma: float = config.SIGMA) -> dict:
    return run_federated(method=f"FedAvg+DP(sigma={sigma})", sigma=sigma, use_grl=False, alpha_grl=0.0)


if __name__ == "__main__":
    print(run())
