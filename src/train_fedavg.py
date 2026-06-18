from __future__ import annotations

try:
    from . import config
    from .federated_core import FederatedRunConfig, run_federated_experiment
except ImportError:
    import config
    from federated_core import FederatedRunConfig, run_federated_experiment


def run_federated(
    method: str,
    sigma: float = -1.0,
    use_grl: bool = False,
    alpha_grl: float = 0.0,
    rounds: int | None = None,
    local_epochs: int | None = None,
    learning_rate: float | None = None,
    clip_norm: float | None = None,
    alpha_mode: str = "fixed",
    alpha_g: float = 0.0,
    seed: int = config.SEED,
    save_main_result: bool = True,
) -> dict:
    cfg = FederatedRunConfig(
        method=method,
        rounds=rounds or config.FEDERATED_ROUNDS,
        local_epochs=local_epochs or config.LOCAL_EPOCHS,
        learning_rate=learning_rate or config.LEARNING_RATE,
        sigma=sigma,
        clip_norm=clip_norm or config.CLIP_NORM,
        alpha_s=alpha_grl if use_grl else 0.0,
        alpha_g=alpha_g,
        alpha_mode=alpha_mode,
        seed=seed,
        save_main_result=save_main_result,
        fast_debug=config.FAST_DEBUG,
    )
    return run_federated_experiment(cfg)


if __name__ == "__main__":
    print(run_federated("FedAvg", sigma=-1.0, use_grl=False, alpha_grl=0.0))
