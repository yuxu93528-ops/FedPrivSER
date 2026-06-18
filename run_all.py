from __future__ import annotations

from src import config
from src.check_data import check_dataset
from src.extract_features import run_feature_extraction
from src.plot_results import main as plot_main
from src.privacy_attacks import run_attack
from src.train_centralized import run as run_centralized
from src.train_fedavg import run_federated
from src.train_fedavg_dp import run as run_fedavg_dp
from src.train_fedavg_grl import run as run_fedavg_grl
from src.train_fedavg_grl_dp import run as run_fedavg_grl_dp
from src.train_local_only import run as run_local_only


def main() -> None:
    config.ensure_directories()
    print("Checking dataset...")
    print(check_dataset())
    print("Extracting features...")
    run_feature_extraction(force=False)
    print("Running experiments...")
    run_centralized()
    run_local_only()
    run_federated("FedAvg", sigma=-1.0, use_grl=False, alpha_grl=0.0)
    for sigma in [0.0, 0.01, 0.05, 0.1, 0.2]:
        run_fedavg_dp(sigma=sigma)
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        run_fedavg_grl(alpha=alpha)
    run_fedavg_grl_dp(alpha=0.5, sigma=0.05)
    print("Running privacy attacks...")
    run_attack()
    print("Plotting results...")
    plot_main()
    print("All done.")


if __name__ == "__main__":
    main()
