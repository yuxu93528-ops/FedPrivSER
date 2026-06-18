from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import config
    from .train_fedavg import run_federated
except ImportError:
    import config
    from train_fedavg import run_federated


def run() -> pd.DataFrame:
    rows = []
    histories = []
    rounds_list = [30, 50, 100]
    local_epochs_list = [1, 2, 5]
    learning_rates = [1e-3, 5e-4, 1e-4]
    for rounds in rounds_list:
        for local_epochs in local_epochs_list:
            for lr in learning_rates:
                method = f"FedAvg-Convergence(r={rounds},e={local_epochs},lr={lr})"
                result = run_federated(
                    method=method,
                    sigma=-1.0,
                    use_grl=False,
                    alpha_grl=0.0,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    learning_rate=lr,
                    save_main_result=False,
                )
                rows.append({
                    "rounds": rounds,
                    "local_epochs": local_epochs,
                    "learning_rate": lr,
                    "accuracy": result["metrics"]["accuracy"],
                    "uar": result["metrics"]["uar"],
                    "macro_f1": result["metrics"]["macro_f1"],
                    "training_time": result["training_time"],
                })
                for item in result["history"]:
                    histories.append({
                        "method": method,
                        "rounds": rounds,
                        "local_epochs": local_epochs,
                        "learning_rate": lr,
                        **item,
                    })
    frame = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    frame.to_csv(config.TABLE_DIR / "fedavg_convergence_results.csv", index=False, encoding="utf-8-sig")

    hist_df = pd.DataFrame(histories)
    best = frame.iloc[0]
    best_history = hist_df[
        (hist_df["rounds"] == best["rounds"]) &
        (hist_df["local_epochs"] == best["local_epochs"]) &
        (hist_df["learning_rate"] == best["learning_rate"])
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(best_history["round"], best_history["macro_f1"], marker="o")
    ax.set_title("FedAvg Convergence Curve (Best Setting)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Macro-F1")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "fedavg_convergence_curve.png", dpi=200)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    print(run())
