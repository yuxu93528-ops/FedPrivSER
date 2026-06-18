from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import config
    from .federated_core import run_representation_attack_for_method
    from .train_fedavg import run_federated
except ImportError:
    import config
    from federated_core import run_representation_attack_for_method
    from train_fedavg import run_federated


def run() -> pd.DataFrame:
    settings = [(0.1, 0.05), (0.1, 0.1), (0.05, 0.1), (0.03, 0.1)]
    rows = []
    for alpha_s, alpha_g in settings:
        method = f"FedAvg+Dual-GRL(alpha_s={alpha_s},alpha_g={alpha_g})"
        result = run_federated(
            method=method,
            sigma=-1.0,
            use_grl=True,
            alpha_grl=alpha_s,
            alpha_g=alpha_g,
            save_main_result=False,
        )
        attack = run_representation_attack_for_method(method)
        rows.append({
            "alpha_s": alpha_s,
            "alpha_g": alpha_g,
            "accuracy": result["metrics"]["accuracy"],
            "uar": result["metrics"]["uar"],
            "macro_f1": result["metrics"]["macro_f1"],
            **{k: v for k, v in attack.items() if k != "method"},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "dual_grl_results.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frame["speaker_attack_acc"], frame["gender_attack_acc"], s=80)
    for _, row in frame.iterrows():
        ax.annotate(f"s={row['alpha_s']}, g={row['alpha_g']}", (row["speaker_attack_acc"], row["gender_attack_acc"]))
    ax.set_xlabel("Speaker attack acc")
    ax.set_ylabel("Gender attack acc")
    ax.set_title("Dual-GRL Privacy Utility")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "dual_grl_privacy_utility.png", dpi=200)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    print(run())
