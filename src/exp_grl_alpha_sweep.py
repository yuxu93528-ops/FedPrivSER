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
    alphas = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    rows = []
    for alpha in alphas:
        if alpha == 0:
            method = "GRL-AlphaSweep(alpha=0.0)"
            result = run_federated(method=method, sigma=-1.0, use_grl=False, alpha_grl=0.0, save_main_result=False)
        else:
            method = f"GRL-AlphaSweep(alpha={alpha})"
            result = run_federated(method=method, sigma=-1.0, use_grl=True, alpha_grl=alpha, save_main_result=False)
        attack = run_representation_attack_for_method(method)
        rows.append({
            "alpha": alpha,
            "accuracy": result["metrics"]["accuracy"],
            "uar": result["metrics"]["uar"],
            "macro_f1": result["metrics"]["macro_f1"],
            **{k: v for k, v in attack.items() if k != "method"},
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "grl_alpha_sweep.csv", index=False, encoding="utf-8-sig")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(frame["alpha"], frame["macro_f1"], marker="o", color="tab:blue", label="Emotion Macro-F1")
    ax2.plot(frame["alpha"], frame["speaker_attack_acc"], marker="s", color="tab:red", label="Speaker attack acc")
    ax2.plot(frame["alpha"], frame["gender_attack_acc"], marker="^", color="tab:green", label="Gender attack acc")
    ax1.set_xlabel("alpha")
    ax1.set_ylabel("Emotion Macro-F1", color="tab:blue")
    ax2.set_ylabel("Attack Accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "grl_alpha_privacy_utility.png", dpi=200)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    print(run())
