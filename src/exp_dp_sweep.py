from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import config
    from .federated_core import run_representation_attack_for_method, run_update_attack_for_method
    from .train_fedavg import run_federated
except ImportError:
    import config
    from federated_core import run_representation_attack_for_method, run_update_attack_for_method
    from train_fedavg import run_federated


def run() -> pd.DataFrame:
    sigmas = [0.000, 0.001, 0.003, 0.005, 0.010, 0.020, 0.050, 0.100]
    clip_norms = [0.1, 0.5, 1.0, 2.0, 5.0]
    rows = []
    for sigma in sigmas:
        for clip_norm in clip_norms:
            method = f"DP-Sweep(sigma={sigma},clip={clip_norm})"
            result = run_federated(
                method=method,
                sigma=sigma,
                use_grl=False,
                alpha_grl=0.0,
                clip_norm=clip_norm,
                save_main_result=False,
            )
            rep = run_representation_attack_for_method(method)
            upd = run_update_attack_for_method(method, attack_model="logreg")
            history_df = pd.read_csv(config.LOG_DIR / f"{method}_curve.csv")
            rows.append({
                "sigma": sigma,
                "clip_norm": clip_norm,
                "accuracy": result["metrics"]["accuracy"],
                "uar": result["metrics"]["uar"],
                "macro_f1": result["metrics"]["macro_f1"],
                "representation_speaker_attack_acc": rep["speaker_attack_acc"],
                "representation_gender_attack_acc": rep["gender_attack_acc"],
                "update_speaker_attack_acc": upd["speaker_attack_acc"],
                "update_gender_attack_acc": upd["gender_attack_acc"],
                "raw_update_norm_mean": history_df["raw_update_norm_mean"].iloc[-1],
                "raw_update_norm_median": history_df["raw_update_norm_median"].iloc[-1],
                "clipped_ratio": history_df["clipped_ratio"].iloc[-1],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "dp_sweep_results.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(frame["macro_f1"], frame["update_speaker_attack_acc"], c=frame["sigma"], cmap="viridis")
    ax.set_xlabel("Emotion Macro-F1")
    ax.set_ylabel("Update Speaker Attack Acc")
    ax.set_title("DP Sweep Privacy-Utility")
    fig.colorbar(scatter, label="sigma")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "dp_sweep_privacy_utility.png", dpi=200)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    print(run())
