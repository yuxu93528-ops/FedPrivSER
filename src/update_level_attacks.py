from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

try:
    from . import config
    from .federated_core import run_update_attack_for_method
except ImportError:
    import config
    from federated_core import run_update_attack_for_method


def _available_update_methods() -> list[str]:
    return [path.stem[:-8] for path in config.LOG_DIR.glob("*_updates.csv")]


def _resolve_methods() -> list[tuple[str, str]]:
    available = set(_available_update_methods())
    resolved: list[tuple[str, str]] = []

    if "FedAvg" in available:
        resolved.append(("FedAvg", "FedAvg"))
    elif "FedAvg(seed=42)" in available:
        resolved.append(("FedAvg", "FedAvg(seed=42)"))

    dp_sweep_path = config.TABLE_DIR / "dp_sweep_results.csv"
    if dp_sweep_path.exists():
        dp_df = pd.read_csv(dp_sweep_path)
        if not dp_df.empty:
            best_dp = dp_df.sort_values("macro_f1", ascending=False).iloc[0]
            dp_method = f"DP-Sweep(sigma={best_dp['sigma']},clip={best_dp['clip_norm']})"
            if dp_method in available:
                resolved.append(("FedAvg+DP(best)", dp_method))

    if "FedAvg+GRL(alpha=0.1)" in available:
        resolved.append(("FedAvg+GRL(alpha=0.1)", "FedAvg+GRL(alpha=0.1)"))
    elif "FedAvg+GRL(alpha=0.1,seed=42)" in available:
        resolved.append(("FedAvg+GRL(alpha=0.1)", "FedAvg+GRL(alpha=0.1,seed=42)"))
    else:
        grl_path = config.TABLE_DIR / "grl_alpha_sweep.csv"
        if grl_path.exists():
            grl_df = pd.read_csv(grl_path)
            if not grl_df.empty:
                best_alpha = grl_df.sort_values("macro_f1", ascending=False).iloc[0]["alpha"]
                method = f"GRL-AlphaSweep(alpha={best_alpha})"
                if method in available:
                    resolved.append(("FedAvg+GRL(best)", method))

    if "FedPriv-SER(alpha=0.5,sigma=0.05)" in available:
        resolved.append(("FedAvg+GRL+DP", "FedPriv-SER(alpha=0.5,sigma=0.05)"))
    elif "FedPriv-SER(best,seed=42)" in available:
        resolved.append(("FedAvg+GRL+DP(best)", "FedPriv-SER(best,seed=42)"))

    return resolved


def run(methods: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    methods = methods or _resolve_methods()
    rows = []
    for display_method, source_method in methods:
        for attack_model in ["logreg", "rf"]:
            result = run_update_attack_for_method(source_method, attack_model=attack_model)
            result["source_method"] = source_method
            result["method"] = display_method
            rows.append(result)
    frame = pd.DataFrame(rows)
    out_path = config.TABLE_DIR / "update_level_attack_results.csv"
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = frame[frame["attack_model"] == "LogisticRegression"].copy()
    x = range(len(plot_df))
    ax.bar([i - 0.15 for i in x], plot_df["speaker_attack_acc"], width=0.3, label="Speaker update attack")
    ax.bar([i + 0.15 for i in x], plot_df["gender_attack_acc"], width=0.3, label="Gender update attack")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["method"], rotation=30, ha="right")
    ax.set_title("Update-level Attack Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "update_level_attack_bar.png", dpi=200)
    plt.close(fig)
    return frame


if __name__ == "__main__":
    print(run())
