from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    import config


PAPER_FIG_DIR = config.PROJECT_ROOT.parent / "论文" / "figures"


def short_method_name(method: str) -> str:
    mapping = {
        "Centralized": "Centralized",
        "FedAvg": "FedAvg",
        "Best FedAvg+GRL": "GRL",
        "Best FedAvg+Dual-GRL": "Dual-GRL",
        "Best FedAvg+DP": "DP",
        "Best FedAvg+GRL+DP": "FedPriv-SER",
    }
    return mapping.get(method, method)


def load_selected_results() -> pd.DataFrame:
    frame = pd.read_csv(config.TABLE_DIR / "final_selected_results.csv")
    ordered = [
        "Centralized",
        "FedAvg",
        "Best FedAvg+GRL",
        "Best FedAvg+Dual-GRL",
        "Best FedAvg+DP",
        "Best FedAvg+GRL+DP",
    ]
    frame = frame.set_index("method").reindex(ordered).reset_index()
    frame["plot_name"] = frame["method"].map(short_method_name)
    return frame


def load_update_attack_results() -> pd.DataFrame:
    update_df = pd.read_csv(config.TABLE_DIR / "update_level_attack_results.csv")
    update_df = update_df[update_df["attack_model"] == "LogisticRegression"].copy()
    name_map = {
        "FedAvg": "FedAvg",
        "FedAvg+GRL(alpha=0.1)": "Best FedAvg+GRL",
        "FedAvg+DP(best)": "Best FedAvg+DP",
        "FedAvg+GRL+DP(best)": "Best FedAvg+GRL+DP",
    }
    update_df["method"] = update_df["method"].map(name_map)
    update_df = update_df.dropna(subset=["method"])
    update_df["plot_name"] = update_df["method"].map(short_method_name)
    return update_df


def plot_fig1a(selected_df: pd.DataFrame) -> Path:
    data = selected_df[selected_df["method"] != "Best FedAvg+DP"].copy()
    x = np.arange(len(data))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(x - width / 2, data["accuracy"], width, label="Accuracy", color="#4C78A8")
    ax.bar(x + width / 2, data["macro_f1"], width, label="Macro-F1", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(data["plot_name"], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    fig.tight_layout()
    output_path = PAPER_FIG_DIR / "paper_fig1_a_effectiveness.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fig1b(selected_df: pd.DataFrame) -> Path:
    data = selected_df.dropna(subset=["representation_speaker_attack_acc"]).copy()
    offsets = {
        "Centralized": (8, 8),
        "FedAvg": (8, 10),
        "GRL": (8, -2),
        "Dual-GRL": (8, -14),
    }
    colors = {
        "Centralized": "#4C78A8",
        "FedAvg": "#F58518",
        "GRL": "#54A24B",
        "Dual-GRL": "#B279A2",
    }

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for _, row in data.iterrows():
        label = row["plot_name"]
        ax.scatter(
            row["macro_f1"],
            row["representation_speaker_attack_acc"],
            s=95,
            color=colors.get(label, "#4C78A8"),
        )
        ax.annotate(
            label,
            (row["macro_f1"], row["representation_speaker_attack_acc"]),
            textcoords="offset points",
            xytext=offsets.get(label, (6, 6)),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "#666666"},
        )

    random_guess = 1 / 24
    ax.axhline(random_guess, color="#777777", linestyle="--", linewidth=1)
    ax.text(
        0.565,
        random_guess + 0.01,
        "Random guess (1/24)",
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    ax.set_xlabel("Emotion Macro-F1")
    ax.set_ylabel("Speaker attack accuracy")
    ax.set_xlim(0.31, 0.73)
    ax.set_ylim(0.0, 1.02)
    ax.grid(linestyle="--", linewidth=0.7, alpha=0.35)

    fig.tight_layout()
    output_path = PAPER_FIG_DIR / "paper_fig1_b_tradeoff.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fig2a(selected_df: pd.DataFrame) -> Path:
    data = selected_df.dropna(subset=["representation_speaker_attack_acc"]).copy()
    x = np.arange(len(data))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(
        x - width / 2,
        data["representation_speaker_attack_acc"],
        width,
        label="Speaker",
        color="#54A24B",
    )
    ax.bar(
        x + width / 2,
        data["representation_gender_attack_acc"],
        width,
        label="Gender",
        color="#4C78A8",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["plot_name"], rotation=15, ha="right")
    ax.set_ylabel("Attack accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    output_path = PAPER_FIG_DIR / "paper_fig2_a_rep_attack.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_fig2b(update_df: pd.DataFrame) -> Path:
    x = np.arange(len(update_df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(
        x - width / 2,
        update_df["speaker_attack_acc"],
        width,
        label="Speaker",
        color="#E45756",
    )
    ax.bar(
        x + width / 2,
        update_df["gender_attack_acc"],
        width,
        label="Gender",
        color="#72B7B2",
    )

    random_guess = 1 / 24
    ax.axhline(random_guess, color="#777777", linestyle="--", linewidth=1)
    ax.text(
        -0.35,
        random_guess + 0.008,
        "Random guess (1/24)",
        fontsize=8,
        color="#555555",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(update_df["plot_name"], rotation=15, ha="right")
    ax.set_ylabel("Attack accuracy")
    ax.set_ylim(0, 0.68)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    fig.tight_layout()
    output_path = PAPER_FIG_DIR / "paper_fig2_b_update_attack.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    config.ensure_directories()
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    selected_df = load_selected_results()
    update_df = load_update_attack_results()

    outputs = [
        plot_fig1a(selected_df),
        plot_fig1b(selected_df),
        plot_fig2a(selected_df),
        plot_fig2b(update_df),
    ]

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
