from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

try:
    from . import config
except ImportError:
    import config


def plot_main_results(main_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(main_df))
    width = 0.25
    ax.bar(x - width, main_df["accuracy"], width, label="Accuracy")
    ax.bar(x, main_df["uar"], width, label="UAR")
    ax.bar(x + width, main_df["macro_f1"], width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(main_df["method"], rotation=30, ha="right")
    ax.legend()
    ax.set_title("Emotion Results")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "emotion_results_bar.png", dpi=200)
    plt.close(fig)


def plot_privacy_results(privacy_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(privacy_df))
    width = 0.35
    ax.bar(x - width / 2, privacy_df["speaker_attack_acc"], width, label="Speaker Attack Acc")
    ax.bar(x + width / 2, privacy_df["gender_attack_acc"], width, label="Gender Attack Acc")
    ax.set_xticks(x)
    ax.set_xticklabels(privacy_df["method"], rotation=30, ha="right")
    ax.legend()
    ax.set_title("Privacy Attack Results")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "privacy_attack_bar.png", dpi=200)
    plt.close(fig)


def plot_tradeoff(main_df: pd.DataFrame, privacy_df: pd.DataFrame) -> None:
    merged = main_df.merge(privacy_df, on="method", how="inner")
    merged["sigma"] = merged["method"].str.extract(r"sigma=([0-9.]+)").fillna("0").astype(float)
    merged["alpha"] = merged["method"].str.extract(r"alpha=([0-9.]+)").fillna("0").astype(float)
    merged[["method", "sigma", "alpha", "macro_f1", "speaker_attack_acc", "gender_attack_acc"]].rename(
        columns={"macro_f1": "emotion_macro_f1"}
    ).to_csv(config.TRADEOFF_CSV, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["macro_f1"], merged["speaker_attack_acc"], label="Speaker attack")
    ax.scatter(merged["macro_f1"], merged["gender_attack_acc"], label="Gender attack")
    for _, row in merged.iterrows():
        ax.annotate(row["method"], (row["macro_f1"], row["speaker_attack_acc"]), fontsize=8)
    ax.set_xlabel("Emotion Macro-F1")
    ax.set_ylabel("Attack Accuracy")
    ax.legend()
    ax.set_title("Privacy-Utility Tradeoff")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "privacy_utility_curve.png", dpi=200)
    plt.close(fig)


def plot_fedavg_curve() -> None:
    curve_path = config.LOG_DIR / "FedAvg_curve.csv"
    if not curve_path.exists():
        return
    frame = pd.read_csv(curve_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame["round"], frame["macro_f1"], marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation Macro-F1")
    ax.set_title("FedAvg Training Curve")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "fedavg_training_curve.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix() -> None:
    fedpriv_logs = sorted(config.LOG_DIR.glob("FedPriv-SER*.json"))
    if not fedpriv_logs:
        return
    latest = fedpriv_logs[-1]
    import json
    with latest.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    matrix = np.array(payload["metrics"]["confusion_matrix"])
    labels = list(config.EMOTION_MAP.values())
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("FedPriv-SER Confusion Matrix")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "confusion_matrix_fedpriv_ser.png", dpi=200)
    plt.close(fig)


def main() -> None:
    main_df = pd.read_csv(config.MAIN_RESULTS_CSV)
    privacy_df = pd.read_csv(config.PRIVACY_RESULTS_CSV)
    plot_main_results(main_df)
    plot_privacy_results(privacy_df)
    plot_tradeoff(main_df, privacy_df)
    plot_fedavg_curve()
    plot_confusion_matrix()


if __name__ == "__main__":
    main()
