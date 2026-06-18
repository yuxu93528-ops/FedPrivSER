from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config
    from .dataset import build_loader, prepare_data
    from .train_common import append_result_row, evaluate_model, save_experiment_artifacts, train_epochs
    from .models import EmotionMLP
    from .utils import Timer, count_parameters, set_seed
except ImportError:
    import config
    from dataset import build_loader, prepare_data
    from train_common import append_result_row, evaluate_model, save_experiment_artifacts, train_epochs
    from models import EmotionMLP
    from utils import Timer, count_parameters, set_seed


def run() -> dict:
    set_seed()
    prepared = prepare_data()
    device = config.get_device()
    rows = []
    aggregate = []
    timer = Timer()
    for actor_id in sorted(prepared.data["actor_id"].unique()):
        train_df = prepared.train_df[prepared.train_df["actor_id"] == actor_id]
        val_df = prepared.val_df[prepared.val_df["actor_id"] == actor_id]
        test_df = prepared.test_df[prepared.test_df["actor_id"] == actor_id]
        model = EmotionMLP().to(device)
        model, _ = train_epochs(
            model,
            build_loader(train_df, prepared.scaler, prepared.emotion_encoder, shuffle=True),
            build_loader(val_df, prepared.scaler, prepared.emotion_encoder, shuffle=False),
            device,
            epochs=config.LOCAL_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        metrics = evaluate_model(model, build_loader(test_df, prepared.scaler, prepared.emotion_encoder, shuffle=False), device)
        rows.append({
            "method": "Local-only",
            "actor_id": actor_id,
            "accuracy": metrics["accuracy"],
            "uar": metrics["uar"],
            "macro_f1": metrics["macro_f1"],
        })
        aggregate.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(config.CLIENT_LEVEL_RESULTS_CSV, index=False, encoding="utf-8-sig")
    mean_metrics = {
        "accuracy": float(np.mean([item["accuracy"] for item in aggregate])),
        "uar": float(np.mean([item["uar"] for item in aggregate])),
        "macro_f1": float(np.mean([item["macro_f1"] for item in aggregate])),
    }
    append_result_row(
        config.MAIN_RESULTS_CSV,
        {
            "method": "Local-only",
            "accuracy": mean_metrics["accuracy"],
            "uar": mean_metrics["uar"],
            "macro_f1": mean_metrics["macro_f1"],
            "params": count_parameters(EmotionMLP()),
            "training_time": timer.elapsed(),
        },
    )
    save_experiment_artifacts(
        method="Local-only",
        experiment_cfg=config.ExperimentConfig(method="Local-only"),
        history=[],
        metrics=mean_metrics,
        model=EmotionMLP(),
        training_time=timer.elapsed(),
    )
    return {"method": "Local-only", "metrics": mean_metrics}


if __name__ == "__main__":
    print(run())
