from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

try:
    from . import config
    from .dataset import PreparedData, build_loader
    from .metrics import emotion_metrics
    from .models import EmotionMLP
    from .utils import Timer, count_parameters, describe_runtime, get_logger, save_json, set_seed
except ImportError:
    import config
    from dataset import PreparedData, build_loader
    from metrics import emotion_metrics
    from models import EmotionMLP
    from utils import Timer, count_parameters, describe_runtime, get_logger, save_json, set_seed


def evaluate_model(model: EmotionMLP, loader, device: torch.device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["emotion"].to(device)
            logits = model(x)["emotion_logits"]
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return emotion_metrics(np.array(y_true), np.array(y_pred), labels=list(range(8)))


def train_epochs(
    model: EmotionMLP,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    alpha_grl: float = 0.0,
    use_grl: bool = False,
    logger=None,
) -> tuple[EmotionMLP, list[dict]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_state = None
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            emo = batch["emotion"].to(device)
            spk = batch["speaker"].to(device)
            outputs = model(x, lambda_grl=1.0)
            loss = criterion(outputs["emotion_logits"], emo)
            if use_grl:
                loss = loss + alpha_grl * criterion(outputs["speaker_logits"], spk)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_metrics = evaluate_model(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        if logger:
            logger.info("epoch=%s loss=%.4f val_macro_f1=%.4f", epoch, row["loss"], row["macro_f1"])
        if row["macro_f1"] > best_val:
            best_val = row["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_experiment_artifacts(
    method: str,
    experiment_cfg,
    history: list[dict],
    metrics: dict,
    model: EmotionMLP,
    training_time: float,
) -> Path:
    config.ensure_directories()
    ckpt_path = config.CHECKPOINT_DIR / f"{method}.pt"
    torch.save(model.state_dict(), ckpt_path)
    payload = {
        "config": asdict(experiment_cfg),
        "history": history,
        "metrics": metrics,
        "training_time": training_time,
        "params": count_parameters(model),
        "runtime": describe_runtime(),
    }
    save_json(payload, config.LOG_DIR / f"{method}.json")
    return ckpt_path


def append_result_row(csv_path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["method"] != row["method"]] if "method" in existing.columns else existing
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")


def run_single_training(
    method: str,
    prepared: PreparedData,
    epochs: int,
    use_grl: bool = False,
    alpha_grl: float = 0.0,
    seed: int = config.SEED,
    model_kwargs: dict | None = None,
) -> dict:
    set_seed(seed)
    device = config.get_device()
    logger = get_logger(method, config.LOG_DIR / f"{method}.log")
    model_kwargs = model_kwargs or {}
    model = EmotionMLP(use_grl=use_grl, **model_kwargs).to(device)
    timer = Timer()
    train_loader = build_loader(prepared.train_df, prepared.scaler, prepared.emotion_encoder, shuffle=True)
    val_loader = build_loader(prepared.val_df, prepared.scaler, prepared.emotion_encoder, shuffle=False)
    test_loader = build_loader(prepared.test_df, prepared.scaler, prepared.emotion_encoder, shuffle=False)
    model, history = train_epochs(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        alpha_grl=alpha_grl,
        use_grl=use_grl,
        logger=logger,
    )
    metrics = evaluate_model(model, test_loader, device)
    training_time = timer.elapsed()
    save_experiment_artifacts(
        method=method,
        experiment_cfg=config.ExperimentConfig(method=method, alpha_grl=alpha_grl),
        history=history,
        metrics=metrics,
        model=model,
        training_time=training_time,
    )
    row = {
        "method": method,
        "accuracy": metrics["accuracy"],
        "uar": metrics["uar"],
        "macro_f1": metrics["macro_f1"],
        "params": count_parameters(model),
        "training_time": training_time,
    }
    append_result_row(config.MAIN_RESULTS_CSV, row)
    return {
        "method": method,
        "metrics": metrics,
        "history": history,
        "training_time": training_time,
        "params": count_parameters(model),
    }
