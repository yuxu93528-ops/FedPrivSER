from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.optim import Adam

try:
    from . import config
    from .dataset import build_loader, prepare_data
    from .metrics import classification_metrics, emotion_metrics, gender_metrics
    from .models import EmotionMLP, clone_model
    from .train_common import append_result_row, save_experiment_artifacts
    from .utils import Timer, count_parameters, get_logger, set_seed
except ImportError:
    import config
    from dataset import build_loader, prepare_data
    from metrics import classification_metrics, emotion_metrics, gender_metrics
    from models import EmotionMLP, clone_model
    from train_common import append_result_row, save_experiment_artifacts
    from utils import Timer, count_parameters, get_logger, set_seed


@dataclass
class FederatedRunConfig:
    method: str
    rounds: int = config.FEDERATED_ROUNDS
    local_epochs: int = config.LOCAL_EPOCHS
    learning_rate: float = config.LEARNING_RATE
    weight_decay: float = config.WEIGHT_DECAY
    sigma: float = -1.0
    clip_norm: float = config.CLIP_NORM
    alpha_s: float = 0.0
    alpha_g: float = 0.0
    alpha_mode: str = "fixed"
    seed: int = config.SEED
    save_main_result: bool = True
    save_updates: bool = True
    save_curve: bool = True
    fast_debug: bool = config.FAST_DEBUG

    def resolved_rounds(self) -> int:
        return min(self.rounds, config.DEBUG_ROUNDS) if self.fast_debug else self.rounds


def method_to_filename(method: str, suffix: str) -> Path:
    return config.LOG_DIR / f"{method}{suffix}"


def build_model(alpha_s: float = 0.0, alpha_g: float = 0.0) -> EmotionMLP:
    return EmotionMLP(use_grl=alpha_s > 0, use_gender_grl=alpha_g > 0)


def grl_lambda(alpha_max: float, mode: str, current_step: int, total_steps: int) -> float:
    if alpha_max <= 0:
        return 0.0
    if mode == "scheduled":
        p = current_step / max(1, total_steps)
        return alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
    return alpha_max


def state_dict_difference(local_state, global_state):
    return OrderedDict((key, local_state[key] - global_state[key]) for key in global_state.keys())


def summarize_update(update: OrderedDict[str, torch.Tensor]) -> dict[str, float]:
    flat = torch.cat([tensor.flatten() for tensor in update.values()]).float()
    stats = {
        "global_update_norm": float(torch.norm(flat, p=2).item()),
        "global_update_mean": float(flat.mean().item()),
        "global_update_std": float(flat.std(unbiased=False).item()),
        "global_update_min": float(flat.min().item()),
        "global_update_max": float(flat.max().item()),
        "global_update_median": float(flat.median().item()),
        "global_update_p25": float(torch.quantile(flat, 0.25).item()),
        "global_update_p75": float(torch.quantile(flat, 0.75).item()),
    }
    for idx, (_, tensor) in enumerate(update.items()):
        layer_flat = tensor.flatten().float()
        stats[f"layer_{idx}_norm"] = float(torch.norm(layer_flat, p=2).item())
        stats[f"layer_{idx}_mean"] = float(layer_flat.mean().item())
        stats[f"layer_{idx}_std"] = float(layer_flat.std(unbiased=False).item())
    return stats


def apply_client_dp(update, clip_norm: float, sigma: float):
    flat = torch.cat([value.flatten() for value in update.values()])
    norm = torch.norm(flat, p=2)
    scale = max(1.0, norm.item() / clip_norm)
    clipped = OrderedDict((key, value / scale) for key, value in update.items())
    clipped_ratio = 1.0 if norm.item() > clip_norm else 0.0
    if sigma > 0:
        perturbed = OrderedDict(
            (key, value + torch.randn_like(value) * (sigma * clip_norm))
            for key, value in clipped.items()
        )
    else:
        perturbed = clipped
    return perturbed, {
        "raw_update_norm": float(norm.item()),
        "clip_scale": float(scale),
        "clipped_ratio": float(clipped_ratio),
    }


def aggregate_updates(global_state, client_updates, weights):
    total = float(sum(weights))
    new_state = OrderedDict((key, value.clone()) for key, value in global_state.items())
    for key in new_state.keys():
        delta = sum((weight / total) * update[key] for update, weight in zip(client_updates, weights))
        new_state[key] = new_state[key] + delta
    return new_state


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))["emotion_logits"]
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(batch["emotion"].numpy())
    return emotion_metrics(np.array(y_true), np.array(y_pred), list(range(8)))


def train_client(
    model,
    loader,
    device,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    alpha_s: float,
    alpha_g: float,
    alpha_mode: str,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, local_epochs * len(loader))
    current_step = 0
    model.train()
    for _ in range(local_epochs):
        for batch in loader:
            current_step += 1
            x = batch["x"].to(device)
            emo = batch["emotion"].to(device)
            spk = batch["speaker"].to(device)
            gender = batch["gender"].to(device)
            lambda_s = grl_lambda(alpha_s, alpha_mode, current_step, total_steps)
            lambda_g = grl_lambda(alpha_g, alpha_mode, current_step, total_steps)
            outputs = model(x, lambda_speaker=lambda_s, lambda_gender=lambda_g)
            loss = criterion(outputs["emotion_logits"], emo)
            if alpha_s > 0:
                loss = loss + criterion(outputs["speaker_logits"], spk)
            if alpha_g > 0:
                loss = loss + criterion(outputs["gender_logits"], gender)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def run_federated_experiment(run_cfg: FederatedRunConfig) -> dict:
    set_seed(run_cfg.seed)
    prepared = prepare_data()
    device = config.get_device()
    logger = get_logger(run_cfg.method, config.LOG_DIR / f"{run_cfg.method}.log")
    model = build_model(run_cfg.alpha_s, run_cfg.alpha_g).to(device)
    history = []
    update_records = []
    timer = Timer()
    val_loader = build_loader(prepared.val_df, prepared.scaler, prepared.emotion_encoder, shuffle=False)
    test_loader = build_loader(prepared.test_df, prepared.scaler, prepared.emotion_encoder, shuffle=False)

    for round_idx in range(1, run_cfg.resolved_rounds() + 1):
        global_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())
        client_updates = []
        weights = []
        round_raw_norms = []
        round_clipped = []
        for actor_id in sorted(prepared.data["actor_id"].unique()):
            local_model = clone_model(model).to(device)
            client_train = prepared.train_df[prepared.train_df["actor_id"] == actor_id]
            loader = build_loader(client_train, prepared.scaler, prepared.emotion_encoder, shuffle=True)
            local_model = train_client(
                local_model,
                loader,
                device,
                local_epochs=run_cfg.local_epochs,
                learning_rate=run_cfg.learning_rate,
                weight_decay=run_cfg.weight_decay,
                alpha_s=run_cfg.alpha_s,
                alpha_g=run_cfg.alpha_g,
                alpha_mode=run_cfg.alpha_mode,
            )
            local_state = OrderedDict((k, v.detach().cpu().clone()) for k, v in local_model.state_dict().items())
            raw_update = state_dict_difference(local_state, global_state)
            noisy_update = raw_update
            dp_meta = {"raw_update_norm": float(torch.norm(torch.cat([v.flatten() for v in raw_update.values()]), p=2).item()), "clipped_ratio": 0.0}
            if run_cfg.sigma >= 0:
                noisy_update, dp_meta = apply_client_dp(raw_update, run_cfg.clip_norm, run_cfg.sigma)
            client_updates.append(noisy_update)
            weights.append(len(client_train))
            round_raw_norms.append(dp_meta["raw_update_norm"])
            round_clipped.append(dp_meta["clipped_ratio"])
            if run_cfg.save_updates:
                update_row = {
                    "method": run_cfg.method,
                    "round": round_idx,
                    "actor_id": actor_id,
                    "gender": "male" if actor_id % 2 == 1 else "female",
                    "sigma": run_cfg.sigma,
                    "clip_norm": run_cfg.clip_norm,
                    "alpha_s": run_cfg.alpha_s,
                    "alpha_g": run_cfg.alpha_g,
                    "alpha_mode": run_cfg.alpha_mode,
                    **dp_meta,
                    **summarize_update(noisy_update),
                }
                update_records.append(update_row)
        aggregated = aggregate_updates(global_state, client_updates, weights)
        model.load_state_dict(aggregated)
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics["raw_update_norm_mean"] = float(np.mean(round_raw_norms))
        val_metrics["raw_update_norm_median"] = float(np.median(round_raw_norms))
        val_metrics["clipped_ratio"] = float(np.mean(round_clipped))
        history.append({"round": round_idx, **val_metrics})
        logger.info("round=%s val_macro_f1=%.4f", round_idx, val_metrics["macro_f1"])

    test_metrics = evaluate_model(model, test_loader, device)
    training_time = timer.elapsed()
    save_experiment_artifacts(
        method=run_cfg.method,
        experiment_cfg=config.ExperimentConfig(
            method=run_cfg.method,
            federated_rounds=run_cfg.rounds,
            local_epochs=run_cfg.local_epochs,
            learning_rate=run_cfg.learning_rate,
            clip_norm=run_cfg.clip_norm,
            sigma=max(0.0, run_cfg.sigma),
            alpha_grl=run_cfg.alpha_s,
            seed=run_cfg.seed,
            fast_debug=run_cfg.fast_debug,
        ),
        history=history,
        metrics=test_metrics,
        model=model,
        training_time=training_time,
    )
    meta = {
        "method": run_cfg.method,
        "model_type": "dual_grl" if run_cfg.alpha_g > 0 else ("grl" if run_cfg.alpha_s > 0 else "plain"),
        "alpha_s": run_cfg.alpha_s,
        "alpha_g": run_cfg.alpha_g,
        "alpha_mode": run_cfg.alpha_mode,
        "sigma": run_cfg.sigma,
        "clip_norm": run_cfg.clip_norm,
        "rounds": run_cfg.rounds,
        "local_epochs": run_cfg.local_epochs,
        "learning_rate": run_cfg.learning_rate,
        "seed": run_cfg.seed,
        "params": count_parameters(model),
        "training_time": training_time,
    }
    with (config.LOG_DIR / f"{run_cfg.method}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if run_cfg.save_curve:
        pd.DataFrame(history).to_csv(config.LOG_DIR / f"{run_cfg.method}_curve.csv", index=False, encoding="utf-8-sig")
    if run_cfg.save_updates:
        pd.DataFrame(update_records).to_csv(config.LOG_DIR / f"{run_cfg.method}_updates.csv", index=False, encoding="utf-8-sig")
    if run_cfg.save_main_result:
        append_result_row(
            config.MAIN_RESULTS_CSV,
            {
                "method": run_cfg.method,
                "accuracy": test_metrics["accuracy"],
                "uar": test_metrics["uar"],
                "macro_f1": test_metrics["macro_f1"],
                "params": count_parameters(model),
                "training_time": training_time,
            },
        )
    return {
        "method": run_cfg.method,
        "metrics": test_metrics,
        "history": history,
        "training_time": training_time,
        "params": count_parameters(model),
        "meta": meta,
    }


def infer_model_from_method(method: str) -> EmotionMLP:
    meta_path = config.LOG_DIR / f"{method}_meta.json"
    alpha_s = 0.0
    alpha_g = 0.0
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        alpha_s = float(meta.get("alpha_s", 0.0))
        alpha_g = float(meta.get("alpha_g", 0.0))
    else:
        if "Dual-GRL" in method:
            alpha_s, alpha_g = 0.1, 0.1
        elif "GRL" in method:
            alpha_s = 0.1
    return build_model(alpha_s, alpha_g)


def extract_representations(method: str):
    prepared = prepare_data()
    device = config.get_device()
    model = infer_model_from_method(method).to(device)
    ckpt_path = config.CHECKPOINT_DIR / f"{method}.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    loader = build_loader(prepared.test_df, prepared.scaler, prepared.emotion_encoder, shuffle=False)
    zs, speakers, genders = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            zs.append(model.extract_representation(batch["x"].to(device)).cpu().numpy())
            speakers.append(batch["speaker"].numpy())
            genders.append(batch["gender"].numpy())
    return np.concatenate(zs), np.concatenate(speakers), np.concatenate(genders)


def run_representation_attack_for_method(method: str) -> dict:
    z, speakers, genders = extract_representations(method)
    speaker_clf = LogisticRegression(max_iter=2000)
    speaker_clf.fit(z, speakers)
    speaker_pred = speaker_clf.predict(z)
    speaker_metrics = classification_metrics(speakers, speaker_pred)
    gender_clf = LogisticRegression(max_iter=1000)
    gender_clf.fit(z, genders)
    gender_pred = gender_clf.predict(z)
    gender_prob = gender_clf.predict_proba(z)[:, 1]
    g_metrics = gender_metrics(genders, gender_pred, gender_prob)
    return {
        "method": method,
        "speaker_attack_acc": speaker_metrics["accuracy"],
        "speaker_attack_macro_f1": speaker_metrics["macro_f1"],
        "gender_attack_acc": g_metrics["accuracy"],
        "gender_attack_macro_f1": g_metrics["macro_f1"],
        "gender_auc": g_metrics["auc"],
    }


def run_update_attack_for_method(method: str, attack_model: str = "logreg") -> dict:
    update_path = config.LOG_DIR / f"{method}_updates.csv"
    if not update_path.exists():
        raise FileNotFoundError(f"Update records not found: {update_path}")
    frame = pd.read_csv(update_path)
    feature_cols = [
        col for col in frame.columns
        if col.startswith("global_update_") or col.startswith("layer_")
    ]
    x = frame[feature_cols].to_numpy(np.float32)
    speaker_y = frame["actor_id"].to_numpy()
    gender_y = (frame["gender"] == "female").astype(np.int64).to_numpy()
    if attack_model == "rf":
        speaker_model = RandomForestClassifier(n_estimators=200, random_state=config.SEED)
        gender_model = RandomForestClassifier(n_estimators=200, random_state=config.SEED)
    else:
        speaker_model = LogisticRegression(max_iter=2000)
        gender_model = LogisticRegression(max_iter=1000)
    speaker_model.fit(x, speaker_y)
    speaker_pred = speaker_model.predict(x)
    speaker_metrics = classification_metrics(speaker_y, speaker_pred)
    gender_model.fit(x, gender_y)
    gender_pred = gender_model.predict(x)
    if hasattr(gender_model, "predict_proba"):
        gender_prob = gender_model.predict_proba(x)[:, 1]
    else:
        gender_prob = None
    g_metrics = gender_metrics(gender_y, gender_pred, gender_prob)
    return {
        "method": method,
        "attack_model": "RandomForest" if attack_model == "rf" else "LogisticRegression",
        "speaker_attack_acc": speaker_metrics["accuracy"],
        "speaker_attack_macro_f1": speaker_metrics["macro_f1"],
        "gender_attack_acc": g_metrics["accuracy"],
        "gender_attack_macro_f1": g_metrics["macro_f1"],
        "gender_auc": g_metrics["auc"],
    }
