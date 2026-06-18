from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

try:
    from . import config
except ImportError:
    import config


FEATURE_COLUMNS = [f"feat_{idx}" for idx in range(config.FEATURE_DIM)]


class FeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, scaler: StandardScaler, emotion_encoder: LabelEncoder) -> None:
        self.frame = frame.reset_index(drop=True)
        self.x = scaler.transform(self.frame[FEATURE_COLUMNS].to_numpy(np.float32))
        self.y = emotion_encoder.transform(self.frame["emotion"].to_numpy())
        self.speaker = self.frame["actor_id"].to_numpy(np.int64) - 1
        self.gender = (self.frame["gender"] == "female").astype(np.int64).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.x[index], dtype=torch.float32),
            "emotion": torch.tensor(self.y[index], dtype=torch.long),
            "speaker": torch.tensor(self.speaker[index], dtype=torch.long),
            "gender": torch.tensor(self.gender[index], dtype=torch.long),
        }


@dataclass
class PreparedData:
    data: pd.DataFrame
    emotion_encoder: LabelEncoder
    scaler: StandardScaler
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def load_feature_frame() -> pd.DataFrame:
    if not config.FEATURE_CSV.exists():
        raise FileNotFoundError(f"Feature file not found: {config.FEATURE_CSV}")
    df = pd.read_csv(config.FEATURE_CSV)
    if config.FAST_DEBUG:
        df = df.groupby("actor_id", group_keys=False).head(max(4, config.DEBUG_MAX_SAMPLES // 24)).reset_index(drop=True)
    return df


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.default_rng(config.SEED)
    for actor_id in sorted(df["actor_id"].unique()):
        actor_df = df[df["actor_id"] == actor_id].copy()
        for emotion in sorted(actor_df["emotion"].unique()):
            emo_df = actor_df[actor_df["emotion"] == emotion].sample(
                frac=1.0,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            n = len(emo_df)
            n_train = max(1, int(round(n * config.TRAIN_RATIO)))
            n_val = max(1, int(round(n * config.VAL_RATIO)))
            n_test = n - n_train - n_val

            if n_test < 1:
                if n_train > n_val:
                    n_train -= 1
                else:
                    n_val -= 1
                n_test = 1
            if n_train + n_val + n_test != n:
                n_train = n - n_val - n_test

            train_actor = emo_df.iloc[:n_train].copy()
            val_actor = emo_df.iloc[n_train:n_train + n_val].copy()
            test_actor = emo_df.iloc[n_train + n_val:].copy()

            train_actor["split"] = "train"
            val_actor["split"] = "val"
            test_actor["split"] = "test"
            train_parts.append(train_actor)
            val_parts.append(val_actor)
            test_parts.append(test_actor)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, val_df, test_df


def prepare_data() -> PreparedData:
    df = load_feature_frame()
    train_df, val_df, test_df = split_dataframe(df)
    full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].to_numpy(np.float32))
    encoder = LabelEncoder()
    encoder.fit(full_df["emotion"].to_numpy())

    summary = (
        full_df.groupby(["split", "actor_id", "emotion"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "actor_id", "emotion"])
    )
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.SPLIT_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    return PreparedData(full_df, encoder, scaler, train_df, val_df, test_df)


def build_loader(frame: pd.DataFrame, scaler: StandardScaler, encoder: LabelEncoder, shuffle: bool) -> DataLoader:
    dataset = FeatureDataset(frame, scaler, encoder)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=shuffle)
