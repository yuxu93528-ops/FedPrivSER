from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
FEATURE_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

DATA_ROOT = Path(r"E:\2\Audio_Speech_Actors_01-24")
METADATA_CSV = FEATURE_DIR / "ravdess_metadata.csv"
FEATURE_CSV = FEATURE_DIR / "ravdess_mfcc_features.csv"
SPLIT_SUMMARY_CSV = TABLE_DIR / "data_split_summary.csv"
MAIN_RESULTS_CSV = TABLE_DIR / "main_emotion_results.csv"
PRIVACY_RESULTS_CSV = TABLE_DIR / "privacy_attack_results.csv"
TRADEOFF_CSV = TABLE_DIR / "privacy_utility_tradeoff.csv"
CLIENT_LEVEL_RESULTS_CSV = TABLE_DIR / "client_level_results.csv"

SEED = 42
SAMPLE_RATE = 16_000
N_MFCC = 40
FEATURE_DIM = 240
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 32
CENTRALIZED_EPOCHS = 50
LOCAL_EPOCHS = 1
FEDERATED_ROUNDS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 64
CLIP_NORM = 1.0
SIGMA = 0.05
ALPHA_GRL = 0.5
FAST_DEBUG = False
DEBUG_MAX_SAMPLES = 240
DEBUG_ROUNDS = 10
DEBUG_EPOCHS = 8

EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_directories() -> None:
    for path in [FEATURE_DIR, RESULTS_DIR, LOG_DIR, TABLE_DIR, FIGURE_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    method: str
    batch_size: int = BATCH_SIZE
    centralized_epochs: int = CENTRALIZED_EPOCHS
    local_epochs: int = LOCAL_EPOCHS
    federated_rounds: int = FEDERATED_ROUNDS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    dropout: float = DROPOUT
    hidden_dim_1: int = HIDDEN_DIM_1
    hidden_dim_2: int = HIDDEN_DIM_2
    feature_dim: int = FEATURE_DIM
    clip_norm: float = CLIP_NORM
    sigma: float = SIGMA
    alpha_grl: float = ALPHA_GRL
    seed: int = SEED
    fast_debug: bool = FAST_DEBUG

    def adjusted(self) -> "ExperimentConfig":
        cfg = ExperimentConfig(**asdict(self))
        if cfg.fast_debug:
            cfg.centralized_epochs = min(cfg.centralized_epochs, DEBUG_EPOCHS)
            cfg.federated_rounds = min(cfg.federated_rounds, DEBUG_ROUNDS)
        return cfg
