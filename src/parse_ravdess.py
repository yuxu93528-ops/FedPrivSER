from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from . import config
except ImportError:
    import config


def parse_filename(path: Path) -> dict:
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(f"Unexpected filename format: {path.name}")
    emotion_id = int(parts[2])
    actor_id = int(parts[6])
    if emotion_id not in config.EMOTION_MAP:
        raise ValueError(f"Unexpected emotion id {emotion_id} in {path.name}")
    return {
        "path": str(path),
        "filename": path.name,
        "modality": int(parts[0]),
        "vocal_channel": int(parts[1]),
        "emotion_id": emotion_id,
        "emotion": config.EMOTION_MAP[emotion_id],
        "emotion_intensity": int(parts[3]),
        "statement": int(parts[4]),
        "repetition": int(parts[5]),
        "actor_id": actor_id,
        "gender": "male" if actor_id % 2 == 1 else "female",
        "client_id": actor_id,
    }


def build_metadata(data_root: Path = config.DATA_ROOT) -> pd.DataFrame:
    rows = []
    for wav_path in sorted(data_root.glob("Actor_*/*.wav")):
        rows.append(parse_filename(wav_path))
    df = pd.DataFrame(rows)
    if len(df) != 1440:
        raise RuntimeError(f"Expected 1440 wav files, found {len(df)}")
    return df
