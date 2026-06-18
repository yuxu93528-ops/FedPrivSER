from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from . import config
    from .parse_ravdess import build_metadata
except ImportError:
    import config
    from parse_ravdess import build_metadata


def extract_feature_vector(audio_path: str) -> np.ndarray:
    signal, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    signal, _ = librosa.effects.trim(signal, top_db=25)
    if signal.size == 0:
        signal = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=signal, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stats = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1),
        delta2.mean(axis=1), delta2.std(axis=1),
    ]
    feature = np.concatenate(stats, axis=0).astype(np.float32)
    if feature.shape[0] != config.FEATURE_DIM:
        raise RuntimeError(f"Expected feature dim {config.FEATURE_DIM}, got {feature.shape[0]}")
    return feature


def run_feature_extraction(force: bool = False) -> tuple[Path, Path]:
    config.ensure_directories()
    if config.METADATA_CSV.exists() and config.FEATURE_CSV.exists() and not force:
        return config.METADATA_CSV, config.FEATURE_CSV

    metadata = build_metadata()
    feature_rows = []
    iterator = metadata.itertuples(index=False)
    for row in tqdm(iterator, total=len(metadata), desc="Extracting MFCC features"):
        feature = extract_feature_vector(row.path)
        record = {
            "path": row.path,
            "filename": row.filename,
            "emotion_id": row.emotion_id,
            "emotion": row.emotion,
            "actor_id": row.actor_id,
            "gender": row.gender,
        }
        for idx, value in enumerate(feature):
            record[f"feat_{idx}"] = float(value)
        feature_rows.append(record)

    metadata.to_csv(config.METADATA_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_rows).to_csv(config.FEATURE_CSV, index=False, encoding="utf-8-sig")
    return config.METADATA_CSV, config.FEATURE_CSV


def main() -> None:
    meta_path, feat_path = run_feature_extraction(force=False)
    print(f"Saved metadata to {meta_path}")
    print(f"Saved features to {feat_path}")


if __name__ == "__main__":
    main()
