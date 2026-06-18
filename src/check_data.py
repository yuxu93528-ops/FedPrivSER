from __future__ import annotations

from pathlib import Path

try:
    from . import config
    from .parse_ravdess import build_metadata
except ImportError:
    import config
    from parse_ravdess import build_metadata


def check_dataset(data_root: Path = config.DATA_ROOT) -> dict:
    actor_dirs = [data_root / f"Actor_{idx:02d}" for idx in range(1, 25)]
    missing = [str(path) for path in actor_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing actor directories: {missing}")
    wav_files = list(data_root.glob("Actor_*/*.wav"))
    if len(wav_files) != 1440:
        raise RuntimeError(f"Expected 1440 wav files, found {len(wav_files)}")
    df = build_metadata(data_root)
    return {
        "data_root": str(data_root),
        "actors_found": len(actor_dirs),
        "wav_files": len(wav_files),
        "emotions": sorted(df["emotion"].unique().tolist()),
        "actor_id_range": [int(df["actor_id"].min()), int(df["actor_id"].max())],
        "genders": sorted(df["gender"].unique().tolist()),
    }


def main() -> None:
    summary = check_dataset()
    print("Dataset check passed.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
