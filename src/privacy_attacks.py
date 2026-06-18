from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from . import config
    from .federated_core import run_representation_attack_for_method
except ImportError:
    import config
    from federated_core import run_representation_attack_for_method


def run_attack(methods: list[str] | None = None) -> pd.DataFrame:
    if methods is None:
        methods = [row["method"] for _, row in pd.read_csv(config.MAIN_RESULTS_CSV).iterrows()]
    frame = pd.DataFrame([run_representation_attack_for_method(method) for method in methods])
    frame.to_csv(config.PRIVACY_RESULTS_CSV, index=False, encoding="utf-8-sig")
    return frame


if __name__ == "__main__":
    print(run_attack())
