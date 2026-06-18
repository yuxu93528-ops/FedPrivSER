from __future__ import annotations

import pandas as pd

try:
    from . import config
    from .federated_core import run_representation_attack_for_method
    from .train_fedavg import run_federated
except ImportError:
    import config
    from federated_core import run_representation_attack_for_method
    from train_fedavg import run_federated


def run() -> pd.DataFrame:
    settings = [
        ("fixed", 0.1, "GRL-Fixed(alpha=0.1)"),
        ("scheduled", 0.1, "GRL-Scheduled(alpha_max=0.1)"),
        ("scheduled", 0.3, "GRL-Scheduled(alpha_max=0.3)"),
        ("scheduled", 0.5, "GRL-Scheduled(alpha_max=0.5)"),
    ]
    rows = []
    for mode, alpha_max, method in settings:
        result = run_federated(
            method=method,
            sigma=-1.0,
            use_grl=True,
            alpha_grl=alpha_max,
            alpha_mode=mode,
            save_main_result=False,
        )
        attack = run_representation_attack_for_method(method)
        rows.append({
            "method": method,
            "alpha_mode": mode,
            "alpha_max": alpha_max,
            "accuracy": result["metrics"]["accuracy"],
            "uar": result["metrics"]["uar"],
            "macro_f1": result["metrics"]["macro_f1"],
            "speaker_attack_acc": attack["speaker_attack_acc"],
            "gender_attack_acc": attack["gender_attack_acc"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "grl_schedule_results.csv", index=False, encoding="utf-8-sig")
    return frame


if __name__ == "__main__":
    print(run())
