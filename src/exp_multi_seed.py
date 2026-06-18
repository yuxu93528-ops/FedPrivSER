from __future__ import annotations

import pandas as pd

try:
    from . import config
    from .dataset import prepare_data
    from .federated_core import run_representation_attack_for_method
    from .train_common import run_single_training
    from .train_fedavg import run_federated
except ImportError:
    import config
    from dataset import prepare_data
    from federated_core import run_representation_attack_for_method
    from train_common import run_single_training
    from train_fedavg import run_federated


def _pick_best_dual() -> tuple[float, float]:
    path = config.TABLE_DIR / "dual_grl_results.csv"
    if not path.exists():
        return 0.1, 0.05
    df = pd.read_csv(path).sort_values("macro_f1", ascending=False)
    row = df.iloc[0]
    return float(row["alpha_s"]), float(row["alpha_g"])


def _pick_best_grl_dp() -> tuple[float, float]:
    return 0.5, 0.05


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [42, 123, 2024]
    dual_alpha_s, dual_alpha_g = _pick_best_dual()
    best_alpha, best_sigma = _pick_best_grl_dp()
    prepared = prepare_data()
    rows = []
    for seed in seeds:
        central_method = f"Centralized(seed={seed})"
        central = run_single_training(central_method, prepared, epochs=config.CENTRALIZED_EPOCHS, seed=seed)
        central_attack = run_representation_attack_for_method(central_method)
        rows.append({
            "method": "Centralized",
            "seed": seed,
            "accuracy": central["metrics"]["accuracy"],
            "uar": central["metrics"]["uar"],
            "macro_f1": central["metrics"]["macro_f1"],
            "speaker_attack_acc": central_attack["speaker_attack_acc"],
            "gender_attack_acc": central_attack["gender_attack_acc"],
        })
        fed = run_federated(f"FedAvg(seed={seed})", sigma=-1.0, use_grl=False, alpha_grl=0.0, seed=seed, save_main_result=False)
        fed_attack = run_representation_attack_for_method(f"FedAvg(seed={seed})")
        rows.append({
            "method": "FedAvg",
            "seed": seed,
            "accuracy": fed["metrics"]["accuracy"],
            "uar": fed["metrics"]["uar"],
            "macro_f1": fed["metrics"]["macro_f1"],
            "speaker_attack_acc": fed_attack["speaker_attack_acc"],
            "gender_attack_acc": fed_attack["gender_attack_acc"],
        })
        grl = run_federated(f"FedAvg+GRL(alpha=0.1,seed={seed})", sigma=-1.0, use_grl=True, alpha_grl=0.1, seed=seed, save_main_result=False)
        grl_attack = run_representation_attack_for_method(f"FedAvg+GRL(alpha=0.1,seed={seed})")
        rows.append({
            "method": "FedAvg+GRL(alpha=0.1)",
            "seed": seed,
            "accuracy": grl["metrics"]["accuracy"],
            "uar": grl["metrics"]["uar"],
            "macro_f1": grl["metrics"]["macro_f1"],
            "speaker_attack_acc": grl_attack["speaker_attack_acc"],
            "gender_attack_acc": grl_attack["gender_attack_acc"],
        })
        dual_method = f"FedAvg+Dual-GRL(best,seed={seed})"
        dual = run_federated(dual_method, sigma=-1.0, use_grl=True, alpha_grl=dual_alpha_s, alpha_g=dual_alpha_g, seed=seed, save_main_result=False)
        dual_attack = run_representation_attack_for_method(dual_method)
        rows.append({
            "method": "FedAvg+Dual-GRL(best)",
            "seed": seed,
            "accuracy": dual["metrics"]["accuracy"],
            "uar": dual["metrics"]["uar"],
            "macro_f1": dual["metrics"]["macro_f1"],
            "speaker_attack_acc": dual_attack["speaker_attack_acc"],
            "gender_attack_acc": dual_attack["gender_attack_acc"],
        })
        fedpriv_method = f"FedPriv-SER(best,seed={seed})"
        fedpriv = run_federated(fedpriv_method, sigma=best_sigma, use_grl=True, alpha_grl=best_alpha, seed=seed, save_main_result=False)
        fedpriv_attack = run_representation_attack_for_method(fedpriv_method)
        rows.append({
            "method": "FedAvg+GRL+DP(best)",
            "seed": seed,
            "accuracy": fedpriv["metrics"]["accuracy"],
            "uar": fedpriv["metrics"]["uar"],
            "macro_f1": fedpriv["metrics"]["macro_f1"],
            "speaker_attack_acc": fedpriv_attack["speaker_attack_acc"],
            "gender_attack_acc": fedpriv_attack["gender_attack_acc"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "multi_seed_results.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby("method").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        uar_mean=("uar", "mean"),
        uar_std=("uar", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        speaker_attack_acc_mean=("speaker_attack_acc", "mean"),
        speaker_attack_acc_std=("speaker_attack_acc", "std"),
        gender_attack_acc_mean=("gender_attack_acc", "mean"),
        gender_attack_acc_std=("gender_attack_acc", "std"),
    ).reset_index()
    summary.to_csv(config.TABLE_DIR / "multi_seed_summary.csv", index=False, encoding="utf-8-sig")
    return frame, summary


if __name__ == "__main__":
    results, summary = run()
    print(results)
    print(summary)
