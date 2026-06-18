from __future__ import annotations

import pandas as pd

try:
    from . import config
except ImportError:
    import config


def _safe_read(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _first_or_none(frame: pd.DataFrame, mask) -> dict:
    subset = frame[mask]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def run() -> pd.DataFrame:
    main_df = _safe_read(config.MAIN_RESULTS_CSV)
    rep_df = _safe_read(config.PRIVACY_RESULTS_CSV)
    upd_df = _safe_read(config.TABLE_DIR / "update_level_attack_results.csv")
    dual_df = _safe_read(config.TABLE_DIR / "dual_grl_results.csv")
    dp_df = _safe_read(config.TABLE_DIR / "dp_sweep_results.csv")
    grl_df = _safe_read(config.TABLE_DIR / "grl_alpha_sweep.csv")

    rows = []

    def add_row(method: str, metric_item: dict, rep_key: str | None = None, update_key: str | None = None):
        if not metric_item:
            return
        rep_item = _first_or_none(rep_df, rep_df["method"] == rep_key) if (rep_key and not rep_df.empty and "method" in rep_df.columns) else {}
        upd_item = _first_or_none(
            upd_df,
            (upd_df["method"] == (update_key or method)) & (upd_df["attack_model"] == "LogisticRegression"),
        ) if (not upd_df.empty and {"method", "attack_model"}.issubset(upd_df.columns)) else {}
        rows.append({
            "method": method,
            "accuracy": metric_item.get("accuracy"),
            "uar": metric_item.get("uar"),
            "macro_f1": metric_item.get("macro_f1"),
            "representation_speaker_attack_acc": metric_item.get("speaker_attack_acc", rep_item.get("speaker_attack_acc")),
            "representation_gender_attack_acc": metric_item.get("gender_attack_acc", rep_item.get("gender_attack_acc")),
            "update_speaker_attack_acc": upd_item.get("speaker_attack_acc"),
            "update_gender_attack_acc": upd_item.get("gender_attack_acc"),
            "params": metric_item.get("params"),
            "training_time": metric_item.get("training_time"),
        })

    if not main_df.empty and "method" in main_df.columns:
        add_row("Centralized", _first_or_none(main_df, main_df["method"] == "Centralized"), "Centralized")
        add_row("FedAvg", _first_or_none(main_df, main_df["method"] == "FedAvg"), "FedAvg", "FedAvg")

    if not grl_df.empty:
        grl_nonzero = grl_df[grl_df["alpha"] > 0].copy()
        best_grl = (grl_nonzero if not grl_nonzero.empty else grl_df).sort_values("macro_f1", ascending=False).iloc[0].to_dict()
        add_row("Best FedAvg+GRL", best_grl, None, "FedAvg+GRL(best)")

    if not dual_df.empty:
        best_dual = dual_df.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
        add_row("Best FedAvg+Dual-GRL", best_dual, None)

    if not dp_df.empty:
        best_dp = dp_df.sort_values("macro_f1", ascending=False).iloc[0].to_dict()
        add_row("Best FedAvg+DP", best_dp, None, "FedAvg+DP(best)")

    if not main_df.empty and "method" in main_df.columns:
        fedpriv_rows = main_df[main_df["method"].astype(str).str.contains("FedPriv-SER", na=False)]
        if not fedpriv_rows.empty:
            add_row("Best FedAvg+GRL+DP", fedpriv_rows.sort_values("macro_f1", ascending=False).iloc[0].to_dict(), None, "Best FedAvg+GRL+DP")

    frame = pd.DataFrame(rows)
    frame.to_csv(config.TABLE_DIR / "final_selected_results.csv", index=False, encoding="utf-8-sig")
    return frame


if __name__ == "__main__":
    print(run())
