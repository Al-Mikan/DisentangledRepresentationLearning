import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import optuna
import torch
from optuna.trial import TrialState, FrozenTrial
try:
    from optuna.storages import InMemoryStorage
except Exception:
    InMemoryStorage = None
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import shutil

# === 自作モジュール ===
from utils import set_seed
from train_core import cleanup_memory
from train_optuna import load_config, objective


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Global settings
# -------------------------------
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 30
DEFAULT_BATCH_SIZE: int = 64
N_TRIALS_PER_STUDY: int = 60

try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


# -------------------------------
# メイン実行ブロック
# -------------------------------
def main() -> None:
    storage_obj = InMemoryStorage() if InMemoryStorage is not None else None

    # === データ読み込み ===
    print("📂 Loading initial data...")
    datatype = "animalkingdom"
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])
    print("✅ Data loaded.")

    # === 設定ファイル読み込み ===
    yml_path = str(Path(__file__).with_name("config_search.yml"))
    search_space = None
    if load_config is not None and Path(yml_path).exists():
        try:
            search_space = load_config(yml_path)
        except Exception as e:
            print(f"⚠️ Failed to load yml search space: {e}")

    seed = int((search_space or {}).get("seed", 42))
    set_seed(seed)
    N_TRIALS_LOCAL = int((search_space or {}).get("n_trials", N_TRIALS_PER_STUDY))

    # === ディレクトリ構築 ===
    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_root = Path((search_space or {}).get("output_root", "./train_result"))
    date_root = out_root / date_dir
    date_root.mkdir(parents=True, exist_ok=True)

    # run_xxx 自動採番
    existing_runs = [p for p in date_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_idx = max([int(p.name.split("_")[-1]) for p in existing_runs], default=0) + 1
    run_dir = date_root / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_root = run_dir

    # === Optuna Study ===
    today = datetime.now().strftime("%m%d")
    study_name = (search_space or {}).get("experiment_name", f"disentangle-study-{today}")
    print(f"\n===== Starting Optuna Study ({study_name}) =====")

    study = optuna.create_study(
        direction="maximize",
        storage=storage_obj,
        study_name=study_name,
        load_if_exists=False,
    )

    study.optimize(
        lambda trial: objective(trial, full_df, le_act, le_sp, results_root, search_space),
        n_trials=N_TRIALS_LOCAL,
        gc_after_trial=True,
    )

    cleanup_memory()

    # === 結果出力 ===
    print(f"\n--- Best Trial ---")
    print(f"Value (combined ARI/NMI): {study.best_value:.6f}")
    print(f"Params: {study.best_trial.params}")

    # === トップ100結果まとめ ===
    trials = [(t, study_name) for t in study.get_trials(states=[TrialState.COMPLETE])]
    trials_sorted = sorted(trials, key=lambda x: x[0].value, reverse=True)[:100]

    summary_path = results_root / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Optuna Trial Summary (Top 100 by combined ARI/NMI) ===\n\n")
        for rank, (trial, _) in enumerate(trials_sorted, 1):
            f.write(f"[{rank:03d}] metric={trial.value:.6f} | loss_type={trial.user_attrs.get('loss_type')}\n")
            f.write(f"    trial_number: {trial.number}\n")
            f.write(f"    params      : {trial.params}\n")
            if "model_save_path" in trial.user_attrs:
                f.write(f"    model_path  : {trial.user_attrs['model_save_path']}\n")
            f.write("\n")
    print(f"✅ Saved summary to {summary_path}")

    # === baseline_config.json 書き出し ===
    best_trial = study.best_trial
    baseline_cfg = {
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
    }
    baseline_path = results_root / "baseline_config.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_cfg, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved baseline config → {baseline_path}")

    # === checkpoints 削除 ===
    checkpoints_root = results_root / "checkpoints"
    if checkpoints_root.exists():
        try:
            shutil.rmtree(checkpoints_root)
            print(f"🧹 Removed checkpoints directory: {checkpoints_root}")
        except Exception as e:
            print(f"⚠️ Failed to remove checkpoints: {e}")

    # === alpha_logs 整理 ===
    alpha_tmp_root = results_root / "alpha_logs_tmp"
    alpha_out_dir = results_root / "alpha_logs"
    if alpha_tmp_root.exists():
        try:
            alpha_out_dir.mkdir(parents=True, exist_ok=True)
            for trial_dir in alpha_tmp_root.glob("trial_*"):
                trial_num = trial_dir.name.split("_")[-1]
                if trial_num.isdigit() and int(trial_num) == best_trial.number:
                    for f in trial_dir.glob("*.npy"):
                        shutil.copy2(f, alpha_out_dir / f.name)
                shutil.rmtree(trial_dir, ignore_errors=True)
            alpha_tmp_root.rmdir()
            print(f"✅ Consolidated alpha logs to {alpha_out_dir}")
        except Exception as e:
            print(f"⚠️ Alpha consolidation failed: {e}")

    # === 実行メモ作成 ===
    print("\n🗒️ Let's record your experiment note!")
    note = input("💬 この実験の目的・メモを書いてください（空Enterでスキップ）: ").strip()

    # 設定情報まとめ（JSON形式で整形して追記）
    run_info = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "seed": seed,
        "datatype": datatype,
        "n_trials": N_TRIALS_LOCAL,
        "config_yml_path": yml_path,
        "search_space": search_space,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_model_path": best_trial.user_attrs.get("model_save_path", None),
        "optuna_version": optuna.__version__,
        "torch_version": torch.__version__,
    }

    run_note_path = results_root / "run_note.txt"
    with open(run_note_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Note ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run: {results_root.name}\n\n")
        f.write(note + "\n" if note else "(No note recorded)\n")

        f.write("\n\n=== Run Configuration ===\n")
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved your note and config → {run_note_path}")


if __name__ == "__main__":
    main()
