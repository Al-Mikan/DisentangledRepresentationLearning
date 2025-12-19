import json
from pathlib import Path
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
import time

# === 自作モジュール ===
from utils import set_seed
from train_core import cleanup_memory
from train_optuna import load_config, objective


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Global settings
# -------------------------------
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
    yml_path = str(Path(__file__).with_name("config_search.yml"))
    search_space = None
    if Path(yml_path).exists():
        try:
            search_space = load_config(yml_path)
        except Exception as e:
            print(f"⚠️ Failed to load yml search space: {e}")

    seed = int(search_space.get("seed", 42))
    set_seed(seed)
    N_TRIALS_LOCAL = int(search_space.get("n_trials", N_TRIALS_PER_STUDY))

    # --------------------------------------------
    # Load CSV & encoders
    # --------------------------------------------
    print("📂 Loading dataset & encoders...")
    datatype = search_space.get("datatype", "animalkingdom")
    train_csv = f"./label/{datatype}/train/labels.csv"

    full_df = pd.read_csv(train_csv)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()

    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])
    print("✅ CSV loaded & encoders created.")


    # === ディレクトリ構築 ===
    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_root = Path((search_space or {}).get("output_root", "./train_result"))
    date_root = out_root / date_dir
    date_root.mkdir(parents=True, exist_ok=True)

    # run_xxx の自動採番
    existing = [p for p in date_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    next_idx = max([int(p.name.split("_")[-1]) for p in existing], default=0) + 1
    results_root = date_root / f"run_{next_idx:03d}"
    results_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # ★ 実験開始前のメモ
    # --------------------------------------------
    print("\n🗒️ Let's record your experiment note BEFORE training.")
    note = input("💬 この実験の目的・メモを書いてください（空Enterでスキップ）: ").strip()

    run_note_path = results_root / "run_note.txt"
    with open(run_note_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Note (Before Training) ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run: {results_root.name}\n\n")
        f.write(note + "\n" if note else "(No note recorded)\n")

    print(f"✅ Saved initial note → {run_note_path}")

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

    # === Optuna 結果出力 ===
    print("\n--- Best Trial ---")
    print(f"Value (score): {study.best_value:.6f}")
    print(f"Params: {study.best_trial.params}")

    # === トップ100 結果まとめ ===
    trials = [t for t in study.get_trials(states=[TrialState.COMPLETE])]
    trials_sorted = sorted(trials, key=lambda t: t.value, reverse=True)[:100]

    summary_path = results_root / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Optuna Trial Summary (Top 100) ===\n\n")
        for rank, trial in enumerate(trials_sorted, 1):
            f.write(f"[{rank:03d}] score={trial.value:.6f} | loss_type={trial.user_attrs.get('loss_type')}\n")
            f.write(f"    trial_number: {trial.number}\n")
            f.write(f"    params      : {trial.params}\n")
            model_path = trial.user_attrs.get("model_save_path")
            if model_path:
                f.write(f"    model_path  : {model_path}\n")
            f.write("\n")

    print(f"✅ Saved summary → {summary_path}")

    # === baseline_config.json ===
    best_trial = study.best_trial
    baseline_cfg = {
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "config_used": search_space,
    }
    baseline_path = results_root / "baseline_config.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_cfg, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved baseline config → {baseline_path}")

    # === checkpoints 削除 ===
    checkpoints_root = results_root / "checkpoints"
    if checkpoints_root.exists():
        shutil.rmtree(checkpoints_root, ignore_errors=True)
        print("🧹 Removed checkpoints directory")

    # === alpha_logs 整理 ===
    alpha_tmp_root = results_root / "alpha_logs_tmp"
    alpha_out_dir = results_root / "alpha_logs"

    if alpha_tmp_root.exists():
        alpha_out_dir.mkdir(parents=True, exist_ok=True)
        for trial_dir in alpha_tmp_root.glob("trial_*"):
            trial_num = trial_dir.name.split("_")[-1]
            if trial_num.isdigit() and int(trial_num) == best_trial.number:
                for f in trial_dir.glob("*.npy"):
                    shutil.copy2(f, alpha_out_dir / f.name)
            shutil.rmtree(trial_dir, ignore_errors=True)

        alpha_tmp_root.rmdir()
        print(f"✅ Consolidated alpha logs → {alpha_out_dir}")

    # --------------------------------------------
    # ★ 実験終了後に追記
    # --------------------------------------------
    print("\n🧾 Appending summary info to run_note.txt ...")

    run_info = {
        "datetime": datetime.now().isoformat(timespec="seconds"),
        "device": str(DEVICE),
        "seed": seed,
        "datatype": search_space.get("datatype", "unknown"),
        "n_trials": N_TRIALS_LOCAL,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_model_path": best_trial.user_attrs.get("model_save_path"),
        "optuna_version": optuna.__version__,
        "torch_version": torch.__version__,
    }

    with open(run_note_path, "a", encoding="utf-8") as f:
        f.write("\n\n=== Run Configuration (After Training) ===\n")
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"✅ Appended summary → {run_note_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()
    main()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print("\n⏱️ Experiment finished!")
    print(f"   Total time: {h}h {m}m {s}s")