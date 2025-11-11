import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
import importlib.util

import numpy as np
import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb
from optuna.trial import TrialState,FrozenTrial
try:
    from optuna.storages import InMemoryStorage 
except Exception:
    InMemoryStorage = None
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import shutil
from urllib import request as urlrequest, error as urlerror
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering


# 必要なファイルをインポート
from utils import set_seed
from train_core import (
    cleanup_memory,
    build_basename_from_config
)
from train_optuna import load_config, suggest_from_yml, objective
from train_ablation import run_ablation



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Global settings and small utils
# -------------------------------
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 30
DEFAULT_BATCH_SIZE: int = 64
TOP_K_TO_KEEP_PER_LOSS: int = 3
N_TRIALS_PER_STUDY: int = 60

try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass

# --- メイン実行ブロック ---
def main() -> None:

    storage_obj = InMemoryStorage() if InMemoryStorage is not None else None

    print("Loading initial data...")
    datatype = 'animalkingdom'
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df['action'])
    le_sp = LabelEncoder().fit(full_df['species'])
    print("Data loaded.")

    yml_path = str(Path(__file__).with_name("config_search.yml"))
    search_space = None
    if load_config is not None and Path(yml_path).exists():
        try:
            search_space = load_config(yml_path)
        except Exception as e:
            print(f"⚠️ Failed to load yml search space: {e}")
    # Seed and trials from yml or defaults
    seed = int((search_space or {}).get("seed", 42))
    set_seed(seed)
    N_TRIALS_PER_STUDY_LOCAL = int((search_space or {}).get("n_trials", N_TRIALS_PER_STUDY))

    date_dir = datetime.now().strftime("%Y-%m-%d")
    out_root = Path((search_space or {}).get("output_root", "./train_result"))
    date_root = out_root / date_dir
    date_root.mkdir(parents=True, exist_ok=True)
    # Determine next run id within the date directory
    existing_runs = [p for p in date_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    def _run_index(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 0
    next_idx = 1
    if existing_runs:
        next_idx = max(_run_index(p.name) for p in existing_runs) + 1
    run_dir = date_root / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_root = run_dir

    # Create a single mixed study exploring loss_type as a hyperparameter
    today = datetime.now().strftime("%m%d")
    study_name = (search_space or {}).get("experiment_name", f"disentangle-study-{today}")
    print(f"\n\n===== Starting Optuna Study (mixed loss types) =====")


    study = optuna.create_study(
        direction="maximize",
        storage=(storage_obj if storage_obj is not None else None),
        study_name=study_name,
        load_if_exists=False
    )

    study.optimize(lambda trial: objective(trial, full_df, le_act, le_sp, results_root, search_space), n_trials=N_TRIALS_PER_STUDY_LOCAL, gc_after_trial=True)

    cleanup_memory()

    print(f"\n--- Best Trial (mixed) ---")
    print(f"Value (combined ARI/NMI): {study.best_value}")
    print(f"Params: {study.best_trial.params}")

    print("\n--- Starting model selection/cleanup ---")
    # Collect trials for this run only when using in-memory storage; otherwise, aggregate from storage
    all_trials_info: List[Tuple[FrozenTrial, str]] = [
        (t, study_name) for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    ]
    all_studies = [type("S", (), {"study_name": study_name})]
    
    # -------------------------------
    # 上位N件（全lossタイプ横断）のモデルを保持
    # -------------------------------
    TOP_K = TOP_K_TO_KEEP_PER_LOSS
    best_trials_overall: List[Tuple[FrozenTrial, str]] = sorted(
        all_trials_info, key=lambda info: info[0].value, reverse=True
    )[:TOP_K]

    # -------------------------------
    # 上位100件の試行をファイル出力
    # -------------------------------
    all_trials_sorted = sorted(all_trials_info, key=lambda info: info[0].value, reverse=True)[:100]
    summary_path = results_root / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        title = "Top 100 by metric (combined ARI/NMI)"
        f.write(f"=== Optuna Trial Summary ({title}) ===\n\n")
        for rank, (trial, study_name_) in enumerate(all_trials_sorted, 1):
            f.write(f"[{rank:03d}] loss_type={trial.user_attrs.get('loss_type')} | metric={trial.value:.6f}\n")
            # Prefer user_attr study_name if present, else use collected name
            sn = trial.user_attrs.get("study_name", study_name_)
            f.write(f"    study_name : {sn}\n")
            f.write(f"    trial_number: {trial.number}\n")
            f.write(f"    params      : {trial.params}\n")
            if "model_save_path" in trial.user_attrs:
                f.write(f"    model_path  : {trial.user_attrs['model_save_path']}\n")
            f.write("\n")
    print(f"✅ Saved summary of top 100 trials to {summary_path}")

    # -------------------------------
    # 最終的なbaseline設定を書き出し
    # -------------------------------
    best_trial = study.best_trial
    baseline_cfg = {
        "value": best_trial.value,          # 最終スコア（例：combined ARI/NMI）
        "params": best_trial.params,        # 全ハイパーパラメータ（学習率・λ・batch_sizeなど）
        "user_attrs": best_trial.user_attrs # メタ情報（loss_type, model_pathなど）
    }

    baseline_path = results_root / "baseline_config.json"

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_cfg, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved baseline configuration (JSON) to {baseline_path}")


    # -------------------------------
    # 保存対象モデルパスの抽出
    # -------------------------------
    # パス比較の不一致（"./models" vs "models"）や相対/絶対の差を避けるため、事前にresolveして揃える
    paths_to_keep: Set[Path] = set()
    for t, _study_name in best_trials_overall:
        path_str = t.user_attrs.get("model_save_path")
        if not path_str:
            continue
        try:
            paths_to_keep.add(Path(path_str).resolve())
        except Exception as e:
            print(f"⚠️ Skipping keep-path '{path_str}': {e}")

    deleted_count = 0

    # Copy kept best models into train_result/<date>/best_model with global rank
    best_model_dir = results_root / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    for rank, (t, _sn) in enumerate(best_trials_overall, start=1):
        src_path_str = t.user_attrs.get("model_save_path")
        if not src_path_str:
            continue
        try:
            src_path = Path(src_path_str)
            if not src_path.exists():
                continue
            # Build a readable basename from trial params akin to ablation naming
            params = t.params or {}
            cfg_name: Dict[str, Any] = {
                "loss_type": t.user_attrs.get("loss_type", params.get("loss_type", "unknown")),
                "train_mode": params.get("train_mode", "unknown"),
                "use_mlp": params.get("use_mlp", False),
                "adversarial_mode": t.user_attrs.get("adversarial_mode", params.get("adversarial_mode", "off")),
                "flow_preprocessing": t.user_attrs.get("flow_preprocessing") 
                           or params.get("flow_preprocessing", "normal"),
            }
            base = build_basename_from_config(cfg_name)
            dst_name = f"{base}_rank{rank}_best.pth"
            dst_path = best_model_dir / dst_name
            shutil.copy2(str(src_path), str(dst_path))
        except Exception as e:
            print(f"⚠️ Failed to copy best model for trial #{t.number}: {e}")

    # Delete non-kept checkpoints under results_root/checkpoints/<study>
    checkpoints_root = results_root / "checkpoints"
    # Limit cleanup scope to this study in in-memory mode
    study_names_to_scan = [study_name]
    for sn in study_names_to_scan:
        models_dir = checkpoints_root / sn
        if models_dir.exists():
            for model_path in models_dir.glob("**/*.pth"):
                try:
                    if model_path.resolve() not in paths_to_keep:
                        model_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to process '{model_path}': {e}")
    
    print(f"Cleanup finished. Kept {len(paths_to_keep)} best models overall.")
    print(f"Deleted {deleted_count} other model checkpoints.")

    # -------------------------------
    # checkpoints ディレクトリを完全削除
    # -------------------------------
    checkpoints_root = results_root / "checkpoints"
    if checkpoints_root.exists():
        try:
            shutil.rmtree(checkpoints_root)
            print(f"🧹 Removed entire checkpoints directory: {checkpoints_root}")
        except Exception as e:
            print(f"⚠️ Failed to remove checkpoints directory '{checkpoints_root}': {e}")


    # Consolidate alpha logs: keep all epochs only for global rank-1 trial
    if best_trials_overall:
        top_trial, _sn = best_trials_overall[0]
        alpha_tmp_root = results_root / 'alpha_logs_tmp'
        alpha_out_dir = results_root / 'alpha_logs'
        alpha_out_dir.mkdir(parents=True, exist_ok=True)
        # Move/copy all temp alphas for the top trial into alpha_logs, remove others
        try:
            if alpha_tmp_root.exists():
                for trial_dir in alpha_tmp_root.glob('trial_*'):
                    trial_num_str = trial_dir.name.split('_')[-1]
                    if trial_num_str.isdigit() and int(trial_num_str) == top_trial.number:
                        # move all files to alpha_logs with same names
                        for f in trial_dir.glob('*.npy'):
                            dst = alpha_out_dir / f.name
                            try:
                                shutil.copy2(str(f), str(dst))
                            except Exception as e:
                                print(f"⚠️ Failed to copy alpha '{f}' -> '{dst}': {e}")
                    # remove the temp directory regardless (we will keep only copied ones)
                    try:
                        for f in trial_dir.glob('*.npy'):
                            try:
                                f.unlink()
                            except Exception:
                                pass
                        trial_dir.rmdir()
                    except Exception:
                        pass
                # remove alpha_logs_tmp if empty
                try:
                    alpha_tmp_root.rmdir()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Alpha consolidation failed: {e}")


if __name__ == "__main__":
    main()
