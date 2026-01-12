import json
import yaml
import optuna
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from train_optuna import objective
from train_core import cleanup_memory

# ============================================================
# Helper functions
# ============================================================

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_baseline_json(run_dir: Path) -> dict:
    baseline_path = run_dir / "baseline_config.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"❌ baseline_config.json not found in {run_dir}")

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged = data.get("params", {}).copy()
    merged.update(data.get("user_attrs", {}))

    # ablation では不要な情報を削除
    for k in [
        "model_save_path",
        "best_epoch",
        "epochs_run",
        "cv_scores",
        "cv_mean",
        "exception",
        "traceback",
    ]:
        merged.pop(k, None)

    return merged


def merge_config_in_memory(yaml_config: dict, json_config: dict) -> dict:
    """json_config をベースに yaml_config で上書き（yaml優先）"""
    def recursive_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = recursive_merge(result[k], v)
            else:
                result[k] = v
        return result

    return recursive_merge(json_config, yaml_config)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_result_log(run_dir: Path, record: dict):
    result_path = run_dir / "logs" / "trial_results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_dataset(config: dict) -> pd.DataFrame:
    """Configに基づいてデータセットを読み込む（Augmentation対応）"""
    use_aug = config.get("use_augmentation", False) # 修正: この行が必要
    # Augmentation有効なら両方を結合
    if use_aug:
        std_paths = config.get("train_label_paths")
        if isinstance(std_paths, str): std_paths = [std_paths]
        if std_paths is None: std_paths = []

        aug_paths = config.get("aug_train_label_paths")
        if isinstance(aug_paths, str): aug_paths = [aug_paths]
        if aug_paths is None: aug_paths = []

        target_paths = std_paths + aug_paths
        print(f"  → Loading STANDARD + AUGMENTED train CSVs: {target_paths}")
    else:
        target_paths = config.get("train_label_paths")
        if target_paths is None:
            raise RuntimeError("❌ train_label_paths が設定されていません")
        print(f"  → Loading STANDARD train CSVs: {target_paths}")

    if isinstance(target_paths, str):
        target_paths = [target_paths]

    dfs = []
    for p in target_paths:
        df_i = pd.read_csv(p)
        dfs.append(df_i)
        
    if not dfs:
        raise RuntimeError("No dataframe loaded.")
        
    return pd.concat(dfs, ignore_index=True)


# ============================================================
# Main Ablation Function
# ============================================================

def run_optuna_ablation(cfg_path: str, abl_path: str, run_dir_manual: str):

    # === 設定読み込み ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    run_dir = Path(run_dir_manual)

    baseline_params = load_baseline_json(run_dir)
    print(f"✅ Loaded baseline parameters from {run_dir}/baseline_config.json")

    merged_config = merge_config_in_memory(base_yaml, baseline_params)

    # === Study（ログ用）===
    study = optuna.create_study(direction="maximize", study_name="ablation_fixed_trials")

    # === ablation specs 展開 ===
    ablation_specs = []
    for key, values in ab_yaml.items():
        if isinstance(values, list):
            for v in values:
                ablation_specs.append((key, v))

    print(f"\n🚀 Starting {len(ablation_specs)} ablation trials")

    # ============================================================
    # Python loop で完全制御（最重要）
    # ============================================================
    
    # WandB Project Name Override
    # 形式: optuna_MMDD_run_xxx_ablation
    date_str = run_dir.parent.name.replace("-", "")[4:8]  # 2026-01-12 -> 0112
    run_num = run_dir.name.split("_")[1] if "_" in run_dir.name else "000"  # run_001_xxx -> 001
    merged_config["project_name"] = f"optuna_{date_str}_run_{run_num}_ablation"

    for idx, (key, value) in enumerate(ablation_specs):
        print(f"\n🔎 [Ablation {idx+1}/{len(ablation_specs)}] {key} = {value}")

        # --- 固定設定を構築 ---
        local_config = json.loads(json.dumps(merged_config))
        local_config[key] = value

        if key == "train_mode":
            print("⚙️ Force adversarial = off (train_mode ablation)")
            local_config["adversarial"] = "off"

        # list 値を固定化 (ただしパス系はリストのまま保持)
        skip_keys = {"train_label_paths", "aug_train_label_paths", "test_label_paths"}
        for k, v in list(local_config.items()):
            if k in skip_keys:
                continue  # パスはリストのまま
            if isinstance(v, list):
                if k in baseline_params and isinstance(baseline_params[k], (str, float, int, bool)):
                    local_config[k] = baseline_params[k]
                else:
                    local_config[k] = v[0]

        # --- データ読み込み (Augmentation設定反映のためここで実施) ---
        full_df = load_dataset(local_config)
        print(f"  → Total train rows: {len(full_df)}")

        # --- Label Encoder ---
        le_sp = LabelEncoder()
        le_sp.fit(full_df["species"])
        le_act = LabelEncoder()
        le_act.fit(full_df["action"])

        ablation_dir = run_dir / "ablation" / key / str(value)
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # --- single-shot objective ---
        def single_objective(trial):

            cfg_log = run_dir / "logs" / f"trial_{trial.number:03d}_{key}_{value}_config.json"
            save_json(cfg_log, local_config)

            try:
                score = objective(
                    trial,
                    full_df,
                    le_act,
                    le_sp,
                    results_root=ablation_dir,
                    next_idx=idx,             
                    search_space=None,        # 探索しない
                    fixed_config=local_config # 完全固定
                )
            except Exception as e:
                import traceback
                print(f"❌ Ablation {key}={value} failed:\n{traceback.format_exc()}")
                append_result_log(run_dir, {
                    "trial": trial.number,
                    "key": key,
                    "value": value,
                    "score": -1.0,
                    "error": str(e),
                })
                return -1.0

            model_path = trial.user_attrs.get("model_save_path")
            renamed = None
            if model_path:
                renamed = ablation_dir / f"{key}_{value}_best.pth"
                if renamed.exists():
                    renamed.unlink()
                Path(model_path).rename(renamed)
                trial.set_user_attr("model_save_path", str(renamed))

            append_result_log(run_dir, {
                "trial": trial.number,
                "key": key,
                "value": value,
                "score": float(score),
                "model": str(renamed) if renamed else None,
            })

            return score

        # --- 1 trial だけ実行 ---
        study.optimize(single_objective, n_trials=1, gc_after_trial=True)
        cleanup_memory()

    print("\n🎯 All Ablation Trials Completed!\n")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
        if not run_dir.exists():
            raise RuntimeError(f"❌ Specified run directory not found: {run_dir}")

        run_optuna_ablation(
            cfg_path="exec/config_search.yml",
            abl_path="exec/ablation.yml",
            run_dir_manual=str(run_dir),
        )
        sys.exit(0)

    all_runs = sorted(
        [d for d in Path("train_result").glob("**/run_*") if d.is_dir()],
        reverse=True
    )

    if not all_runs:
        raise RuntimeError("❌ No run_* directories found")

    latest_run = all_runs[0]
    print(f"▶ Auto-selected latest run directory: {latest_run}")

    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual=str(latest_run),
    )
