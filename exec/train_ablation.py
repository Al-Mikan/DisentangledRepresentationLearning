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
    """Optuna のベスト結果（baseline_config.json）を読み込む"""
    baseline_path = run_dir / "baseline_config.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"❌ baseline_config.json not found in {run_dir}")

    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged = data.get("params", {}).copy()
    merged.update(data.get("user_attrs", {}))
    return merged


def merge_config_in_memory(yaml_config: dict, json_config: dict) -> dict:
    """
    json_config をベースにしつつ、yaml_config で上書きする。
    → yaml が強い（優先）
    """
    def recursive_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if (
                isinstance(v, dict)
                and k in result
                and isinstance(result[k], dict)
            ):
                result[k] = recursive_merge(result[k], v)
            else:
                result[k] = v
        return result

    return recursive_merge(json_config, yaml_config)


def save_json(path: Path, data: dict):
    """JSON で保存 (pretty-print)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_result_log(run_dir: Path, record: dict):
    """JSONL に結果を追記"""
    result_path = run_dir / "logs" / "trial_results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Main Ablation Function
# ============================================================

def run_optuna_ablation(cfg_path: str, abl_path: str, run_dir_manual: str):
    """
    ablation.yml の複数パターンを固定 Trial として実行し
    各 key/value ごとにモデルファイルを保存する
    """

    # === 設定ファイル読み込み ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    run_dir = Path(run_dir_manual)

    # === baseline_config.json 読み込み ===
    baseline_params = load_baseline_json(run_dir)
    print(f"✅ Loaded baseline parameters from {run_dir}/baseline_config.json")

    # === baseline と config_search.yml をマージ ===
    merged_config = merge_config_in_memory(base_yaml, baseline_params)

    # === データ読み込み ===
    datatype = merged_config.get("datatype", "animalkingdom")
    train_csv = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(train_csv)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    # === Study 作成 ===
    study = optuna.create_study(direction="maximize", study_name="ablation_fixed_trials")

    # === ablation パターン整理 ===
    ablation_specs = []
    for key, values in ab_yaml.items():
        if isinstance(values, list):
            for v in values:
                ablation_specs.append((key, v))

    n_trials = len(ablation_specs)
    print(f"\n🚀 Enqueued {n_trials} ablation trials")
    print(ablation_specs)

    # ============================================================
    # ablation_objective
    # ============================================================

    def ablation_objective(trial: optuna.trial.Trial):
        idx = trial.number
        if idx >= n_trials:
            raise optuna.TrialPruned()

        key, value = ablation_specs[idx]
        print(f"\n🔎 [Trial {idx}] Running ablation: {key} = {value}")

        # === Trial 専用設定 ===
        local_config = json.loads(json.dumps(merged_config))
        local_config[key] = value

        # --- list は baseline を優先 (固定化)---
        for k, v in list(local_config.items()):
            if isinstance(v, list):
                if k in baseline_params and isinstance(baseline_params[k], (str, float, int, bool)):
                    local_config[k] = baseline_params[k]
                else:
                    local_config[k] = v[0]

        # === Config をログに保存 ===
        cfg_log = run_dir / "logs" / f"trial_{idx:03d}_{key}_{value}_config.json"
        save_json(cfg_log, local_config)
        print(f"📝 Saved config → {cfg_log}")

        # === 出力ディレクトリ ===
        ablation_dir = run_dir / "ablation" / f"{key}" / str(value)
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # === objective 実行 ===
        try:
            score = objective(
                trial,
                full_df,
                le_act,
                le_sp,
                results_root=ablation_dir,
                search_space=local_config,
            )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("❌ objective() failed:\n", tb)

            append_result_log(run_dir, {
                "trial": idx,
                "key": key,
                "value": value,
                "score": -1.0,
                "error": str(e),
            })
            return -1.0

        # === モデル保存 ===
        model_path = trial.user_attrs.get("model_save_path")
        if model_path:
            renamed = ablation_dir / f"{key}_{value}_best.pth"
            if renamed.exists():
                renamed.unlink()
            Path(model_path).rename(renamed)
            trial.set_user_attr("model_save_path", str(renamed))

        # === 結果ログ ===
        append_result_log(run_dir, {
            "trial": idx,
            "key": key,
            "value": value,
            "score": float(score),
            "model": trial.user_attrs.get("model_save_path"),
        })

        print(f"✅ Completed ablation {key}={value} → score={score:.4f}")
        return score

    # ============================================================
    # 全 Trial 実行
    # ============================================================

    study.optimize(ablation_objective, n_trials=n_trials, gc_after_trial=True)

    print("\n🎯 All Ablation Trials Completed!\n")
    cleanup_memory()


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
        raise RuntimeError("❌ No run_* directories found under train_result/")

    latest_run = all_runs[0]
    print(f"▶ Auto-selected latest run directory: {latest_run}")

    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual=str(latest_run),
    )
