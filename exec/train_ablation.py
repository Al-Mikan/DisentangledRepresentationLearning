import json
import yaml
import optuna
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from train_optuna import objective
from train_core import cleanup_memory


# === Helper functions ===

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_baseline_json(run_dir: Path) -> dict:
    """Optunaのベスト結果（baseline_config.json）を読み込んで統合"""
    baseline_path = run_dir / "baseline_config.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"❌ baseline_config.json not found in {run_dir}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = data.get("params", {}).copy()
    merged.update(data.get("user_attrs", {}))
    return merged


def merge_config_in_memory(yaml_config: dict, json_config: dict) -> dict:
    """YAML設定にbaseline JSONを上書き（ネスト対応）"""
    def recursive_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = recursive_merge(result[k], v)
            else:
                result[k] = v
        return result

    return recursive_merge(yaml_config, json_config)


# === Main function ===

def run_optuna_ablation(cfg_path: str, abl_path: str, run_dir_manual: str):
    """
    ablation.ymlに書かれた複数パターンを、
    Optunaの固定Trialでまとめて実行する。
    """
    # === 設定ファイル読み込み ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    run_dir = Path(run_dir_manual)

    # === baseline_config.json読み込み ===
    baseline_params = load_baseline_json(run_dir)
    print(f"✅ Loaded baseline parameters from {run_dir}/baseline_config.json")

    # === baselineとconfig_search.ymlをマージ ===
    merged_config = merge_config_in_memory(base_yaml, baseline_params)

    # === データ読み込み ===
    datatype = merged_config.get("datatype", "animalkingdom")
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    # === Study作成 ===
    study_name = "ablation_fixed_trials"
    study = optuna.create_study(direction="maximize", study_name=study_name)

    # === AblationパターンをOptunaにキュー追加 ===
    enqueue_count = 0
    for key, values in ab_yaml.items():
        if not isinstance(values, list):
            continue
        for v in values:
            trial_params = merged_config.copy()
            trial_params[key] = v
            study.enqueue_trial(trial_params)
            enqueue_count += 1
            print(f"🧩 Enqueued: {key} = {v}")

    print(f"\n🚀 Enqueued {enqueue_count} fixed ablation trials")

    # === Ablation出力ディレクトリ ===
    ablation_dir = run_dir / "ablation_optuna"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    # === 一括実行 ===
    study.optimize(
        lambda trial: objective(
            trial,
            full_df,
            le_act,
            le_sp,
            results_root=ablation_dir,
            search_space=None,  # Ablation用（固定Trial）
        ),
        n_trials=enqueue_count,
        gc_after_trial=True,
    )

    # === 結果出力 ===
    print("\n🎯 All Ablation Trials Completed!\n")
    for t in study.get_trials():
        print(f"[Trial #{t.number}] {t.params} => {t.value}")
        if "model_save_path" in t.user_attrs:
            print(f"  📦 Model: {t.user_attrs['model_save_path']}")

    cleanup_memory()


# === Entry point ===

if __name__ == "__main__":
    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual="train_result/2025-11-11/run_001",
    )
