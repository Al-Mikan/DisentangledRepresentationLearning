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


def merge_configs(base_yaml: dict, baseline_json: dict) -> dict:
    """
    config_search.yml のデフォルト設定をベースに、
    baseline_config.json の値で上書きして統合。
    """
    merged = {}

    def recursive_merge(yaml_dict, baseline_dict):
        result = {}
        for k, v in yaml_dict.items():
            if isinstance(v, dict):
                result[k] = recursive_merge(v, baseline_dict.get(k, {}))
            else:
                # baselineにキーがあれば上書き、なければYAMLの値
                result[k] = baseline_dict.get(k, v)
        # baselineにあってyamlにないキーも足す
        for k, v in baseline_dict.items():
            if k not in result:
                result[k] = v
        return result

    merged = recursive_merge(base_yaml, baseline_json)
    return merged


# === Main function ===

def run_optuna_ablation(cfg_path: str, abl_path: str, run_dir_manual: str):
    """
    Ablation.ymlに書かれた複数パターンを、Optunaの固定Trialでまとめて実行。
    """
    # === 設定ファイル読み込み ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    output_root = Path(base_yaml.get("output_root", "./train_result"))
    run_dir = Path(run_dir_manual)

    # === baseline_config.json読み込み ===
    baseline_params = load_baseline_json(run_dir)
    print(f"✅ Loaded baseline parameters from {run_dir}/baseline_config.json")

    # === baselineとconfig_search.ymlをマージ ===
    merged_base = merge_configs(base_yaml, baseline_params)

    # === データ読み込み ===
    datatype = merged_base.get("datatype", "animalkingdom")
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
            trial_params = merged_base.copy()  # ← 統合済み設定を基礎に
            trial_params[key] = v
            study.enqueue_trial(trial_params)
            enqueue_count += 1
            print(f"🧩 Enqueued: {key} = {v}")

    print(f"\n🚀 Enqueued {enqueue_count} fixed ablation trials")

    # === Ablationルートを保存先に設定 ===
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
            search_space=None,  # Ablation用
        ),
        n_trials=enqueue_count,
        gc_after_trial=True,
    )

    # === 結果出力 ===
    print("\n🎯 All Ablation Trials Completed!\n")
    for t in study.get_trials():
        print(f"[Trial #{t.number}] {t.params} => {t.value:.6f}")
        if "model_save_path" in t.user_attrs:
            print(f"  📦 Model: {t.user_attrs['model_save_path']}")

    cleanup_memory()


# === Entry point ===

if __name__ == "__main__":
    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual="train_result/2025-11-11/run_001"
    )
