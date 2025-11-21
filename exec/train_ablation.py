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
    ablation.ymlに書かれた複数パターンをOptunaの固定Trialとして実行。
    各key/valueごとにフォルダを分けてモデル保存。
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

    # --- 🔧 adversarialキーを明示的に上書き（baseline優先） ---
    adv = None
    if "adversarial" in baseline_params:
        adv = baseline_params["adversarial"]
    elif "adversarial" in baseline_params.get("user_attrs", {}):
        adv = baseline_params["user_attrs"]["adversarial"]

    # baselineが文字列型なら、それを優先してYAML上書き
    if isinstance(adv, str):
        merged_config["adversarial"] = adv


    

    # === データ読み込み ===
    datatype = merged_config.get("datatype", "animalkingdom")
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    # === Study作成 ===
    study = optuna.create_study(direction="maximize", study_name="ablation_fixed_trials")

    # === ablationパターン列挙 ===
    ablation_specs = []
    for key, values in ab_yaml.items():
        if not isinstance(values, list):
            continue
        for v in values:
            ablation_specs.append((key, v))

    n_trials = len(ablation_specs)
    print(ablation_specs)
    print(f"\n🚀 Enqueued {n_trials} ablation trials")

    # === 一括実行 ===
    def ablation_objective(trial: optuna.trial.Trial):
        idx = trial.number
        if idx >= n_trials:
            raise optuna.TrialPruned()

        key, value = ablation_specs[idx]
        print(f"\n🔎 [Trial {idx}] Running ablation: {key} = {value}")

            # === 各Trial専用のsearch_space生成 ===
        local_config = dict(merged_config)
        local_config[key] = value

        # --- 🔒 リスト型パラメータを固定化（adversarialなどが壊れないように） ---
        for k, v in list(local_config.items()):
            if isinstance(v, list):
                if k in baseline_params and isinstance(baseline_params[k], (str, float, int)):
                    local_config[k] = baseline_params[k]
                else:
                    local_config[k] = v[0]

        # === 🧩 実際に使用する設定を可視化 ===
        print("🧩 Effective training parameters:")
        for k, v in sorted(local_config.items()):
            if k in ["train_mode", "adversarial", "loss_type", "flow_preprocessing", 
                    "triplet_margin", "lambda_adv", "lr_enc", "lr_disc"]:
                print(f"   {k}: {v}")
        print("")

        # === 各Trial専用のsearch_space生成 ===
        local_config = dict(merged_config)
        local_config[key] = value

        # === 出力ディレクトリ作成（key/valueごと） ===
        ablation_dir = run_dir / "ablation" / f"{key}" / str(value)
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # === objective実行 ===
        result = objective(
            trial,
            full_df,
            le_act,
            le_sp,
            results_root=ablation_dir,
            search_space=local_config,
        )

        # === モデルパスの整形 ===
        model_path = trial.user_attrs.get("model_save_path", None)
        if model_path:
            renamed = ablation_dir / f"{key}_{value}_best.pth"

            dst_path = Path(renamed)

            # 既に存在する場合は削除してからリネーム
            if dst_path.exists():
                dst_path.unlink()
            
            Path(model_path).rename(dst_path)
            trial.set_user_attr("model_save_path", str(dst_path))

        print(f"✅ Completed ablation {key}={value} → score={result:.4f}")
        return result

    # === 実行 ===
    study.optimize(ablation_objective, n_trials=n_trials, gc_after_trial=True)

    # === 結果出力 ===
    print("\n🎯 All Ablation Trials Completed!\n")
    for t in study.get_trials():
        print(f"📦 {t.user_attrs.get('model_save_path', '(no model)')}")
        print(f"  ➤ Score: {t.value:.4f}\n")

    cleanup_memory()


# === Entry point ===

if __name__ == "__main__":
    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual="train_result/2025-11-11/run_001",
    )
