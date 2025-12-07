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
    """YAML 設定に baseline JSON をネスト対応で上書き"""
    def recursive_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = recursive_merge(result[k], v)
            else:
                result[k] = v
        return result
    return recursive_merge(yaml_config, json_config)


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

    # --- adversarial キーを baseline 優先で上書き ---
    adv = None
    if "adversarial" in baseline_params:
        adv = baseline_params["adversarial"]
    elif "adversarial" in baseline_params.get("user_attrs", {}):
        adv = baseline_params["user_attrs"]["adversarial"]

    if isinstance(adv, str):
        merged_config["adversarial"] = adv

    # === データ読み込み ===
    # config_search.yml 由来の train_csv を優先して使用
    train_csv = merged_config.get("train_csv", "./label/animalkingdom/train/labels.csv")
    full_df = pd.read_csv(train_csv)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    # === Study 作成 ===
    study = optuna.create_study(direction="maximize", study_name="ablation_fixed_trials")

    # === ablation パターンを配列化 ===
    ablation_specs = []
    for key, values in ab_yaml.items():
        if isinstance(values, list):
            for v in values:
                ablation_specs.append((key, v))

    n_trials = len(ablation_specs)
    print(ablation_specs)
    print(f"\n🚀 Enqueued {n_trials} ablation trials")

    # ============================================================
    # ablation_objective（注意：内部関数）
    # ============================================================

    def ablation_objective(trial: optuna.trial.Trial):
        idx = trial.number
        if idx >= n_trials:
            raise optuna.TrialPruned()

        key, value = ablation_specs[idx]
        print(f"\n🔎 [Trial {idx}] Running ablation: {key} = {value}")

        # === Trial 専用の設定（deep copy）===
        local_config = json.loads(json.dumps(merged_config))
        local_config[key] = value

        # --- list を固定化（baseline を優先）---
        for k, v in list(local_config.items()):
            if isinstance(v, list):
                if k in baseline_params and isinstance(baseline_params[k], (str, float, int, bool)):
                    local_config[k] = baseline_params[k]
                else:
                    local_config[k] = v[0]

        # === 使用する設定の表示 ===
        print("🧩 Effective training parameters:")
        debug_keys = [
            "train_mode", "adversarial", "loss_type",
            "flow_preprocessing", "triplet_margin",
            "lambda_adv", "lr_enc", "lr_disc",
        ]
        for k, v in sorted(local_config.items()):
            if k in debug_keys:
                print(f"   {k}: {v}")
        print("")

        # === 出力ディレクトリ ===
        ablation_dir = run_dir / "ablation" / f"{key}" / str(value)
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # === objective 実行 ===
        try:
            result = objective(
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
            trial.set_user_attr("exception", str(e))
            trial.set_user_attr("traceback", tb)
            raise optuna.TrialPruned(f"objective failed: {e}")

        # === モデル保存 ===
        model_path = trial.user_attrs.get("model_save_path")
        if model_path:
            renamed = ablation_dir / f"{key}_{value}_best.pth"
            dst_path = Path(renamed)

            if dst_path.exists():
                dst_path.unlink()  # 古いファイル削除

            Path(model_path).rename(dst_path)
            trial.set_user_attr("model_save_path", str(dst_path))

        print(f"✅ Completed ablation {key}={value} → score={result:.4f}")
        return result

    # ============================================================
    # 実行
    # ============================================================

    study.optimize(ablation_objective, n_trials=n_trials, gc_after_trial=True)

    # === 結果出力 ===
    print("\n🎯 All Ablation Trials Completed!\n")
    for t in study.get_trials():
        print(f"📦 {t.user_attrs.get('model_save_path', '(no model)')}")
        print(f"  ➤ Score: {t.value:.4f}\n")

    cleanup_memory()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_optuna_ablation(
        cfg_path="exec/config_search.yml",
        abl_path="exec/ablation.yml",
        run_dir_manual="train_result/2025-11-26/run_001",
    )
