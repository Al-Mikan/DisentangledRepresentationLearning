import torch
import yaml
import json
from copy import deepcopy
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train_core import train_with_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_baseline_json(run_dir: Path) -> dict:
    """Optunaのベスト結果（baseline_config.json）を読み込んで統合"""
    baseline_path = run_dir / "baseline_config.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"❌ baseline_config.json not found in {run_dir}")
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # params（Optuna探索パラメータ）と user_attrs（学習モードなど）を統合
    merged = data.get("params", {}).copy()
    merged.update(data.get("user_attrs", {}))
    return merged

def run_ablation(cfg_path, abl_path, full_df, le_act, le_sp, run_dir_manual: str = None):
    # === baseline設定（共通） ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)

    output_root = Path(base_yaml.get("output_root", "./train_result"))

    # ✅ ここで手動指定を優先
    if run_dir_manual is not None:
        run_dir = Path(run_dir_manual)
        if not run_dir.exists():
            raise FileNotFoundError(f"❌ 指定された run_dir が存在しません: {run_dir}")
        latest_run_dir = run_dir
        print(f"🧩 Using manually specified run dir: {latest_run_dir}")
    else:
        # 自動で最新のrunを探す
        run_dirs = sorted(output_root.glob("run_*"))
        if not run_dirs:
            raise FileNotFoundError(f"❌ No run_xxx directory found under {output_root}")
        latest_run_dir = run_dirs[-1]
        print(f"🧩 Using latest run dir: {latest_run_dir}")

    # === baseline_config.json 読み込み ===
    baseline_params = load_baseline_json(latest_run_dir)
    print(f"✅ Loaded baseline parameters from {latest_run_dir}/baseline_config.json")

    # === ablation/ フォルダ作成 ===
    ablation_root = latest_run_dir / "ablation"
    ablation_root.mkdir(parents=True, exist_ok=True)

    # === 各キーごとに ablation/{key}/value/ を作成して実行 ===
    for key, values in ab_yaml.items():
        if not isinstance(values, list):
            continue

        key_name = "mode" if key == "train_mode" else key
        key_dir = ablation_root / key_name
        key_dir.mkdir(parents=True, exist_ok=True)

        for v in values:
            cfg = deepcopy(base_yaml)
            cfg.update(baseline_params)
            cfg[key] = v
            
            ab_dir = key_dir / str(v)
            ab_dir.mkdir(parents=True, exist_ok=True)
            cfg["output_root"] = str(ab_dir)

            print(f"🚀 Running ablation: {key} = {v}")
            val_loss, model_path = train_with_config(cfg, full_df, le_act, le_sp, ab_dir)



if __name__ == "__main__":
    datatype = "animalkingdom"
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    # ✅ 手動で run ディレクトリを指定
    run_ablation(
        "exec/config_search.yml",
        "exec/ablation.yml",
        full_df,
        le_act,
        le_sp,
        run_dir_manual="train_result/2025-11-11/run_001"
    )