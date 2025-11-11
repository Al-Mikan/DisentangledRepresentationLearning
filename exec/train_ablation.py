import torch
import yaml
import json
import gc
import multiprocessing
from copy import deepcopy
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from train_core import train_with_config, cleanup_memory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Helper functions ===

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

    merged = data.get("params", {}).copy()
    merged.update(data.get("user_attrs", {}))
    return merged


# === Main Ablation runner ===

def run_ablation(cfg_path, abl_path, full_df, le_act, le_sp, run_dir_manual: str = None):
    # === baseline設定 ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    output_root = Path(base_yaml.get("output_root", "./train_result"))

    # === 実行対象 run_dir の決定 ===
    if run_dir_manual is not None:
        run_dir = Path(run_dir_manual)
        if not run_dir.exists():
            raise FileNotFoundError(f"❌ 指定された run_dir が存在しません: {run_dir}")
        latest_run_dir = run_dir
        print(f"🧩 Using manually specified run dir: {latest_run_dir}")
    else:
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

    # === 各キーごとに ablation/{key}/value/ を作成して順番に実行 ===
    for key, values in ab_yaml.items():
        if not isinstance(values, list):
            continue

        key_name = "mode" if key == "train_mode" else key
        key_dir = ablation_root / key_name
        key_dir.mkdir(parents=True, exist_ok=True)

        for v in values:
            # ベース設定 + baseline最適値をマージ
            cfg = deepcopy(base_yaml)
            cfg.update(baseline_params)
            cfg[key] = v

            # === 共通デフォルトをここで強制しておく ===

            # device（必ずGPUを使う）
            cfg.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")

            # DataLoader最適化（なければ設定）
            if "num_workers" not in cfg:
                try:
                    cpu_count = multiprocessing.cpu_count()
                except Exception:
                    cpu_count = 4
                # CPUに合わせてほどほど（必要ならここ調整）
                cfg["num_workers"] = max(2, cpu_count // 2) if torch.cuda.is_available() else 0

            if "pin_memory" not in cfg:
                cfg["pin_memory"] = torch.cuda.is_available()

            # 検証頻度（train_core側で val_interval 対応しているなら有効）
            cfg.setdefault("val_interval", 1)

            # 出力先（各 ablation パターンごとに分離）
            ab_dir = key_dir / str(v)
            ab_dir.mkdir(parents=True, exist_ok=True)
            cfg["output_root"] = str(ab_dir)

            print(f"\n🚀 Running ablation: {key} = {v}")
            print(f"   device={cfg['device']}, num_workers={cfg['num_workers']}, pin_memory={cfg['pin_memory']}")

            # === GPUキャッシュを事前クリア ===
            cleanup_memory()

            try:
                val_loss, model_path = train_with_config(
                    cfg, full_df, le_act, le_sp, results_root=ab_dir
                )
                print(f"✅ Done: {key}={v}, val_loss={val_loss:.6f}")
                if model_path:
                    print(f"📦 Saved model: {model_path}")
            except Exception as e:
                print(f"❌ Error in ablation {key}={v}: {e}")
            finally:
                # === GPU/CPUメモリを完全解放 ===
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("🧹 Memory cleaned up\n")


# === Entry point ===

if __name__ == "__main__":
    datatype = "animalkingdom"
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df["action"])
    le_sp = LabelEncoder().fit(full_df["species"])

    run_ablation(
        "exec/config_search.yml",
        "exec/ablation.yml",
        full_df,
        le_act,
        le_sp,
        run_dir_manual="train_result/2025-11-11/run_001"
    )
