import torch
import yaml
import json
import gc
from copy import deepcopy
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from train_core import (
    cleanup_memory,
    build_datasets_and_loaders,
    train_model,
    build_basename_from_config,
    DummyTrial,
)


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
    # === 設定ファイル読み込み ===
    base_yaml = load_yaml(cfg_path)
    ab_yaml = load_yaml(abl_path)
    output_root = Path(base_yaml.get("output_root", "./train_result"))

    # === 実行対象 run_dir の決定 ===
    if run_dir_manual is not None:
        run_dir = Path(run_dir_manual)
        if not run_dir.exists():
            raise FileNotFoundError(f"❌ 指定された run_dir が存在しません: {run_dir}")
        latest_run_dir = run_dir
    else:
        run_dirs = sorted(output_root.glob("run_*"))
        if not run_dirs:
            raise FileNotFoundError(f"❌ No run_xxx directory found under {output_root}")
        latest_run_dir = run_dirs[-1]

    # === baseline_config.json 読み込み ===
    baseline_params = load_baseline_json(latest_run_dir)
    print(f"✅ Loaded baseline parameters from {latest_run_dir}/baseline_config.json")

    # --- 共通データ分割 ---
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df["action"]
    )

    # --- 共通DataLoaderを1回だけ構築 ---
    base_cfg = deepcopy(base_yaml)
    base_cfg.update(baseline_params)
    train_loader, val_loader, _ = build_datasets_and_loaders(
        base_cfg, train_df, val_df, le_act, le_sp
    )

    # === ablation/ フォルダ作成 ===
    ablation_root = latest_run_dir / "ablation_fast"
    ablation_root.mkdir(parents=True, exist_ok=True)

    # === Ablation loop ===
    for key, values in ab_yaml.items():
        if not isinstance(values, list):
            continue

        key_dir = ablation_root / key
        key_dir.mkdir(parents=True, exist_ok=True)

        for v in values:
            # --- baseline設定を複製して差分適用 ---
            cfg = deepcopy(base_cfg)
            cfg[key] = v
            ab_dir = key_dir / str(v)
            ab_dir.mkdir(parents=True, exist_ok=True)
            cfg["output_root"] = str(ab_dir)

            # --- gatedモードのみfusionを初期化 ---
            fusion_model = None
            if cfg.get("train_mode") == "gated":
                from model import GatedFusion
                fusion_model = GatedFusion(2048, 768, int(cfg["fused_dim"])).to(DEVICE)

            dummy_trial = DummyTrial()
            run_name = build_basename_from_config(cfg)

            print(f"\n🚀 Running Ablation: {key} = {v}")

            # === 学習（DataLoader再利用） ===
            best_val = train_model(
                cfg,
                train_loader,
                val_loader,
                le_sp,
                dummy_trial,
                study_name="ablation",
                fusion=fusion_model,
                results_root=ab_dir,
                run_name_override=run_name,
                is_ablation=True,
                ablation_subdir=key,
            )

            print(f"✅ Done: {key}={v}, val_loss={best_val:.5f}")

            # --- メモリ解放 ---
            del fusion_model
            cleanup_memory()

    print("\n🎯 All ablations completed successfully!\n")


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
        run_dir_manual="train_result/2025-11-11/run_001",
    )
