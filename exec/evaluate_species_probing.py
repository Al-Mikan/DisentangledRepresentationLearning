# evaluate_species_probing.py
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from model import GatedFusion, ActionMLPNet, SpeciesDiscriminator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")


def setup_environment() -> None:
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"


# =================================
# データロード (共通)
# =================================
def load_data_for_eval(
    train_label_paths: Union[str, List[str]], 
    target_test_csv: str,  
    pooling: bool, 
    default_datatype: str = "animalkingdom"
):
    print(f"📂 Loading labels... Test target: {target_test_csv}")

    # train_csv 群の読み込み（train ソース）
    if isinstance(train_label_paths, str):
        train_label_paths = [train_label_paths]

    train_dfs = []
    for p in train_label_paths:
        df_i = pd.read_csv(p)
        df_i["source"] = "train"
        train_dfs.append(df_i)
    
    if not train_dfs:
        raise RuntimeError("[eval] train_label_paths から有効な train CSV が読み込めませんでした。")
    
    train_df = pd.concat(train_dfs, ignore_index=True)

    # test_csv の読み込み（test ソース）
    test_df = pd.read_csv(target_test_csv)
    test_df["source"] = "test"
    
    # train+test を連結
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()

    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])
    
    # Species Label Encoder (重要: Nan埋めはしない前提だが念のため)
    # full_df["species"] = full_df["species"].fillna("unknown")
    le_sp = LabelEncoder().fit(full_df["species"].astype(str))
    full_df["sp_id"] = le_sp.transform(full_df["species"].astype(str))

    features = {
        "vmae": {},
        "flow": {},
        "flow_centered": {},
    }

    # root 自動判定
    def detect_vector_root(path_str: str) -> str:
        low = path_str.lower()
        if "polar" in low: return "polar"
        if "elephant" in low: return "elephant"
        if "animalkingdom" in low: return "animalkingdom"
        return default_datatype

    # feature ロード
    def load_vectors(base_dir: Path) -> Optional[List[np.ndarray]]:
        if pooling:
            avg_path = base_dir / "avg_pooling.npy"
            if avg_path.exists():
                arr = np.load(avg_path)
                arr = arr.squeeze(0) if arr.ndim > 1 else arr
                return [arr]
            return None
        else:
            slide = base_dir / "sliding_list"
            if slide.exists():
                files = sorted(slide.glob("*.npy"))
                if files:
                    return [np.load(p) for p in files]
            return None

    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Loading features"):
        p = row["video_path"]
        vid = Path(p).stem
        root = detect_vector_root(p)

        # VMAE
        v_dir = Path(f"./vector/{root}/{vid}")
        v_vecs = load_vectors(v_dir)
        if v_vecs is not None:
            features["vmae"][p] = v_vecs

        # X3D normal
        x_dir = Path(f"./x3d_vector/{root}/{vid}")
        x_vecs = load_vectors(x_dir)
        if x_vecs is not None:
            features["flow"][p] = x_vecs

        # X3D centered
        xc_dir = Path(f"./x3d_vector_centered/{root}/{vid}")
        xc_vecs = load_vectors(xc_dir)
        if xc_vecs is not None:
            features["flow_centered"][p] = xc_vecs

    return full_df, le_act, le_sp, features


# =================================
# モデルロード
# =================================
def build_and_load_model(params: Dict, le_sp: LabelEncoder):

    models = nn.ModuleDict()
    model_path = Path(params["model_path"])

    if not model_path.exists():
        print("⚠️ Model not found", model_path)
        return None

    state_dict = torch.load(model_path, map_location=DEVICE)

    train_mode = params.get("train_mode")
    if train_mode == "gated":
        D = int(params.get("fused_dim", 512))   
    elif train_mode == "flow":
        D = 2048
    elif train_mode == "mae":
        D = 768
    else:
        raise ValueError("Unknown train_mode")
    
    feature_dim = int(params.get("feature_dim", 256))

    # 1. Fusion モデル
    if params["train_mode"] == "gated":
        fusion = GatedFusion(2048, 768, int(params.get("fused_dim", 512))).to(DEVICE).eval()
        fusion_state = {k.replace("fusion.", ""): v
                        for k, v in state_dict.items()
                        if k.startswith("fusion.")}
        fusion.load_state_dict(fusion_state, strict=False)
        models["fusion"] = fusion

    # 2. Encoder
    prefix = next((p for p in ["action_encoder.", "net."]
                   if any(k.startswith(p) for k in state_dict)), "")
    enc_state = {k.replace(prefix, ""): v
                 for k, v in state_dict.items() if k.startswith(prefix)}

    encoder = ActionMLPNet(input_dim=D, feature_dim=feature_dim, hidden_dim=512).to(DEVICE).eval()
    encoder.load_state_dict(enc_state, strict=False)
    models["encoder"] = encoder
    
    # 3. Discriminator (Adv ONの場合のみロード)
    if params.get("adversarial", "off") != "off":
        disc_state = {k.replace("discriminator.", ""): v
                      for k, v in state_dict.items()
                      if k.startswith("discriminator.")}
                      
        # Discriminatorの重みがあるか確認
        if disc_state:
            # Checkpointからクラス数(出力次元)を推定
            # SpeciesDiscriminator の最後は "classifier.4.weight"
            weight_key = "classifier.4.weight"
            bias_key = "classifier.4.bias"
            
            if weight_key in disc_state:
                ckpt_num_species = disc_state[weight_key].shape[0]
                # print(f"ℹ️ Loaded discriminator with {ckpt_num_species} species classes (Current data: {len(le_sp.classes_)})")
                
                # Checkpointに合わせてモデル構築
                disc = SpeciesDiscriminator(feature_dim, ckpt_num_species).to(DEVICE).eval()
                disc.load_state_dict(disc_state, strict=False)
                models["discriminator"] = disc
            else:
                print(f"⚠️ Warning: Discriminator weights found but '{weight_key}' is missing. Skipping.")
        else:
            print("⚠️ Warning: adversarial is ON but no discriminator weights found.")

    return models


# =================================
# 埋め込み抽出
# =================================
def extract_embeddings(df, features, models, params):

    mode = params["train_mode"]
    flow_key = "flow_centered" if params.get("flow_preprocessing") == "centered" else "flow"

    encoder = models["encoder"]
    fusion = models["fusion"] if "fusion" in models else None

    emb_list = []
    sp_ids = []
    sources = []
    meta_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting ({mode})"):
        p = row["video_path"]

        if mode == "gated":
            if p not in features[flow_key] or p not in features["vmae"]:
                continue
            x_list = features[flow_key][p]
            v_list = features["vmae"][p]

            n_windows = min(len(x_list), len(v_list))
            for x_vec, v_vec in zip(x_list[:n_windows], v_list[:n_windows]):
                xx = torch.tensor(x_vec).unsqueeze(0).float().to(DEVICE)
                vv = torch.tensor(v_vec).unsqueeze(0).float().to(DEVICE)
                fused, _ = fusion(xx, vv)
                emb = encoder(fused)
                emb = nn.functional.normalize(emb, dim=-1) # Normalize

                emb_list.append(emb.squeeze(0).cpu())
                sp_ids.append(row["sp_id"])
                sources.append(row["source"])
                meta_rows.append(row)

        else:
            key = "vmae" if mode == "mae" else flow_key
            if p not in features[key]:
                continue
            for vec in features[key][p]:
                t = torch.tensor(vec).unsqueeze(0).float().to(DEVICE)
                emb = encoder(t)
                emb = nn.functional.normalize(emb, dim=-1) # Normalize

                emb_list.append(emb.squeeze(0).cpu())
                sp_ids.append(row["sp_id"])
                sources.append(row["source"])
                meta_rows.append(row)

    if not emb_list:
        return None, None, None, None

    return torch.stack(emb_list), np.array(sp_ids), np.array(sources), pd.DataFrame(meta_rows)


# =========================================================
# Case 1: Pretrained Discriminator Evaluation (Adv ON)
# =========================================================
def eval_pretrained_discriminator(models, emb, sp_ids, src):
    """
    Adv有効時に、学習済みDiscriminatorを使ってTestデータを評価する。
    """
    if "discriminator" not in models:
        print("⚠️ Pretrained discriminator not found in models.")
        return None
        
    disc = models["discriminator"]
    disc.eval()
    
    # Testデータのみ抽出
    mask_test = (src == "test")
    if not np.any(mask_test):
        return None
        
    X_test = emb[mask_test].to(DEVICE)
    y_test = torch.from_numpy(sp_ids[mask_test]).long().to(DEVICE)
    
    with torch.no_grad():
        logits = disc(X_test)
        pred = logits.argmax(dim=1)
        acc = (pred == y_test).float().mean().item()
        
    return acc


# =========================================================
# Case 2: Fresh Probing (Adv OFF / Reference)
# =========================================================
def train_fresh_discriminator(emb, sp_ids, src, num_species):
    """
    Adv無効時（または比較用）に、新しいDiscriminatorをTrainで学習し、Testで評価する。
    Validation Split + Early Stopping あり。
    """
    mask_train = (src == "train")
    mask_test  = (src == "test")
    
    if not np.any(mask_train) or not np.any(mask_test):
        print("⚠️ Train/Test data missing for probing.")
        return None
        
    X_train = emb[mask_train].numpy()
    y_train = sp_ids[mask_train]
    X_test  = emb[mask_test].numpy()
    y_test  = sp_ids[mask_test]
    
    # input_dim
    input_dim = X_train.shape[1]
    
    # Train / Valid Split (8:2)
    # クラス数が少なすぎてsplitできない場合の対策
    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
    except ValueError:
        # サンプル数が少なすぎて層化抽出できない場合は単純ランダム
        print("⚠️ Stratified split failed, using random split.")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    # Tensor化
    t_X_tr  = torch.from_numpy(X_tr).float()
    t_y_tr  = torch.from_numpy(y_tr).long()
    
    t_X_val = torch.from_numpy(X_val).float().to(DEVICE)
    t_y_val = torch.from_numpy(y_val).long().to(DEVICE)
    
    t_X_test = torch.from_numpy(X_test).float().to(DEVICE)
    t_y_test = torch.from_numpy(y_test).long().to(DEVICE)

    ds_train = TensorDataset(t_X_tr, t_y_tr)
    # バッチサイズ調整: データ数が少ない場合は小さくする
    batch_size = min(128, len(X_tr))
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    # モデル構築 (新規初期化)
    net = SpeciesDiscriminator(input_dim, num_species).to(DEVICE)
    
    # 学習用設定
    torch.set_grad_enabled(True) # ここだけGrad有効化
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    max_epochs = 100
    patience = 10
    best_val_loss = float("inf")
    no_improve = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        net.train()
        for bx, by in dl_train:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = net(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
        # Validation
        net.eval()
        with torch.no_grad():
            logits_val = net(t_X_val)
            val_loss = criterion(logits_val, t_y_val).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_model_state = copy.deepcopy(net.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    torch.set_grad_enabled(False) # 戻す
    
    # Test評価
    if best_model_state:
        net.load_state_dict(best_model_state)
    
    net.eval()
    with torch.no_grad():
        logits_test = net(t_X_test)
        pred_test = logits_test.argmax(dim=1)
        acc = (pred_test == t_y_test).float().mean().item()
        
    return acc


# =================================
# メイン
# =================================
def main(run_dir: Path, pooling_mode: str = "both"):
    setup_environment()

    run_dir = Path(run_dir)
    ablation_root = run_dir / "ablation"

    # Config読み込み
    baseline_path = run_dir / "baseline_config.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        params = {}
        if "config_used" in base_cfg and isinstance(base_cfg["config_used"], dict):
            params.update(base_cfg["config_used"])
        params.update(base_cfg.get("params", {}))
        params.update(base_cfg.get("user_attrs", {}))
    else:
        params = {}

    DATATYPE = params.get("datatype", "animalkingdom")
    
    if pooling_mode == "true":
        POOLING_MODES = [True]
    elif pooling_mode == "false":
        POOLING_MODES = [False]
    else:
        POOLING_MODES = [True, False]

    train_label_paths = params.get("train_label_paths", None)
    test_label_paths = params.get("test_label_paths", None)

    if train_label_paths is None or test_label_paths is None:
        print("⚠️ Label paths not found in config.")
        return
    
    if isinstance(test_label_paths, str):
        test_label_paths = [test_label_paths]

    model_paths = list(ablation_root.glob("**/*.pth"))
    if not model_paths:
        print("⚠️ No model files found.")
        return

    for POOLING in POOLING_MODES:
        pooling_str = "pooling_true" if POOLING else "pooling_false"
        print(f"\n >>> Processing {pooling_str} ...")

        for test_csv in test_label_paths:
            test_stem = Path(test_csv).stem
            eval_root = run_dir / "eval" / test_stem / pooling_str
            eval_root.mkdir(parents=True, exist_ok=True)
            
            full_df, le_act, le_sp, features = load_data_for_eval(train_label_paths, test_csv, POOLING, DATATYPE)
            results = []

            for mp in tqdm(model_paths, desc=f"Models ({test_stem})"):
                p = params.copy()
                p["model_path"] = mp

                # ディレクトリ名からパラメータ推定
                rel = mp.relative_to(ablation_root).parts
                key = rel[0]
                val = rel[1]
                if key in {"train_mode", "adversarial", "flow_preprocessing"}:
                    p[key] = val

                if p.get("train_mode") is None: 
                    continue

                # モデルロード
                models = build_and_load_model(p, le_sp)
                if models is None:
                    continue

                # 埋め込み抽出
                emb, sp_ids, src, _ = extract_embeddings(full_df, features, models, p)
                if emb is None:
                    continue
                
                # ============================================
                # 分岐: Adv ON vs OFF
                # ============================================
                adv_mode = p.get("adversarial", "off")
                acc = None
                method = ""

                if adv_mode != "off":
                    # Case 1: Use Pretrained Discriminator
                    acc = eval_pretrained_discriminator(models, emb, sp_ids, src)
                    method = "pretrained_disc"
                else:
                    # Case 2: Fresh Probing (Train -> Test)
                    num_species = len(le_sp.classes_)
                    acc = train_fresh_discriminator(emb, sp_ids, src, num_species)
                    method = "fresh_probing"

                res_dict = {
                    "name": mp.stem,
                    "train_mode": p.get("train_mode"),
                    "adversarial": adv_mode,
                    "pooling": POOLING,
                    "species_acc": acc if acc is not None else 0.0,
                    "eval_method": method
                }
                results.append(res_dict)

            if results:
                out_path = eval_root / "species_eval_summary.csv"
                pd.DataFrame(results).to_csv(out_path, index=False)
                print(f"✅ Saved results to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=str, help="Path to run directory")
    parser.add_argument("--pooling", type=str, choices=["true", "false", "both"], default="both")
    args = parser.parse_args()

    if args.run_dir:
        main(Path(args.run_dir), pooling_mode=args.pooling)
    else:
        dirs = [d for d in Path("train_result").glob("**/run_*") if d.is_dir()]
        dirs = sorted(dirs, reverse=True)
        if len(dirs) == 0:
            raise RuntimeError("No run_* directory found.")
        main(dirs[0], pooling_mode=args.pooling)
