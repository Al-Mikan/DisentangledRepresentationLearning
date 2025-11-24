import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# =============================
# モデルのimport
# =============================
from model import (
    GatedFusion, ActionLinearNet, ActionMLPNet,
    SimpleLinearNet, SimpleMLPNet
)

# =============================
# 共通設定
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")


def setup_environment():
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"


# =============================
# データ読み込み
# =============================
def load_data_for_eval(config: Dict) -> Tuple[pd.DataFrame, LabelEncoder, Dict]:
    print("📂 Loading labels and features...")
    train_df = pd.read_csv(config['train_csv'])
    test_df = pd.read_csv(config['test_csv'])
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()

    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])

    features = {"flow": {}, "flow_centered": {}, "vmae": {}}

    # VMAE features
    if config['vmae_json'].exists():
        features["vmae"].update(json.loads(config['vmae_json'].read_text()))
    if config.get('vmae_json_test') and Path(config['vmae_json_test']).exists():
        features["vmae"].update(json.loads(Path(config['vmae_json_test']).read_text()))

    # Load X3D / flow features
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Loading .npy features"):
        path_str = row["video_path"]
        vid = Path(path_str).stem
        source_dir = "train" if row["source"] == "train" else "test"

        for key in ["flow", "flow_centered"]:
            base_dir = config['x3d_dir_centered'] if key == 'flow_centered' else config['x3d_dir']
            npy_path = base_dir / source_dir / vid / f"{vid}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                features[key][path_str] = arr.squeeze(0) if arr.ndim > 1 else arr

    return full_df, le_act, features


# =============================
# モデル構築とロード
# =============================
def build_and_load_model(params: Dict):
    models = nn.ModuleDict()
    model_path = Path(params['model_path'])
    if not model_path.exists():
        print(f"⚠️ Model not found: {model_path}")
        return None

    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

    fused_dim = params.get('fused_dim', 512)
    D = fused_dim if params['train_mode'] == 'gated' else (2048 if params['train_mode'] == 'flow' else 768)

    # Fusion
    if params['train_mode'] == 'gated':
        models['fusion'] = GatedFusion(2048, 768, fused_dim).to(DEVICE).eval()
        fusion_state = {k.replace('fusion.', ''): v for k, v in state_dict.items() if k.startswith('fusion.')}
        if fusion_state:
            models['fusion'].load_state_dict(fusion_state)

    # Encoder
    possible_prefixes = ['action_encoder.', 'net.']
    chosen_prefix = next((p for p in possible_prefixes if any(k.startswith(p) for k in state_dict.keys())), '')
    enc_state = {k.replace(chosen_prefix, ''): v for k, v in state_dict.items() if k.startswith(chosen_prefix)}

    if not enc_state:
        print(f"⚠️ Encoder weights not found in {model_path}")
        return None

    keys = list(enc_state.keys())
    has_mlp = any('.0.' in k or 'act_embed.0' in k for k in keys)

    if has_mlp:
        encoder = ActionMLPNet(D, 256, 256).to(DEVICE).eval()
    else:
        encoder = ActionLinearNet(D, 256).to(DEVICE).eval()

    try:
        encoder.load_state_dict(enc_state, strict=False)
    except Exception as e:
        print(f"⚠️ Loaded encoder non-strict due to {e}")

    models['encoder'] = encoder
    return models


# =============================
# 埋め込み抽出
# =============================
def extract_embeddings(df, features, models, params):
    emb_list, labels, sources = [], [], []
    encoder = models['encoder']
    fusion = models['fusion'] if 'fusion' in models else None
    mode = params['train_mode']

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting ({mode})"):
        path = row["video_path"]

        if mode == 'gated':
            flow_key = 'flow_centered' if params.get('flow_preprocessing') == 'centered' else 'flow'
            if path not in features[flow_key] or path not in features['vmae']:
                continue
            x3d = torch.tensor(features[flow_key][path]).unsqueeze(0).float().to(DEVICE)
            vmae = torch.tensor(features['vmae'][path]).unsqueeze(0).float().to(DEVICE)
            x, _ = fusion(x3d, vmae)
        else:
            fkey = 'vmae' if mode == 'mae' else ('flow_centered' if params.get('flow_preprocessing') == 'centered' else 'flow')
            if path not in features[fkey]:
                continue
            x = torch.tensor(features[fkey][path]).unsqueeze(0).float().to(DEVICE)

        a_vec = encoder(x)
        emb_list.append(nn.functional.normalize(a_vec, dim=-1).squeeze(0).cpu())
        labels.append(row["act_id"])
        sources.append(row["source"])

    if not emb_list:
        return None, None, None

    return torch.stack(emb_list), np.array(labels), np.array(sources)


# =============================
# 評価 + 可視化
# =============================
def evaluate_and_visualize(embeddings, labels, sources, le_act, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    if embeddings is None:
        print(f"⚠️ No embeddings extracted for {name}")
        return np.nan, np.nan

    test_mask = (sources == 'test')
    X_test, y_test = embeddings[test_mask], labels[test_mask]

    if len(y_test) == 0:
        return np.nan, np.nan

    n_clusters = len(np.unique(y_test))
    X_np = X_test.numpy()

    try:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric='cosine', linkage='average'
        )
        pred = clustering.fit_predict(X_np)
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, affinity='cosine', linkage='average'
        )
        pred = clustering.fit_predict(X_np)

    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)

    return ari, nmi


# =============================
# メイン処理
# =============================
def main(run_dir):
    setup_environment()

    run_dir = Path(run_dir)
    ablation_root = run_dir / "ablation"
    if not ablation_root.exists():
        raise FileNotFoundError(f"❌ Ablation directory not found: {ablation_root}")

    # Load baseline config
    baseline_path = run_dir / "baseline_config.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_cfg = json.load(f)
        base_params = baseline_cfg.get("params", {})
        base_params.update(baseline_cfg.get("user_attrs", {}))
        print(f"✅ Loaded baseline parameters from {baseline_path}")
    else:
        print("⚠ baseline_config.json not found, using default params.")
        base_params = {
            "train_mode": "gated",
            "adversarial": "off",
            "flow_preprocessing": "normal",
            "loss_type": "triplet",
            "fused_dim": 512,
        }

    eval_root = Path(f"./eval_result/{run_dir.parent.name}/{run_dir.name}")
    img_dir = eval_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    DATATYPE = "animalkingdom"
    config = {
        "train_csv": Path(f"./label/{DATATYPE}/train/labels.csv"),
        "test_csv": Path(f"./label/{DATATYPE}/test/labels_test.csv"),
        "vmae_json": Path(f"./vector/{DATATYPE}/train/vectors_sliding_base.json"),
        "vmae_json_test": Path(f"./vector/{DATATYPE}/test/vectors_sliding_base.json"),
        "x3d_dir": Path(f"./x3d_output/{DATATYPE}"),
        "x3d_dir_centered": Path(f"./x3d_output_centered/{DATATYPE}"),
    }

    full_df, le_act, features = load_data_for_eval(config)

    model_paths = list(ablation_root.glob("**/*.pth"))
    if not model_paths:
        print("⚠️ No model files found!")
        return

    results = []

    for model_path in tqdm(model_paths, desc="Evaluating models"):
        rel_parts = model_path.relative_to(ablation_root).parts
        ab_key = rel_parts[0] if len(rel_parts) > 0 else "unknown"
        ab_value = rel_parts[1] if len(rel_parts) > 1 else "unknown"

        params = base_params.copy()
        params["model_path"] = model_path

        # 変更対象パラメータを上書き
        if ab_key == "train_mode":
            params["train_mode"] = ab_value
        elif ab_key == "adversarial":
            params["adversarial"] = ab_value
        elif ab_key == "flow_preprocessing":
            params["flow_preprocessing"] = ab_value

        print(f"\n🧩 Evaluating {ab_key} = {ab_value}")

        models = build_and_load_model(params)
        if models is None:
            continue

        embeddings, labels, sources = extract_embeddings(full_df, features, models, params)
        ari, nmi = evaluate_and_visualize(embeddings, labels, sources, le_act, model_path.stem, img_dir)

        results.append({
            "ablation_key": ab_key,
            "ablation_value": ab_value,
            "name": model_path.stem,
            "train_mode": params.get("train_mode"),
            "adversarial": params.get("adversarial"),
            "flow_preprocessing": params.get("flow_preprocessing"),
            "test_ari": ari,
            "test_nmi": nmi,
            "model_path": str(model_path)
        })

    # ============================================
    # Markdown 出力のみ
    # ============================================
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="test_nmi", ascending=False)

    eval_root.mkdir(parents=True, exist_ok=True)
    md_path = eval_root / "eval_summary.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary (Markdown)\n\n")
        f.write(f"Generated from `{run_dir}`\n\n")
        f.write(df_sorted.to_markdown(index=False))
        f.write("\n")

    print(f"📄 Markdown summary saved to {md_path}")


if __name__ == "__main__":
    main("train_result/2025-11-11/run_001")
