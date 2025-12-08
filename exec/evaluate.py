# eval.py
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap


# =================================
# モデルのimport（学習時と同じ）
# =================================
from model import (
    GatedFusion, ActionLinearNet, ActionMLPNet,
    SimpleLinearNet, SimpleMLPNet
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")


def setup_environment() -> None:
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"


def load_data_for_eval(data_dt: str, pooling: bool):
    print("📂 Loading labels and features... pooling =", pooling)

    # CSV 読み込み
    train_csv = f"./label/{data_dt}/train/labels.csv"
    test_csv  = f"./label/{data_dt}/test/labels_test.csv"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df["source"] = "train"
    test_df["source"] = "test"
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()

    # LabelEncoder
    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])

    # 特徴量ディクショナリ（フレームリスト）
    features = {
        "vmae": {},
        "flow": {},
        "flow_centered": {},
    }

    # root 自動判定
    def detect_vector_root(path_str: str) -> str:
        low = path_str.lower()
        if "polar" in low: return "polar"
        if "animalkingdom" in low: return "animalkingdom"
        return data_dt

    def load_vectors(base_dir: Path) -> Optional[List[np.ndarray]]:
        
        if pooling:
            avg_path = base_dir / "avg_pooling.npy"
            if avg_path.exists():
                arr = np.load(avg_path)
                arr = arr.squeeze(0) if arr.ndim > 1 else arr
                return [arr]    # ← 1件のリストで返す
            return None

        else:
            slide = base_dir / "sliding_list"
            if slide.exists():
                frames = sorted(slide.glob("*.npy"))
                if frames:
                    vecs = [np.load(p) for p in frames]
                    return vecs     # ← フレームのリストで返す
            return None

    # ---------------------------------------------------------
    # すべての動画に対して feature をロード（フレームレベル）
    # ---------------------------------------------------------
    for _, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Loading features"):
        p = row["video_path"]
        vid = Path(p).stem
        root = detect_vector_root(p)

        # VMAE
        v_dir = Path(f"./vector/{root}/{vid}")
        v_vecs = load_vectors(v_dir)
        if v_vecs is not None:
            features["vmae"][p] = v_vecs

        # X3D
        x_dir = Path(f"./x3d_vector/{root}/{vid}")
        x_vecs = load_vectors(x_dir)
        if x_vecs is not None:
            features["flow"][p] = x_vecs

        # X3D centered
        xc_dir = Path(f"./x3d_vector_centered/{root}/{vid}")
        xc_vecs = load_vectors(xc_dir)
        if xc_vecs is not None:
            features["flow_centered"][p] = xc_vecs

    return full_df, le_act, features


# =================================
# モデルロード
# =================================
def build_and_load_model(params: Dict):

    models = nn.ModuleDict()
    model_path = Path(params["model_path"])

    if not model_path.exists():
        print("⚠️ Model not found", model_path)
        return None

    state_dict = torch.load(model_path, map_location=DEVICE)

    fused_dim = int(params.get("fused_dim", 512))
    D = fused_dim if params["train_mode"] == "gated" else (
        2048 if params["train_mode"] == "flow" else 768
    )

    # Fusion
    if params["train_mode"] == "gated":
        fusion = GatedFusion(2048, 768, fused_dim).to(DEVICE).eval()
        fusion_state = {k.replace("fusion.", ""): v for k, v in state_dict.items() if k.startswith("fusion.")}
        fusion.load_state_dict(fusion_state, strict=False)
        models["fusion"] = fusion

    # Encoder
    prefix = next((p for p in ["action_encoder.", "net."] if any(k.startswith(p) for k in state_dict)), "")
    enc_state = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}

    use_mlp = any(".0." in k or "act_embed.0" in k for k in enc_state)
    encoder = (
        ActionMLPNet(D, 256, 256).to(DEVICE).eval()
        if use_mlp else ActionLinearNet(D, 256).to(DEVICE).eval()
    )
    encoder.load_state_dict(enc_state, strict=False)
    models["encoder"] = encoder

    return models


# =================================
# 埋め込み抽出（フレームレベル）
# =================================
def extract_embeddings(df, features, models, params):

    mode = params["train_mode"]
    flow_key = "flow_centered" if params.get("flow_preprocessing") == "centered" else "flow"

    encoder = models["encoder"]
    fusion = models.get("fusion", None)

    emb_list = []
    labels = []
    sources = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting ({mode})"):
        p = row["video_path"]

        # 取り出す feature のセット
        if mode == "gated":
            if p not in features[flow_key] or p not in features["vmae"]:
                continue
            x_list = features[flow_key][p]
            v_list = features["vmae"][p]
            # フレームレベル：短いほうに合わせる
            for x_vec, v_vec in zip(x_list, v_list):
                xx = torch.tensor(x_vec).unsqueeze(0).float().to(DEVICE)
                vv = torch.tensor(v_vec).unsqueeze(0).float().to(DEVICE)
                fused, _ = fusion(xx, vv)
                emb = encoder(fused)

                emb = nn.functional.normalize(emb, dim=-1)
                emb_list.append(emb.squeeze(0).cpu())
                labels.append(row["act_id"])
                sources.append(row["source"])

        else:
            key = "vmae" if mode == "mae" else flow_key
            if p not in features[key]:
                continue
            for vec in features[key][p]:
                t = torch.tensor(vec).unsqueeze(0).float().to(DEVICE)
                emb = encoder(t)
                emb = nn.functional.normalize(emb, dim=-1)

                emb_list.append(emb.squeeze(0).cpu())
                labels.append(row["act_id"])
                sources.append(row["source"])

    if not emb_list:
        return None, None, None

    return torch.stack(emb_list), np.array(labels), np.array(sources)


# =================================
# 可視化（train/test区別）
# =================================
def _build_colors(n):
    cmaps = [get_cmap("tab20"), get_cmap("tab20b"), get_cmap("tab20c")]
    pool = [cm(i) for cm in cmaps for i in range(cm.N)]
    return pool[:n]

def _plot_with_source(X, y, s, le_act, title, save_path):
    label_names = le_act.inverse_transform(y)
    uniq = np.unique(label_names)
    colors = _build_colors(len(uniq))
    color_map = {lab: colors[i] for i, lab in enumerate(uniq)}

    marker = {"train": "o", "test": "^"}

    plt.figure(figsize=(8, 6))
    for lab in uniq:
        for src in ["train", "test"]:
            mask = (label_names == lab) & (s == src)
            if mask.sum() == 0: continue
            plt.scatter(
                X[mask, 0], X[mask, 1],
                color=color_map[lab],
                marker=marker[src],
                s=12, alpha=0.8,
                label=f"{lab} ({src})"
            )
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220)
    plt.close()


# =================================
# 評価 + 可視化
# =================================
def evaluate_and_visualize(emb, lab, src, le_act, name, out_dir):
    tsne_dir = out_dir / "tsne"
    umap_dir = out_dir / "umap"
    tsne_dir.mkdir(parents=True, exist_ok=True)
    umap_dir.mkdir(parents=True, exist_ok=True)

    # === test のみで評価 ===
    mask = (src == "test")
    X_test = emb[mask].numpy()
    y_test = lab[mask]

    n_clusters = len(np.unique(y_test))

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    pred = clustering.fit_predict(X_test)
    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)

    # === t-SNE ===
    try:
        ts = TSNE(n_components=2, random_state=42, perplexity=30)
        X2 = ts.fit_transform(emb.numpy())
        # テストのみ
        # X2 = ts.fit_transform(emb[src == "test"].numpy())
        _plot_with_source(X2, lab, src, le_act, f"t-SNE - {name}", tsne_dir / f"{name}.png")
    except Exception as e:
        print("t-SNE failed:", e)

    # === UMAP ===
    try:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
        U = reducer.fit_transform(emb.numpy())
        _plot_with_source(U, lab, src, le_act, f"UMAP - {name}", umap_dir / f"{name}.png")
    except Exception as e:
        print("UMAP failed:", e)

    return ari, nmi


# =================================
# メイン
# =================================
def main(run_dir: Path):
    setup_environment()

    run_dir = Path(run_dir)
    ablation_root = run_dir / "ablation"

    baseline_path = run_dir / "baseline_config.json"
    if baseline_path.exists():
        base_cfg = json.load(open(baseline_path))
        params = base_cfg.get("params", {})
        params.update(base_cfg.get("user_attrs", {}))
    else:
        params = {
            "train_mode": "gated",
            "flow_preprocessing": "normal",
            "fused_dim": 512,
            "datatype": "animalkingdom",
            "pooling": True,   # ← default は pooling=False にしてある
        }

    DATATYPE = params["datatype"]
    POOLING = params.get("pooling", False)

    eval_root = run_dir / "eval"
    eval_root.mkdir(exist_ok=True)

    full_df, le_act, features = load_data_for_eval(DATATYPE, POOLING)

    results = []
    model_paths = list(ablation_root.glob("**/*.pth"))

    for mp in tqdm(model_paths, desc="Evaluating"):
        rel = mp.relative_to(ablation_root).parts
        key = rel[0] if len(rel) > 0 else "unknown"
        val = rel[1] if len(rel) > 1 else "unknown"

        p = params.copy()
        p["model_path"] = mp
        if key == "train_mode": p["train_mode"] = val
        if key == "flow_preprocessing": p["flow_preprocessing"] = val

        model = build_and_load_model(p)
        if model is None: continue

        emb, lab, src = extract_embeddings(full_df, features, model, p)
        ari, nmi = evaluate_and_visualize(emb, lab, src, le_act, mp.stem, eval_root)

        results.append({
            "name": mp.stem,
            "train_mode": p.get("train_mode"),
            "flow_preprocessing": p.get("flow_preprocessing"),
            "pooling": POOLING,
            "ari": ari,
            "nmi": nmi,
        })

    pd.DataFrame(results).to_csv(eval_root / "eval_summary.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        dirs = sorted(Path("train_result").glob("**/run_*"), reverse=True)
        main(dirs[0])
