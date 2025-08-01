from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import os
os.environ["OMP_NUM_THREADS"] = "8"
import torch.nn as nn

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from umap import UMAP

from model import (
    DisentangleEmbedOnlySimple,
    DisentangleEmbedOnlyMLP,
)

# ======================================
# 1. CONFIG
# ======================================

CONFIG: Dict[str, object] = {
    "LOSS_FUNCTION": ["improved"],
    "SUFFIXES": ["mlp-grl", "mlp-nogrl", "linear-grl", "linear-nogrl"],
    "VISUALIZE": ["tsne", "umap"],
    "DATA_MODE": ["test", "all"],
    "USE_24FPS": False,
    "DATASET_NAME": "animalkingdom",
    "PLOT_CLUSTER": True,   # ← ここでクラスタ可視化のON/OFF
}

# ======================================
# 2. Utils
# ======================================

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def load_labels(cfg: Dict[str, object]) -> pd.DataFrame:
    dtype = cfg["DATASET_NAME"]
    base = Path("label") / dtype
    suffix = "_24fps" if cfg["USE_24FPS"] else ""

    df_train = pd.read_csv(base / "train" / "labels.csv")
    df_test = pd.read_csv(base / "test" / f"labels_test{suffix}.csv")

    df_train["video_path"] = df_train["video_path"].str.replace("\\", "/")
    df_test["video_path"] = df_test["video_path"].str.replace("\\", "/")

    df_train["source"] = "train"
    df_test["source"] = "test"

    if cfg["DATA_MODE"] == "train":
        df = df_train.copy()
    elif cfg["DATA_MODE"] == "test":
        df = df_test.copy()
    else:
        df = pd.concat([df_train, df_test], ignore_index=True)

    return df

def load_flow(cfg: Dict[str, object], suffix_24fps: str) -> Dict[str, np.ndarray]:
    dtype = cfg["DATASET_NAME"]

    flow_dir_train = Path("x3d_output") / dtype / "train"
    flow_dir_test  = Path("x3d_output") / dtype / "test"

    df_train = pd.read_csv(Path("label") / dtype / "train" / "labels_filtered.csv")
    df_test  = pd.read_csv(Path("label") / dtype / "test" / f"labels_test{suffix_24fps}.csv")

    df = pd.concat([df_train, df_test], ignore_index=True)

    flow_dict = {}

    for _, row in df.iterrows():
        video_path = row["video_path"].replace("\\", "/").strip()
        video_id = Path(video_path).stem

        npy_train = flow_dir_train / video_id / f"{video_id}.npy"
        npy_test  = flow_dir_test / video_id / f"{video_id}.npy"

        npy_path = npy_train if npy_train.exists() else npy_test

        if npy_path.exists():
            arr = np.load(str(npy_path))
            if arr.ndim > 1:
                arr = arr.squeeze(0)
            flow_dict[video_path] = arr

    print(f"✅ flow npy 読み込み数: {len(flow_dict)} / {len(df)}")

    return flow_dict

def build_encoder(use_mlp: bool, D: int, H=256) -> torch.nn.Module:
    encoder_cls = DisentangleEmbedOnlyMLP if use_mlp else DisentangleEmbedOnlySimple
    return encoder_cls(D=D, H=H).cuda().eval()

def extract_action_embeddings_flow_infer(
    flow_dict: Dict[str, np.ndarray],
    df: pd.DataFrame,
    encoder: torch.nn.Module,
) -> Tuple[torch.Tensor, List[int], List[str], List[str]]:
    emb_list, labels, paths, sources = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row["video_path"]
            if path not in flow_dict:
                continue
            flow = torch.tensor(flow_dict[path]).unsqueeze(0).float().cuda()
            flow = nn.functional.normalize(flow, dim=-1)
            a_vec, _ = encoder(flow)
            emb_list.append(a_vec.squeeze(0).cpu())
            labels.append(row["act_id"])
            paths.append(path)
            sources.append(row["source"])
    return torch.stack(emb_list), labels, paths, sources

# ======================================
# 3. Clustering + 可視化
# ======================================

def evaluate_clustering(X: torch.Tensor, labels: List[int], name: str, out_dir: Path, max_k: int = 10) -> Tuple[float, float, np.ndarray]:
    print(f"\n=== 📊 Evaluating {name} ===")
    X_np = X.numpy()
    true_k = len(set(labels))

    k_candidates = range(2, min(max_k, len(X_np)))
    sil_scores = [silhouette_score(X_np, KMeans(k, random_state=0).fit_predict(X_np)) for k in k_candidates]
    best_k = true_k if true_k in k_candidates else k_candidates[int(np.argmax(sil_scores))]

    kmeans_pred = KMeans(best_k, random_state=0).fit_predict(X_np)
    ari_k = adjusted_rand_score(labels, kmeans_pred)
    nmi_k = normalized_mutual_info_score(labels, kmeans_pred)

    print(f"KMeans (k={best_k}) ARI={ari_k:.4f}, NMI={nmi_k:.4f}")

    log_file = out_dir / "clustering_log.txt"
    with open(log_file, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] {name}\n")
        f.write(f"KMeans ARI={ari_k:.4f} NMI={nmi_k:.4f}\n\n")

    return ari_k, nmi_k, kmeans_pred

def visualize(
    X: torch.Tensor,
    labels: List[int],
    label_encoder: LabelEncoder,
    paths: List[str],
    sources: List[str],
    cfg: Dict[str, object],
    variant_name: str,
    ari: float,
    nmi: float,
    kmeans_pred: np.ndarray = None,
) -> None:
    vis_out_base = Path("results") / Path("result_flow") / cfg["DATASET_NAME"] / cfg["DATA_MODE"]

    for method in cfg["VISUALIZE"]:
        proj = (
            TSNE(n_components=2, random_state=0).fit_transform(X)
            if method == "tsne"
            else UMAP().fit_transform(X)
        )

        out_path = vis_out_base / method / f"{variant_name}.png"
        ensure_dirs(out_path.parent)

        fig, ax = plt.subplots(figsize=(8, 6))

        cmap = plt.get_cmap("tab20")
        labels_np = np.array(labels)
        sources_np = np.array(sources)

        for label in np.unique(labels):
            mask = labels_np == label
            color = cmap(label % 20)

            mask_train = mask & (sources_np == "train")
            mask_test = mask & (sources_np == "test")

            if mask_train.any():
                ax.scatter(
                    proj[mask_train, 0],
                    proj[mask_train, 1],
                    s=20,
                    c=[color],
                    marker="o",
                    alpha=0.5,
                    label=f"{label_encoder.classes_[label]} (train)",
                )
            if mask_test.any():
                ax.scatter(
                    proj[mask_test, 0],
                    proj[mask_test, 1],
                    s=40,
                    c=[color],
                    marker="^",
                    edgecolors="black",
                    linewidths=0.5,
                    label=f"{label_encoder.classes_[label]} (test)",
                )

        # === クラスタ枠線だけ ===
        if cfg.get("PLOT_CLUSTER", True) and kmeans_pred is not None:
            for cluster_id in np.unique(kmeans_pred):
                mask = kmeans_pred == cluster_id
                ax.scatter(
                    proj[mask, 0],
                    proj[mask, 1],
                    s=100,
                    facecolors='none',
                    edgecolors='black',
                    linewidths=1.0,
                    marker="o",
                    label=f"Cluster {cluster_id}"
                )

        ax.set_title(
            f"{method.upper()} | {variant_name}\n"
            f"ARI={ari:.4f} | NMI={nmi:.4f}"
        )
        ax.legend(markerscale=1.2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()

# ======================================
# 4. MAIN
# ======================================

def main(cfg: Dict[str, object]) -> None:
    for datamode in cfg["DATA_MODE"]:
        print(f"\n=== [MODE] {datamode} ===")
        cfg_one = cfg.copy()
        cfg_one["DATA_MODE"] = datamode

        df = load_labels(cfg_one)
        le_act = LabelEncoder().fit(df["action"])
        df["act_id"] = le_act.transform(df["action"])

        for loss in cfg["LOSS_FUNCTION"]:
            for suffix in cfg["SUFFIXES"]:
                suffix_24fps = "_24fps" if cfg["USE_24FPS"] else ""
                variant = f"flow_{loss}_{suffix}{suffix_24fps}"
                print(f"\n=== Processing {variant} ===")

                flow_dict = load_flow(cfg_one, suffix_24fps)
                flow_input_dim = next(iter(flow_dict.values())).shape[0]
                encoder = build_encoder(use_mlp="mlp" in suffix, D=flow_input_dim)

                model_path = Path("models") /Path("model_flow") / cfg["DATASET_NAME"] / loss / f"{suffix}.pth"
                checkpoint = torch.load(model_path, map_location="cuda")
                encoder.load_state_dict(checkpoint, strict=False)

                df_valid = df[df["video_path"].isin(flow_dict.keys())].copy()

                if df_valid.empty:
                    print("⚠️ 有効なサンプルがありません。スキップします。")
                    continue

                a_vecs, a_labels, paths, sources = extract_action_embeddings_flow_infer(
                    flow_dict, df_valid, encoder
                )

                out_dir = Path("results") /Path("result_flow") / cfg["DATASET_NAME"] / datamode
                if cfg["PLOT_CLUSTER"]:
                    out_dir = out_dir / "plot_cluster"
                ensure_dirs(out_dir)
                ari, nmi, kmeans_pred = evaluate_clustering(a_vecs, a_labels, variant, out_dir)
                visualize(
                    a_vecs, a_labels, le_act, paths, sources,
                    cfg_one, variant, ari, nmi, kmeans_pred,
                )


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(CONFIG)
