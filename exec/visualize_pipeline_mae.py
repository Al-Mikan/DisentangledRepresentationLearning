from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple
import os
os.environ["OMP_NUM_THREADS"] = "8"
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
    davies_bouldin_score,
    pairwise_distances,
)
from umap import UMAP

from model import DisentangleEmbedOnlySimple, DisentangleEmbedOnlyMLP


# ====== 1. 設定 ======
CONFIG: Dict[str, object] = {
    "VECTOR_MODES": ["sliding"],
    "LOSS_FUNCTION": ["improved"],
    "SUFFIXES": ["linear-grl", "mlp-grl", "linear-nogrl", "mlp-nogrl"],
    "VISUALIZE": ["tsne", "umap"],
    "VMAE_VERSION": "base",
    "DATA_MODE": "test",  # "train" | "test" | "all"
    "USE_24FPS": True,
    "INTERACTIVE": False, 
    "DATASET_NAME": "animalkingdom",  # "animalkingdom" | "wolf" など
}
# ====== 2. ユーティリティ ======

def ensure_dirs(*paths: Path) -> None:
    """渡されたディレクトリをすべて作成 (存在チェック付き)"""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_labels(cfg: Dict[str, object]) -> pd.DataFrame:
    """train / test ラベルを読み込み、source 列を付加した DataFrame を返す"""
    dtype = cfg["DATASET_NAME"]
    base = Path("label") / dtype
    if cfg["USE_24FPS"]:
        suffix = "_24fps"
    else:
        suffix = ""

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
    else:  # "all"
        df = pd.concat([df_train, df_test], ignore_index=True)

    return df


def load_vectors(cfg: Dict[str, object], mode: str, suffix_24fps: str) -> Dict[str, List[float]]:
    """指定モードに対応する train / test / (ucf) のベクトル JSON をまとめてロード"""
    dtype = cfg["DATASET_NAME"]
    vmae = cfg["VMAE_VERSION"]
    vecs: Dict[str, List[float]] = {}

    # --- ベクトルファイルパスを組み立て ---
    base_vec = Path("vector") / dtype
    train_vec = base_vec / "train" / f"vectors_{mode}_{vmae}.json"
    test_vec = base_vec / "test" / f"vectors_{mode}{suffix_24fps}_{vmae}.json"

    for path in [train_vec, test_vec]:
        if path.exists():
            vecs.update(json.loads(path.read_text()))

    return vecs


def build_encoder(use_mlp: bool) -> torch.nn.Module:
    """ヘッド (Simple or MLP) を CUDA 上にロードして返す"""
    encoder_cls = DisentangleEmbedOnlyMLP if use_mlp else DisentangleEmbedOnlySimple
    return encoder_cls().cuda().eval()


def extract_action_embeddings(
    vecs: Dict[str, List[float]],
    df: pd.DataFrame,
    encoder: torch.nn.Module,
) -> Tuple[torch.Tensor, List[int], List[str], List[str]]:
    """対象 DataFrame の video_path が vecs に存在するものだけ抽出"""
    emb_list, labels, paths, sources = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row["video_path"]
            if path not in vecs:
                continue
            z = torch.tensor(vecs[path]).unsqueeze(0).float().cuda()
            a_vec, _ = encoder(z)
            emb_list.append(a_vec.squeeze(0).cpu())
            labels.append(row["act_id"])
            paths.append(path)
            sources.append(row["source"])
    return torch.stack(emb_list), labels, paths, sources


# ====== 3. 評価 & 可視化 ======

def evaluate_clustering(
    X: torch.Tensor,
    labels: List[int],
    name: str,
    out_dir: Path,
    max_k: int = 10,
) -> Tuple[float, float]:
    """KMeans (k 探索付き), Agglomerative, DBSCAN でスコア計算"""
    print(f"\n=== 📊 Evaluating {name} ===")
    X_np = X.numpy()
    true_k = len(set(labels))

    k_candidates = range(2, min(max_k, len(X_np)))
    sil_scores = [
        silhouette_score(X_np, KMeans(k, random_state=0).fit_predict(X_np)) for k in k_candidates
    ]
    best_k = true_k if true_k in k_candidates else k_candidates[int(np.argmax(sil_scores))]

    kmeans_pred = KMeans(best_k, random_state=0).fit_predict(X_np)
    ari_k = adjusted_rand_score(labels, kmeans_pred)
    nmi_k = normalized_mutual_info_score(labels, kmeans_pred)

    agg_pred = AgglomerativeClustering(best_k).fit_predict(X_np)
    ari_a = adjusted_rand_score(labels, agg_pred)
    nmi_a = normalized_mutual_info_score(labels, agg_pred)

    dbscan_result = ""
    db_pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_np)
    if (db_pred != -1).sum() > 2 and len(set(db_pred) - {-1}) >= 2:
        ari_d = adjusted_rand_score(np.array(labels)[db_pred != -1], db_pred[db_pred != -1])
        nmi_d = normalized_mutual_info_score(np.array(labels)[db_pred != -1], db_pred[db_pred != -1])
        dbscan_result = f"DBSCAN ARI={ari_d:.4f}, NMI={nmi_d:.4f}"
    else:
        dbscan_result = "DBSCAN Insufficient clusters"

    # === 出力 ===
    print(f"KMeans (k={best_k}) ARI={ari_k:.4f}, NMI={nmi_k:.4f}")
    print(f"Agglomerative ARI={ari_a:.4f}, NMI={nmi_a:.4f}")
    print(dbscan_result)

    # === ログを日時付きで保存 ===
    log_file = out_dir / "clustering_log.txt"
    with open(log_file, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] {name}\n")
        f.write(f" KMeans (k={best_k}) ARI={ari_k:.4f}, NMI={nmi_k:.4f}\n")
        f.write(f" Agglomerative ARI={ari_a:.4f}, NMI={nmi_a:.4f}\n")
        f.write(f" {dbscan_result}\n\n")

    save_dist_hist(X_np, labels, out_dir / f"{name}_distance.png")
    return ari_k, nmi_k

def save_dist_hist(X: np.ndarray, labels: List[int], out_file: Path) -> None:
    """ラベル同一 / 異なるペア距離のヒストグラムを描画"""
    dists = pairwise_distances(X)
    same, diff = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            (same if labels[i] == labels[j] else diff).append(dists[i, j])

    # plt.figure(figsize=(8, 5))
    # plt.hist(same, bins=50, alpha=0.5, label="Same Class")
    # plt.hist(diff, bins=50, alpha=0.5, label="Different Class")
    # plt.title("Pairwise Distance Distribution")
    # plt.xlabel("Distance")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(out_file)
    # plt.close()


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
) -> None:
    """tsne / umap で散布図を描画"""
    vis_out_base = Path("result") / cfg["DATASET_NAME"] / cfg["VMAE_VERSION"] / cfg["DATA_MODE"]
    for method in cfg["VISUALIZE"]:
        proj = (
            TSNE(n_components=2, random_state=0).fit_transform(X)
            if method == "tsne"
            else UMAP().fit_transform(X)
        )
        out_path = vis_out_base / method / f"{variant_name}.png"
        ensure_dirs(out_path.parent)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 6))

        dataset = cfg["DATASET_NAME"]
        vmae = cfg["VMAE_VERSION"]
        datamode = cfg["DATA_MODE"]

        mode, loss, suffix = variant_name.split("_", 2)
        title = (
            f"{method.upper()} - {dataset} [{vmae}] ({datamode})\n"
            f"Mode: {mode} | Loss: {loss} | Suffix: {suffix}\n"
            f"ARI={ari:.4f} | NMI={nmi:.4f}"
        )
        ax.set_title(title)
        cmap = plt.get_cmap("tab20")
        for label in np.unique(labels):
            mask = np.array(labels) == label
            color = cmap(label % 20)
            mask_train = mask & (np.array(sources) == "train")
            if mask_train.any():
                ax.scatter(
                    proj[mask_train, 0],
                    proj[mask_train, 1],
                    s=8,
                    c=[color],
                    alpha=0.2,
                    marker="o",
                    label=f"{label_encoder.classes_[label]} (train)",
                )
                            # test → 三角 (^)
            mask_test = mask & (np.array(sources) == "test")
            if mask_test.any():
                ax.scatter(
                    proj[mask_test, 0],
                    proj[mask_test, 1],
                    s=15,
                    c=[color],
                    alpha=1.0,
                    marker="^",
                    label=f"{label_encoder.classes_[label]} (test)",
                )


        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()


# ====== 4. 実行フロー ======

def main(cfg: Dict[str, object]) -> None:
    df = load_labels(cfg)
    print("✅ DATA_MODE:", cfg["DATA_MODE"])
    print("✅ action value_counts:\n", df["action"].value_counts())

    # --- ラベルエンコード (アクション) ---
    le_act = LabelEncoder().fit(df["action"])
    df["act_id"] = le_act.transform(df["action"])

    # --- ループ: mode × loss × suffix ---
    for mode in cfg["VECTOR_MODES"]:
        for loss in cfg["LOSS_FUNCTION"]:
            for suffix in cfg["SUFFIXES"]:
                suffix_24fps = "_24fps" if cfg["USE_24FPS"] else ""
                variant = f"{mode}_{loss}_{suffix}{suffix_24fps}"
                print(f"\n=== Processing {variant} ===")

                # 1) ベクトル読み込み
                vecs = load_vectors(cfg, mode, suffix_24fps)

                # 2) モデル読み込み
                encoder = build_encoder("mlp" in suffix)
                model_path = (
                    Path("models")
                    / Path("model_mae")
                    / cfg["DATASET_NAME"]
                    / cfg["VMAE_VERSION"]
                    / loss
                    / f"{suffix}.pth"
                )
                encoder.load_state_dict(torch.load(model_path, map_location="cuda"), strict=False)

                # 3) 対象動画のみ抽出 & 埋め込み生成
                df_valid = df[df["video_path"].isin(vecs)].copy()
                a_vecs, a_labels, paths, sources = extract_action_embeddings(vecs, df_valid, encoder)

                # 4) 評価 & 可視化
                out_dir = Path("results") / Path("result_mae") / cfg["DATASET_NAME"] / cfg["VMAE_VERSION"] / cfg["DATA_MODE"]
                ensure_dirs(out_dir)
                ari, nmi = evaluate_clustering(a_vecs, a_labels, variant, out_dir)
                visualize(a_vecs, a_labels, le_act, paths, sources, cfg, variant, ari, nmi)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(CONFIG)
