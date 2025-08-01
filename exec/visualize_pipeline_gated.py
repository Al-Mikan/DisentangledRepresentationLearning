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
    pairwise_distances,
)
from umap import UMAP
import json

from model import (
    GatedFusion,
    DisentangleEmbedOnlySimple,
    DisentangleEmbedOnlyMLP,
)

# ======================================
# 1. 設定
# ======================================
CONFIG: Dict[str, object] = {
    "VECTOR_MODES": ["sliding"],  # 使用する特徴ベクトルのモード
    "LOSS_FUNCTION": ["improved"],  # 学習で使った損失関数
    "SUFFIXES": ["mlp-grl", "mlp-nogrl", "linear-grl", "linear-nogrl"],  # モデルパターン
    "VISUALIZE": ["tsne", "umap"],  # 可視化手法
    "VMAE_VERSION": "base",  # VMAEバージョン名
    "DATA_MODE": ["test", "all"],  # train / test / all
    "USE_24FPS": False,  # 24fpsで使うか
    "DATASET_NAME": "wolf",  # データセット名
}

# ======================================
# 2. Utility 関数群
# ======================================

def ensure_dirs(*paths: Path) -> None:
    """指定したディレクトリがなければ作成する"""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def load_labels(cfg: Dict[str, object]) -> pd.DataFrame:
    """train/test のラベルCSVを読み込んで結合する"""
    dtype = cfg["DATASET_NAME"]
    base = Path("label") / dtype
    suffix = "_24fps" if cfg["USE_24FPS"] else ""

    #animalkingdomのときはlabels_filtered
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

def load_x3d_and_vmae(cfg: Dict[str, object], mode: str, suffix_24fps: str) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    """
    train と test 両方の vectors を読み込み、辞書を合体して返す
    """
    dtype = cfg["DATASET_NAME"]
    vmae = cfg["VMAE_VERSION"]
    vecs: Dict[str, List[float]] = {}

    # === train vectors ===
    vmae_train_path = Path("vector") / dtype / "train" / f"vectors_{mode}_{vmae}.json"
    # === test vectors ===
    vmae_test_path = Path("vector") / dtype / "test" / f"vectors_{mode}{suffix_24fps}_{vmae}.json"

    # === 両方マージ ===
    for path in [vmae_train_path, vmae_test_path]:
        if path.exists():
            vecs.update(json.loads(path.read_text()))

    # === X3D は train/test 両方のパスに対応して検索する ===
    x3d_dir_train = Path("x3d_output") / Path("animalkingdom") / "train"
    x3d_dir_test = Path("x3d_output") / Path("animalkingdom") / "test"
    x3d_dict = {}

    for video_path in vecs.keys():
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        npy_train = x3d_dir_train / video_id / f"{video_id}.npy"
        npy_test = x3d_dir_test / video_id / f"{video_id}.npy"

        npy_path = npy_train if npy_train.exists() else npy_test
        if npy_path.exists():
            arr = np.load(str(npy_path))
            if arr.ndim > 1:
                arr = arr.squeeze(0)
            x3d_dict[video_path] = arr

    return x3d_dict, vecs


def build_gated_infer(use_mlp: bool, d_x3d=2048, d_vmae=768, d_hidden=512, D=512, H=256) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    GatedFusion + 推論用EmbedOnly ヘッドをロードして返す
    """
    fusion = GatedFusion(d_x3d=d_x3d, d_vmae=d_vmae, d_hidden=d_hidden).cuda().eval()
    encoder_cls = DisentangleEmbedOnlyMLP if use_mlp else DisentangleEmbedOnlySimple
    encoder = encoder_cls(D=D, H=H).cuda().eval()
    return fusion, encoder

def extract_action_embeddings_gated_infer(
    x3d_dict: Dict[str, np.ndarray],
    vmae_dict: Dict[str, List[float]],
    df: pd.DataFrame,
    fusion: torch.nn.Module,
    encoder: torch.nn.Module,
) -> Tuple[torch.Tensor, List[int], List[str], List[str]]:
    """
    GatedFusionでx3dとvmaeを結合 → 推論用ヘッドで行動埋め込みを抽出
    """
    emb_list, labels, paths, sources = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row["video_path"]
            if path not in x3d_dict or path not in vmae_dict:
                continue
            x3d = torch.tensor(x3d_dict[path]).unsqueeze(0).float().cuda()
            vmae = torch.tensor(vmae_dict[path]).unsqueeze(0).float().cuda()
            fused_vec, _ = fusion(x3d, vmae)
            fused_vec = nn.functional.normalize(fused_vec, dim=-1)
            a_vec, _ = encoder(fused_vec)
            emb_list.append(a_vec.squeeze(0).cpu())
            labels.append(row["act_id"])
            paths.append(path)
            sources.append(row["source"])
    return torch.stack(emb_list), labels, paths, sources

# ======================================
# 3. 評価と可視化
# ======================================

def evaluate_clustering(X: torch.Tensor, labels: List[int], name: str, out_dir: Path, max_k: int = 10) -> Tuple[float, float]:
    """クラスタリングのARI/NMIを計算"""
    print(f"\n=== 📊 Evaluating {name} ===")
    X_np = X.numpy()
    true_k = len(set(labels))

    k_candidates = range(2, min(max_k, len(X_np)))
    sil_scores = [silhouette_score(X_np, KMeans(k, random_state=0).fit_predict(X_np)) for k in k_candidates]
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

    print(f"KMeans (k={best_k}) ARI={ari_k:.4f}, NMI={nmi_k:.4f}")
    print(f"Agglomerative ARI={ari_a:.4f}, NMI={nmi_a:.4f}")
    print(dbscan_result)

    log_file = out_dir / "clustering_log.txt"
    with open(log_file, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] {name}\n")
        f.write(f"KMeans ARI={ari_k:.4f} NMI={nmi_k:.4f}\n")
        f.write(f"Agglomerative ARI={ari_a:.4f} NMI={nmi_a:.4f}\n")
        f.write(f"{dbscan_result}\n\n")

    return ari_k, nmi_k

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
    vis_out_base = Path("result_gated") / cfg["DATASET_NAME"] / cfg["VMAE_VERSION"] / cfg["DATA_MODE"]
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
            f"{method.upper()} - {dataset} ({datamode})\n"
            f"Mode: {mode} | Loss: {loss} | Suffix: {suffix}\n"
            f"ARI={ari:.4f} | NMI={nmi:.4f}"
        )
        ax.set_title(title)
        num_train = sum([s == "train" for s in sources])
        num_test = sum([s == "test" for s in sources])

        info_text = (
            f"Train: {num_train}    Test: {num_test}\n"
        )

        fig.subplots_adjust(bottom=0.22)  # 下側の余白を広げる（調整可）
        fig.text(
            0.99, 0.01,          # x=0.99, y=0.01 → 右下
            info_text,
            fontsize=9,
            ha='right',          # テキスト右端が(0.99, 0.01)に合う
            va='bottom',         # 下端揃え
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
        )

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
                    s=30,  # 少し大きめでも見やすい
                    c=[color],
                    alpha=1.0,
                    marker="^",
                    edgecolors="black",    # 縁取り色
                    linewidths=0.2,        # 縁の太さ
                    label=f"{label_encoder.classes_[label]} (test)",
                )


        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        fig.tight_layout()
        plt.savefig(out_path)
        plt.close()

# ======================================
# 4. メインフロー
# ======================================

def main(cfg: Dict[str, object]) -> None:
    for datamode in cfg["DATA_MODE"]:
        print(f"\n=== [MODE] {datamode} ===")
        # === DATA_MODE を一時的に上書き ===
        cfg_one = cfg.copy()
        cfg_one["DATA_MODE"] = datamode

        # === ラベル読み込み ===
        df = load_labels(cfg_one)
        le_act = LabelEncoder().fit(df["action"])
        df["act_id"] = le_act.transform(df["action"])

        for mode in cfg["VECTOR_MODES"]:
            for loss in cfg["LOSS_FUNCTION"]:
                for suffix in cfg["SUFFIXES"]:
                    suffix_24fps = "_24fps" if cfg["USE_24FPS"] else ""
                    variant = f"{mode}_{loss}_{suffix}{suffix_24fps}"
                    print(f"\n=== Processing {variant} ===")

                    # === 特徴量ロード ===
                    x3d_dict, vmae_dict = load_x3d_and_vmae(cfg_one, mode, suffix_24fps)

                    # === モデルロード ===
                    model_path = Path("models")/Path("model_gated") / cfg["DATASET_NAME"] / loss / f"{suffix}.pth"
                    use_mlp = "mlp" in suffix
                    fusion, encoder = build_gated_infer(use_mlp=use_mlp)

                    checkpoint = torch.load(model_path, map_location="cuda")
                    fusion.load_state_dict(checkpoint['fusion'])
                    encoder.load_state_dict(checkpoint['net'], strict=False)

                    # === 有効な動画パスでフィルタリング ===
                    df_valid = df[
                        (df["video_path"].isin(x3d_dict.keys())) &
                        (df["video_path"].isin(vmae_dict.keys()))
                    ].copy()

                    if df_valid.empty:
                        print("⚠️ 有効なサンプルがありません。スキップします。")
                        continue

                    # === 埋め込み生成 ===
                    a_vecs, a_labels, paths, sources = extract_action_embeddings_gated_infer(
                        x3d_dict, vmae_dict, df_valid, fusion, encoder
                    )

                    # === 評価 & 可視化 ===
                    out_dir =Path("results")/ Path("result_gated") / cfg["DATASET_NAME"] / cfg["VMAE_VERSION"] / datamode
                    ensure_dirs(out_dir)
                    ari, nmi = evaluate_clustering(a_vecs, a_labels, variant, out_dir)
                    visualize(a_vecs, a_labels, le_act, paths, sources, cfg_one, variant, ari, nmi)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main(CONFIG)