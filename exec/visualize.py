import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP
from tqdm import tqdm

from model import (
    GatedFusion,
    ActionLinearNet, ActionMLPNet, 
)
# =============== CONFIG ===============
def setup_environment():
    os.environ["OMP_NUM_THREADS"] = "2"
    torch.set_grad_enabled(False)

def load_all_labels(train_csv_path: Path, test_csv_path: Path) -> Tuple[pd.DataFrame, LabelEncoder]:
    """訓練・テストのラベルを読み込み、source列を追加して結合する"""
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_train["source"] = "train"
    df_test["source"] = "test"
    
    full_df = pd.concat([df_train, df_test], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()
    
    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])
    
    return full_df, le_act

def load_all_features(df: pd.DataFrame, feature_dirs: Dict[str, Path], vmae_json_path: Path) -> Dict[str, Dict]:
    """必要な特徴量(flow, x3d, vmae)をすべてメモリに読み込む"""
    features = {"flow": {}, "x3d": {}, "vmae": {}}
    
    # VMAE
    if vmae_json_path.exists():
        with open(vmae_json_path, 'r') as f:
            features["vmae"] = json.load(f)
            
    # X3D / Flow
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading features"):
        path_str = row["video_path"]
        vid = Path(path_str).stem
        for key in ["flow", "x3d"]:
            npy_path = feature_dirs[key] / vid / f"{vid}.npy"
            if npy_path.exists():
                arr = np.load(npy_path)
                features[key][path_str] = arr.squeeze(0) if arr.ndim > 1 else arr
    return features


# ================================================================
# 2. モデル構築と特徴抽出 (整理)
# ================================================================
def build_and_load_encoder(job: Dict, D: int) -> nn.Module:
    use_mlp = job.get("use_mlp", False)
    encoder_cls = ActionMLPNet if use_mlp else ActionLinearNet
    encoder = encoder_cls(D, 256, 256) if use_mlp else encoder_cls(D, 256)
    
    # state_dictのロードはjobにfusionキーがあるかで分岐
    state_dict = torch.load(job["checkpoint_path"])
    if 'fusion' in state_dict:
        encoder.load_state_dict(state_dict['net'])
    else:
        # SimpleNet/ActionNetのstate_dictを直接ロード
        # 注意: Simple... と Action... のネットワーク構造が互換性を持つ必要がある
        encoder.load_state_dict(state_dict.get('net', state_dict.get('action_encoder', state_dict)))
        
    return encoder.cuda().eval()

def extract_embeddings(df: pd.DataFrame, features: Dict, model: nn.Module, mode: str, fusion_model: nn.Module = None) -> Tuple:
    emb_list, labels, sources = [], [], []
    
    for _, row in df.iterrows():
        path = row["video_path"]
        
        # モードに応じて入力を準備
        if mode == 'flow':
            if path not in features['flow']: continue
            x = torch.tensor(features['flow'][path]).unsqueeze(0).float().cuda()
        elif mode == 'mae':
            if path not in features['vmae']: continue
            x = torch.tensor(features['vmae'][path]).unsqueeze(0).float().cuda()
        elif mode == 'gated':
            if path not in features['x3d'] or path not in features['vmae']: continue
            x3d = torch.tensor(features['x3d'][path]).unsqueeze(0).float().cuda()
            vmae = torch.tensor(features['vmae'][path]).unsqueeze(0).float().cuda()
            x, _ = fusion_model(x3d, vmae)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        a_vec = model(x)
        emb_list.append(nn.functional.normalize(a_vec, dim=-1).squeeze(0).cpu())
        labels.append(row["act_id"])
        sources.append(row["source"])
        
    return torch.stack(emb_list), labels, sources

# =============== CLUSTERING ===============
def evaluate_clustering(X: torch.Tensor, labels: List[int]) -> Tuple[float, float]:
    X_np = X.numpy()
    true_k = len(set(labels))
    pred = KMeans(n_clusters=true_k, random_state=0, n_init='auto').fit_predict(X_np)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: List[int],
    sources: List[str],
    label_encoder: LabelEncoder,
    title: str,
    out_path: Path
):
    """
    Args:
        embeddings (torch.Tensor): (N, D)次元の埋め込みベクトル。
        labels (List[int]): 各ベクトルの行動ラベルID。
        sources (List[str]): 各ベクトルの出所 ('train' or 'test')。
        label_encoder (LabelEncoder): ラベルIDとクラス名を対応させるエンコーダ。
        title (str): グラフのタイトル。
        out_path (Path): 画像の保存先パス。
    """
    print(f"Visualizing embeddings with t-SNE... Saving to {out_path}")
    
    # --- 1. 次元削減 ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(embeddings.numpy())

    # --- 2. 描画の準備 ---
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title(title, fontsize=16)
    cmap = plt.get_cmap("tab20")
    
    labels_np = np.array(labels)
    sources_np = np.array(sources)
    
    # --- 3. クラスごとに描画 ---
    for class_id in np.unique(labels_np):
        class_name = label_encoder.classes_[class_id]
        color = cmap(class_id % 20)
        
        # 訓練データをプロット (半透明の円)
        train_mask = (labels_np == class_id) & (sources_np == 'train')
        if np.any(train_mask):
            ax.scatter(
                proj[train_mask, 0], proj[train_mask, 1],
                color=color, marker='o', s=20, alpha=0.3,
                label=f"{class_name} (Train)"
            )
            
        # テストデータをプロット (枠付きの三角)
        test_mask = (labels_np == class_id) & (sources_np == 'test')
        if np.any(test_mask):
            ax.scatter(
                proj[test_mask, 0], proj[test_mask, 1],
                color=color, marker='^', s=80, alpha=1.0,
                edgecolors='black', linewidths=0.5,
                label=f"{class_name} (Test)"
            )

    # --- 4. 凡例とレイアウト調整 ---
    ax.legend(
        markerscale=1.5,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    # --- 5. 保存 ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)



# =============== MAIN ===============
def main():
    setup_environment()

    # --- 基本設定 ---
    DATATYPE = "animalkingdom"
    train_csv = Path(f"./label/{DATATYPE}/train/train.csv") # 分割済みのCSVを想定
    test_csv = Path(f"./label/{DATATYPE}/test/labels_test.csv")
    
    feature_dirs = {
        "flow": Path(f"x3d_output/{DATATYPE}"),
        "x3d": Path(f"x3d_output/{DATATYPE}"),
    }
    vmae_json = Path(f"vector/{DATATYPE}/train/vectors_sliding_base.json")
    
    # --- データの事前読み込み ---
    full_df, le_act = load_all_labels(train_csv, test_csv)
    features = load_all_features(full_df, feature_dirs, vmae_json)

    # --- ✨評価したいモデルのリストをここで定義 ---
    # Optunaで見つけたベストモデルや、比較したいモデルの情報を辞書として追加する
    evaluation_jobs = [
        {
            "name": "Flow_MLP_Simple",
            "mode": "flow", "use_mlp": True,
            "checkpoint_path": "models/flow/animalkingdom/improved/simple_mlp_la1.0_ls0.00.pth"
        },
        {
            "name": "Gated_MLP_Adversarial",
            "mode": "gated", "use_mlp": True,
            "checkpoint_path": "models/gated/animalkingdom/improved/adv_mlp_la1.0_ls0.20.pth"
        },
        # 他に評価したいモデルがあればここに追加
    ]

    results = []
    for job in evaluation_jobs:
        print(f"\n===== Evaluating: {job['name']} =====")
        mode = job['mode']
        
        # モデル構築 & 重みロード
        fusion_model = None
        if mode == 'gated':
            fusion_model = GatedFusion(2048, 768, 512).cuda().eval()
            state_dict = torch.load(job['checkpoint_path'])
            if 'fusion' in state_dict: fusion_model.load_state_dict(state_dict['fusion'])
            D = 512
        elif mode == 'flow':
            D = 2048 # Flowの特徴次元数
        elif mode == 'mae':
            D = 768 # MAEの特徴次元数
            
        encoder = build_and_load_encoder(job, D)
        
        # 特徴抽出
        a_vecs, labels, sources = extract_embeddings(full_df, features, encoder, mode, fusion_model)
        
        # 評価
        ari, nmi = evaluate_clustering(a_vecs, labels)
        print(f"ARI={ari:.4f}, NMI={nmi:.4f}")

        out_dir = Path("results") / job['mode']
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job['name']}_tsne.png"

        title = f"t-SNE for {job['name']}\nNMI: {nmi:.4f}, ARI: {ari:.4f}"
        
        # 可視化
        visualize_embeddings(
            embeddings=a_vecs,
            labels=labels,
            sources=sources,
            label_encoder=le_act,
            title=title,
            out_path=out_path
        )
        
        results.append({"name": job['name'], "ari": ari, "nmi": nmi})

    # --- 最終結果の表示 ---
    print("\n\n===== FINAL RESULTS =====")
    results_df = pd.DataFrame(results).sort_values(by="nmi", ascending=False)
    print(results_df)
    results_df.to_csv("results/final_evaluation_summary.csv", index=False)

if __name__ == "__main__":
    main()