import os
import json
from datetime import datetime
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

from model import (
    GatedFusion,
    DisentangleEmbedOnlySimple,
    DisentangleEmbedOnlyMLP
)

# =============== CONFIG ===============

os.environ["OMP_NUM_THREADS"] = "2"


# =============== UTILS ===============
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_labels(cfg: Dict[str, object]) -> pd.DataFrame:
    dtype = cfg["DATASET_NAME"]
    if dtype == "animalkingdom":
        df_train = pd.read_csv(Path("label") / dtype / "train" / "labels_filtered.csv")
    else:
        df_train = pd.read_csv(Path("label") / dtype / "train" / "labels.csv")

    suffix = "_24fps" if cfg["USE_24FPS"] else ""
    df_test = pd.read_csv(Path("label") / dtype / "test" / f"labels_test{suffix}.csv")
    df_train["video_path"] = df_train["video_path"].str.replace("\\", "/")
    df_test["video_path"] = df_test["video_path"].str.replace("\\", "/")
    df_train["source"] = "train"
    df_test["source"] = "test"
    if cfg["DATA_MODE"] == "train":
        return df_train.copy()
    elif cfg["DATA_MODE"] == "test":
        return df_test.copy()
    else:
        return pd.concat([df_train, df_test], ignore_index=True)


def build_encoder(use_mlp: bool, D: int, H=256):
    cls = DisentangleEmbedOnlyMLP if use_mlp else DisentangleEmbedOnlySimple
    return cls(D=D, H=H).cuda().eval()


def build_gated_infer(use_mlp: bool, d_x3d=2048, d_vmae=768, d_hidden=512, D=512, H=256):
    fusion = GatedFusion(d_x3d, d_vmae, d_hidden).cuda().eval()
    encoder_cls = DisentangleEmbedOnlyMLP if use_mlp else DisentangleEmbedOnlySimple
    encoder = encoder_cls(D=D, H=H).cuda().eval()
    return fusion, encoder


# =============== LOADERs ===============
def load_flow(cfg: Dict[str, object], suffix_24fps: str) -> Dict[str, np.ndarray]:
    dtype = cfg["DATASET_NAME"]
    flow_dir_train = Path("x3d_output") / dtype / "train"
    flow_dir_test = Path("x3d_output") / dtype / "test"
    dtype = cfg["DATASET_NAME"]
    if dtype == "animalkingdom":
        df_train = pd.read_csv(Path("label") / dtype / "train" / "labels_filtered.csv")
    else:
        df_train = pd.read_csv(Path("label") / dtype / "train" / "labels.csv")
    df_test = pd.read_csv(Path("label") / dtype / "test" / f"labels_test{suffix_24fps}.csv")
    df = pd.concat([df_train, df_test], ignore_index=True)
    flow_dict = {}
    for _, row in df.iterrows():
        path = row["video_path"].replace("\\", "/").strip()
        vid = Path(path).stem
        npy = flow_dir_train / vid / f"{vid}.npy"
        if not npy.exists():
            npy = flow_dir_test / vid / f"{vid}.npy"
        if npy.exists():
            arr = np.load(npy)
            flow_dict[path] = arr.squeeze(0) if arr.ndim > 1 else arr
    return flow_dict


def load_vectors(cfg: Dict[str, object], mode: str, suffix_24fps: str) -> Dict[str, List[float]]:
    dtype = cfg["DATASET_NAME"]
    vmae = cfg["VMAE_VERSION"]
    vecs = {}
    train = Path("vector") / dtype / "train" / f"vectors_{mode}_{vmae}.json"
    test = Path("vector") / dtype / "test" / f"vectors_{mode}{suffix_24fps}_{vmae}.json"
    for path in [train, test]:
        if path.exists():
            vecs.update(json.loads(path.read_text()))
    return vecs


# =============== EXTRACT ===============
def extract_flow(flow_dict: Dict[str, np.ndarray], df: pd.DataFrame, encoder) -> Tuple[torch.Tensor, List[int], List[str], List[str]]:
    emb_list, labels, paths, sources = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row["video_path"]
            if path not in flow_dict:
                continue
            z = torch.tensor(flow_dict[path]).unsqueeze(0).float().cuda()
            z = nn.functional.normalize(z, dim=-1)
            a_vec, _ = encoder(z)
            emb_list.append(a_vec.squeeze(0).cpu())
            labels.append(row["act_id"])
            paths.append(path)
            sources.append(row["source"])
    return torch.stack(emb_list), labels, paths, sources


def extract_mae(vecs, df, encoder):
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


def extract_gated(x3d_dict, vmae_dict, df, fusion, encoder):
    emb_list, labels, paths, sources = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row["video_path"]
            if path not in x3d_dict or path not in vmae_dict:
                continue
            x3d = torch.tensor(x3d_dict[path]).unsqueeze(0).float().cuda()
            vmae = torch.tensor(vmae_dict[path]).unsqueeze(0).float().cuda()
            fused, _ = fusion(x3d, vmae)
            fused = nn.functional.normalize(fused, dim=-1)
            a_vec, _ = encoder(fused)
            emb_list.append(a_vec.squeeze(0).cpu())
            labels.append(row["act_id"])
            paths.append(path)
            sources.append(row["source"])
    return torch.stack(emb_list), labels, paths, sources


# =============== CLUSTERING ===============
def evaluate(X: torch.Tensor, labels: List[int], variant: str, out_dir: Path) -> Tuple[float, float]:
    X_np = X.numpy()
    true_k = len(set(labels))
    pred = KMeans(true_k, random_state=0).fit_predict(X_np)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    print(f"[{variant}] ARI={ari:.4f}, NMI={nmi:.4f}")
    return ari, nmi


def visualize(X, labels, encoder, paths, sources, cfg, variant, ari, nmi):
    for method in cfg["VISUALIZE"]:
        proj = TSNE(n_components=2).fit_transform(X) if method == "tsne" else UMAP().fit_transform(X)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"{variant} {method.upper()} ARI={ari:.4f} NMI={nmi:.4f}")
        cmap = plt.get_cmap("tab20")
        for lbl in np.unique(labels):
            mask = np.array(labels) == lbl
            ax.scatter(proj[mask, 0], proj[mask, 1], s=10, label=f"{encoder.classes_[lbl]}")
        ax.legend()
        path = Path("result") / variant
        ensure_dirs(path)
        plt.savefig(path / f"{method}.png")
        plt.close()


# =============== MAIN ===============
def main(cfg: Dict[str, object]) -> None:
    df = load_labels(cfg)
    le_act = LabelEncoder().fit(df["action"])
    df["act_id"] = le_act.transform(df["action"])

    # --- ベスト結果を複数保持 ---
    results = []

    for mode in cfg["MODE"]:
        for loss in cfg["LOSS_FUNCTION"]:
            for suffix in cfg["SUFFIXES"]:
                for adv in cfg["ADV"]:
                    suffix_24fps = "_24fps" if cfg["USE_24FPS"] else ""
                    adv_str = f"-adv{adv:.2f}"
                    variant = f"{suffix}{adv_str}{suffix_24fps}"
                    print(f"=== {variant} ===")

                    if mode == "flow":
                        flow = load_flow(cfg, suffix_24fps)
                        D = next(iter(flow.values())).shape[0]
                        encoder = build_encoder("mlp" in suffix, D)
                        encoder.load_state_dict(torch.load(Path("models") / "model_flow" / cfg["DATASET_NAME"] / loss / f"flow_{suffix}{adv_str}.pth", weights_only=True), strict=False)
                        df_valid = df[df["video_path"].isin(flow)].copy()
                        a_vecs, labels, paths, sources = extract_flow(flow, df_valid, encoder)

                    elif mode == "mae":
                        vecs = load_vectors(cfg, cfg["VECTOR_MODES"][0], suffix_24fps)
                        encoder = build_encoder("mlp" in suffix, D=768)
                        encoder.load_state_dict(torch.load(Path("models") / "model_mae" / cfg["DATASET_NAME"] / loss / f"mae_{suffix}{adv_str}.pth", weights_only=True), strict=False)
                        df_valid = df[df["video_path"].isin(vecs)].copy()
                        a_vecs, labels, paths, sources = extract_mae(vecs, df_valid, encoder)

                    elif mode == "gated":
                        vecs = load_vectors(cfg, cfg["VECTOR_MODES"][0], suffix_24fps)
                        x3d_dir_train = Path("x3d_output") / cfg["DATASET_NAME"] / "train"
                        x3d_dir_test = Path("x3d_output") / cfg["DATASET_NAME"] / "test"
                        x3d_dict = {}
                        for path in vecs:
                            vid = Path(path).stem
                            npy = x3d_dir_train / vid / f"{vid}.npy"
                            if not npy.exists():
                                npy = x3d_dir_test / vid / f"{vid}.npy"
                            if npy.exists():
                                arr = np.load(npy)
                                x3d_dict[path] = arr.squeeze(0) if arr.ndim > 1 else arr
                        fusion, encoder = build_gated_infer("mlp" in suffix)
                        model_path = Path("models") / "model_gated" / cfg["DATASET_NAME"] / loss / f"gated_{suffix}{adv_str}.pth"
                        ckpt = torch.load(model_path, weights_only=True)
                        fusion.load_state_dict(ckpt["fusion"], strict=False)
                        encoder.load_state_dict(ckpt["net"], strict=False)
                        df_valid = df[df["video_path"].isin(x3d_dict) & df["video_path"].isin(vecs)].copy()
                        a_vecs, labels, paths, sources = extract_gated(x3d_dict, vecs, df_valid, fusion, encoder)

                    else:
                        raise ValueError(f"Unsupported MODE: {mode}")

                    out_dir = (
                        Path("result")
                        / mode
                        / cfg["DATASET_NAME"]
                        / loss
                        / cfg["DATA_MODE"]
                        / variant
                    )
                    ensure_dirs(out_dir)
                    ari, nmi = evaluate(a_vecs, labels, variant, out_dir)
                    visualize(a_vecs, labels, le_act, paths, sources, cfg, variant, ari, nmi)

                    results.append({
                        "ari": ari,
                        "nmi": nmi,
                        "mode": mode,
                        "loss": loss,
                        "suffix": suffix,
                        "variant": variant,
                        "adv": adv,
                        "dataset": cfg["DATASET_NAME"],
                        "data_mode": cfg["DATA_MODE"],
                        "vector_mode": cfg["VECTOR_MODES"][0]
                    })

        # --- 上位5件を NMI でソート ---
    results_sorted = sorted(results, key=lambda x: (x["nmi"], x["ari"]), reverse=True)[:5]
    results_sorted_all = sorted(results, key=lambda x: (x["nmi"], x["ari"]), reverse=True)


    print("\n=== ✅ TOP 5 RESULTS ===")
    for i, r in enumerate(results_sorted, 1):
        print(f"[Rank {i}] NMI={r['nmi']:.4f} | ARI={r['ari']:.4f}")
        print(f"  MODE   : {r['mode']}")
        print(f"  LOSS   : {r['loss']}")
        print(f"  SUFFIX : {r['suffix']}")
        print(f"  ADV    : {r['adv']}")
        print(f"  VECTOR : {r['vector_mode']}")
        print(f"  DATA   : {r['dataset']} | {r['data_mode']}")
        print(f"  VARIANT: {r['variant']}\n")
    # === テキストに保存 ===
    txt_out_path = Path("results") / "result_summary_all.txt"
    ensure_dirs(txt_out_path.parent)
    with open(txt_out_path, "w") as f:
        for i, r in enumerate(results_sorted_all, 1):
            f.write(f"[Rank {i}] NMI={r['nmi']:.4f} | ARI={r['ari']:.4f}\n")
            f.write(f"  MODE   : {r['mode']}\n")
            f.write(f"  LOSS   : {r['loss']}\n")
            f.write(f"  SUFFIX : {r['suffix']}\n")
            f.write(f"  ADV    : {r['adv']}\n")
            f.write(f"  VECTOR : {r['vector_mode']}\n")
            f.write(f"  DATA   : {r['dataset']} | {r['data_mode']}\n")
            f.write(f"  VARIANT: {r['variant']}\n\n")
    print(f"✅ TXT saved to: {txt_out_path.resolve()}")



if __name__ == "__main__":

    CONFIG: Dict[str, object] = {
        "MODE": ["mae", "flow", "gated"],  # 実行するモードをリストで
        "VECTOR_MODES": ["sliding"],
        "LOSS_FUNCTION": ["improved", "triplet"],
        "SUFFIXES": ["mlp-grl", "linear-grl", "mlp-nogrl", "linear-nogrl"],
        "VISUALIZE": ["tsne", "umap"],
        "VMAE_VERSION": "base",
        "DATA_MODE": "test",
        "USE_24FPS": True,
        "DATASET_NAME": "animalkingdom",
        "ADV": [0.10, 0.05],  # adversarial lossの係数
    }
    torch.set_grad_enabled(False)
    main(CONFIG)
