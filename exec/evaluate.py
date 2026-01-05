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
# モデルの import
# =================================
from model import (
    GatedFusion,  ActionMLPNet,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")


def setup_environment() -> None:
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"


# =================================
# データロード
# =================================
def load_data_for_eval(train_label_paths, test_label_path: str, pooling: bool, default_datatype: str = "animalkingdom"):
    """評価用に train/test ラベルと特徴量をロードする。

    train_label_paths: 学習時と同様、train 用ラベル CSV（文字列 or リスト）。
    test_label_path  : 評価用 test ラベル CSV（単一パス）。
    """

    print("📂 Loading labels and features... pooling =", pooling)

    # train_csv 群の読み込み（train ソース）
    if isinstance(train_label_paths, str):
        train_label_paths = [train_label_paths]

    train_dfs = []
    for p in train_label_paths:
        df_i = pd.read_csv(p)
        df_i["source"] = "train"
        train_dfs.append(df_i)
        print(f"  → [eval] Loaded TRAIN labels from: {p} (rows={len(df_i)})")

    if not train_dfs:
        raise RuntimeError("[eval] train_label_paths から有効な train CSV が読み込めませんでした。")

    train_df = pd.concat(train_dfs, ignore_index=True)

    # test_csv の読み込み（test ソース）
    test_df = pd.read_csv(test_label_path)
    test_df["source"] = "test"
    print(f"  → [eval] Loaded TEST labels from: {test_label_path} (rows={len(test_df)})")

    # train+test を連結
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()

    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])

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

    # full_df のすべての動画に対して feature をロード
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

    train_mode = params.get("train_mode")
    if train_mode == "gated":
        D = int(params.get("fused_dim", 512))   
    elif train_mode == "flow":
        D = 2048
    elif train_mode == "mae":
        D = 768
    else:
        raise ValueError("Unknown train_mode")

    # Fusion モデル
    if params["train_mode"] == "gated":
        fusion = GatedFusion(2048, 768, int(params.get("fused_dim", 512))   ).to(DEVICE).eval()
        fusion_state = {k.replace("fusion.", ""): v
                        for k, v in state_dict.items()
                        if k.startswith("fusion.")}
        fusion.load_state_dict(fusion_state, strict=False)
        models["fusion"] = fusion

    # Encoder
    prefix = next((p for p in ["action_encoder.", "net."]
                   if any(k.startswith(p) for k in state_dict)), "")
    enc_state = {k.replace(prefix, ""): v
                 for k, v in state_dict.items() if k.startswith(prefix)}

    encoder = ActionMLPNet(D, 256, 256).to(DEVICE).eval()
    encoder.load_state_dict(enc_state, strict=False)
    models["encoder"] = encoder

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
    labels = []
    sources = []
    meta_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting ({mode})"):
        p = row["video_path"]

        if mode == "gated":
            if p not in features[flow_key] or p not in features["vmae"]:
                continue
            x_list = features[flow_key][p]
            v_list = features["vmae"][p]

            assert len(x_list) == len(v_list), (
                f"[GatedFusion] Sliding window mismatch for {p}: "
                f"flow={len(x_list)}, vmae={len(v_list)}"
            )

            for x_vec, v_vec in zip(x_list, v_list):
                xx = torch.tensor(x_vec).unsqueeze(0).float().to(DEVICE)
                vv = torch.tensor(v_vec).unsqueeze(0).float().to(DEVICE)
                fused, _ = fusion(xx, vv)
                emb = encoder(fused)
                emb = nn.functional.normalize(emb, dim=-1)

                emb_list.append(emb.squeeze(0).cpu())
                labels.append(row["act_id"])
                sources.append(row["source"])
                meta_rows.append(row)

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
                meta_rows.append(row)

    if not emb_list:
        return None, None, None, None

    return torch.stack(emb_list), np.array(labels), np.array(sources), pd.DataFrame(meta_rows)


# =================================
# 可視化用（ラベル + train/test）
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

    plt.figure(figsize=(10, 8))
    handled = set()

    for lab in uniq:
        for src in ["train", "test"]:
            mask = (label_names == lab) & (s == src)
            if mask.sum() == 0:
                continue

            legend_name = f"{lab} ({src})"
            lbl = legend_name if legend_name not in handled else None
            handled.add(legend_name)

            plt.scatter(
                X[mask, 0], X[mask, 1],
                color=color_map[lab],
                marker=marker[src],
                s=14, alpha=0.75,
                label=lbl
            )

    plt.title(title)
    plt.xticks([]); plt.yticks([])

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


# =================================
# 評価 + 可視化
# =================================
def evaluate_and_visualize(emb, lab, src, le_act, name, out_dir):

    tsne_all_dir = out_dir / "tsne" / "all"
    tsne_test_dir = out_dir / "tsne" / "test_only"
    umap_all_dir = out_dir / "umap" / "all"
    umap_test_dir = out_dir / "umap" / "test_only"

    for d in [tsne_all_dir, tsne_test_dir, umap_all_dir, umap_test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === test のみで評価 ===
    mask_test = (src == "test")
    X_test = emb[mask_test].numpy()
    y_test = lab[mask_test]

    n_clusters = len(np.unique(y_test))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    pred = clustering.fit_predict(X_test)
    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)

    # ==========
    # t-SNE（all）
    # ==========
    try:
        ts = TSNE(n_components=2, random_state=42, perplexity=30)

        X_all = emb.numpy()
        X2_all = ts.fit_transform(X_all)

        X2_test = X2_all[mask_test]

        _plot_with_source(
            X2_all, lab, src, le_act,
            f"t-SNE (train+test unified) - {name}",
            tsne_all_dir / f"{name}.png"
        )

        _plot_with_source(
            X2_test, y_test, np.array(["test"] * len(y_test)), le_act,
            f"t-SNE (test only, same coords) - {name}",
            tsne_test_dir / f"{name}.png"
        )

    except Exception as e:
        print("t-SNE failed:", e)

    # ==========
    # UMAP（all）
    # ==========
    try:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")

        U_all = reducer.fit_transform(emb.numpy())
        _plot_with_source(
            U_all, lab, src, le_act,
            f"UMAP (train+test) - {name}",
            umap_all_dir / f"{name}.png"
        )

        U_test = reducer.fit_transform(X_test)
        _plot_with_source(
            U_test, y_test, np.array(["test"] * len(y_test)), le_act,
            f"UMAP (test only) - {name}",
            umap_test_dir / f"{name}.png"
        )

    except Exception as e:
        print("UMAP failed:", e)

    return ari, nmi


# =================================
# α ログの解析（mean / var を epoch ごとに可視化）
# =================================
def analyze_alpha_logs(run_dir: Path, out_dir: Path):
    alpha_dir = run_dir / "alpha_logs"
    if not alpha_dir.exists():
        print("ℹ️ alpha_logs directory not found, skip alpha analysis.")
        return

    stats = []  # (epoch, mean, var)

    for npy_path in sorted(alpha_dir.glob("*.npy")):
        stem = npy_path.stem  # e.g. alpha_trial023_epoch027
        epoch = None
        for part in stem.split("_"):
            if part.startswith("epoch"):
                try:
                    epoch = int(part.replace("epoch", ""))
                except ValueError:
                    pass

        if epoch is None:
            print(f"⚠️ Could not parse epoch from filename: {npy_path.name}")
            continue

        try:
            alpha_epoch = np.load(npy_path)  # shape: (N, D)
            alpha_epoch = alpha_epoch.astype(np.float32)
        except Exception as e:
            print(f"⚠️ Failed to load {npy_path}: {e}")
            continue

        mean_alpha = float(alpha_epoch.mean())
        var_alpha = float(alpha_epoch.var())

        stats.append((epoch, mean_alpha, var_alpha))

    if not stats:
        print("ℹ️ No valid alpha logs found.")
        return

    # epoch 順にソート
    stats.sort(key=lambda x: x[0])
    epochs  = [s[0] for s in stats]
    means   = [s[1] for s in stats]
    variances = [s[2] for s in stats]

    # CSV 保存
    alpha_csv = out_dir / "alpha_stats.csv"
    df_alpha = pd.DataFrame({
        "epoch": epochs,
        "alpha_mean": means,
        "alpha_var": variances,
    })
    df_alpha.to_csv(alpha_csv, index=False)
    print(f"✅ Saved alpha stats → {alpha_csv}")

    # グラフ保存（mean と var を同じ図に描く）
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, means, marker="o", label="mean α")
    plt.plot(epochs, variances, marker="s", label="var α")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Gating α statistics over epochs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = out_dir / "alpha_mean_var.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"✅ Saved alpha mean/var plot → {out_path}")


# =================================
# メイン
# =================================
def main(run_dir: Path):
    setup_environment()

    run_dir = Path(run_dir)
    ablation_root = run_dir / "ablation"

    # ---- baseline_config.json 読み込み ----
    baseline_path = run_dir / "baseline_config.json"
    if baseline_path.exists():
        with open(baseline_path, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)
        params = {}
        params.update(base_cfg.get("params", {}))
        params.update(base_cfg.get("user_attrs", {}))

        if "config_used" in base_cfg and isinstance(base_cfg["config_used"], dict):
            params.update(base_cfg["config_used"])
            print("📘 Expanded config_used into params")
    else:
        params = {}

    # デフォルト値
    DATATYPE = params.get("datatype", "animalkingdom")
    POOLING = bool(params.get("pooling", True))

    # train_label_paths / test_label_path を config から取得（必須扱い）
    train_label_paths = params.get("train_label_paths", None)
    test_label_path = params.get("test_label_path", None)

    # ---- run_note.txt から上書き（あれば）----
    note_path = run_dir / "run_note.txt"
    if note_path.exists():
        try:
            txt = note_path.read_text(encoding="utf-8")
            json_part = txt.split("=== Run Configuration (After Training) ===")[-1]
            run_info = json.loads(json_part)

            DATATYPE = run_info.get("datatype", DATATYPE)
            POOLING = run_info.get("pooling", POOLING)

            params.setdefault("datatype", DATATYPE)
            params.setdefault("pooling", POOLING)

            print(f"📘 Loaded run config from run_note: datatype={DATATYPE}, pooling={POOLING}")
        except Exception as e:
            print("⚠ Could not read run_note.txt:", e)

    # train_label_paths / test_label_path の最終確認
    if train_label_paths is None:
        raise RuntimeError("[eval] 'train_label_paths' が baseline_config.json / params に含まれていません。")
    if test_label_path is None:
        raise RuntimeError("[eval] 'test_label_path' が baseline_config.json / params に含まれていません。")

    eval_root = run_dir / "eval"
    eval_root.mkdir(exist_ok=True)

    full_df, le_act, features = load_data_for_eval(train_label_paths, test_label_path, POOLING, default_datatype=DATATYPE)

    results = []
    model_paths = list(ablation_root.glob("**/*.pth"))

    for mp in tqdm(model_paths, desc="Evaluating"):
        rel = mp.relative_to(ablation_root).parts
        key = rel[0] if len(rel) > 0 else "unknown"
        val = rel[1] if len(rel) > 1 else "unknown"

        p = params.copy()
        p["model_path"] = mp
        if key == "train_mode":
            p["train_mode"] = val
        if key == "flow_preprocessing":
            p["flow_preprocessing"] = val

        model = build_and_load_model(p)
        if model is None:
            continue

        emb, lab, src, df_meta = extract_embeddings(full_df, features, model, p)
        if emb is None:
            continue

        # =============================
        # train / test に分割して JSONL 保存
        # =============================
        mask_train = (src == "train")
        mask_test  = (src == "test")

        df_train = df_meta[mask_train]
        df_test  = df_meta[mask_test]

        e_train = emb[mask_train]
        e_test  = emb[mask_test]

        save_jsonl(eval_root / f"{mp.stem}_train.jsonl",
                   df_train, lab[mask_train], src[mask_train], e_train)

        save_jsonl(eval_root / f"{mp.stem}_test.jsonl",
                   df_test, lab[mask_test], src[mask_test], e_test)

        ari, nmi = evaluate_and_visualize(emb, lab, src, le_act, mp.stem, eval_root)

        results.append({
            "name": mp.stem,
            "train_mode": p.get("train_mode"),
            "flow_preprocessing": p.get("flow_preprocessing"),
            "pooling": POOLING,
            "ari": ari,
            "nmi": nmi,
        })

    # CSV 出力
    pd.DataFrame(results).to_csv(eval_root / "eval_summary.csv", index=False)

    # Markdown 出力
    md_path = eval_root / "eval_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Summary\n\n")
        f.write("| Model | Train Mode | Flow Preprocess | Pooling | ARI | NMI |\n")
        f.write("|-------|------------|-----------------|---------|------|------|\n")
        for r in results:
            f.write(
                f"| {r['name']} | {r['train_mode']} | {r['flow_preprocessing']} | "
                f"{r['pooling']} | {r['ari']:.4f} | {r['nmi']:.4f} |\n"
            )

     # --- α ログの解析 & 可視化 ---
    analyze_alpha_logs(run_dir, eval_root)
    
    print("Done.")


# =================================
# JSONL 保存ユーティリティ
# =================================
def save_jsonl(path: Path, df, labels, sources, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(len(embeddings)):
            obj = {
                "videopath": df.iloc[i]["video_path"],
                "label": df.iloc[i]["action"],
                "act_id": int(labels[i]),
                "source": df.iloc[i]["source"],
                "vector": embeddings[i].tolist(),
            }
            f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        dirs = [d for d in Path("train_result").glob("**/run_*") if d.is_dir()]
        dirs = sorted(dirs, reverse=True)
        if len(dirs) == 0:
            raise RuntimeError("No run_* directory found under train_result/")
        main(dirs[0])
