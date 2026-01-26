# eval.py
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# =================================
# モデルの import
# =================================
from model import (
    GatedFusion,  ActionMLPNet,
)

# train_coreから距離計算関数をimport
from train_core import compute_distance_stats_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")


def setup_environment() -> None:
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"


# =================================
# データロード
# =================================
def load_data_for_eval(
    train_label_paths: Union[str, List[str]], 
    target_test_csv: str,  # 今回評価する単一のCSVパス
    pooling: bool, 
    default_datatype: str = "animalkingdom"
):
    """評価用に train/test ラベルと特徴量をロードする。
    """

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
    full_df["act_id"] = le_act.transform(full_df["action"]) # 共通のLabelEncoderを使用

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

    encoder = ActionMLPNet(input_dim=D, feature_dim=256, hidden_dim=512).to(DEVICE).eval()
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

            # フレーム数が少ない方に合わせる
            n_windows = min(len(x_list), len(v_list))

            for x_vec, v_vec in zip(x_list[:n_windows], v_list[:n_windows]):
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
# 評価 (t-SNE/UMAP は削除)
# =================================
def compute_species_clf_accuracy(emb, df_meta, src):
    """
    埋め込みから species を予測する分類器の精度を計算。
    低い精度 = 種情報が除去されている（良いdisentanglement）
    """
    # test データのみで評価
    mask_test = (src == "test")
    X = emb[mask_test].numpy() if hasattr(emb[mask_test], 'numpy') else emb[mask_test]
    
    # species ラベルを取得
    if "species" not in df_meta.columns:
        print("⚠️ Species information not found in metadata columns:", df_meta.columns)   
        return None  # species 情報がない
    
    # df_metaはreset_indexされていない可能性があるので、valuesで取得
    df_meta_reset = df_meta.reset_index(drop=True)
    species_labels = df_meta_reset["species"].values[mask_test]
    
    # デバッグ: ラベルの中身確認
    # print(f"DEBUG: species_labels sample: {species_labels[:5]}")
    
    # ユニークなクラス数を確認
    unique_species = np.unique(species_labels)
    if len(unique_species) < 2:
        print(f"⚠️ Not enough unique species for classification: {unique_species}")
        return None  # 分類できない
    
    # LabelEncoding
    le_sp = LabelEncoder()
    y = le_sp.fit_transform(species_labels)
    
    # LogisticRegression で交差検証
    try:
        clf = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs')
        # 各クラスの最小サンプル数を考慮してn_splitsを決定
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        # クラスごとのサンプル数が少なすぎる場合の警告
        if min_class_count < 2:
             print(f"⚠️ Some classes have too few samples ({min_class_count}) for CV. Counts: {class_counts}")
             # それでも強引にstratified k-foldはできないので、n_splitsを計算
        
        n_splits = min(5, len(unique_species), min_class_count, len(y) // 2)
        if n_splits < 2:
            print(f"⚠️ Not enough splits possible: n_splits={n_splits} (min_class_count={min_class_count})")
            return None
            
        scores = cross_val_score(clf, X, y, cv=n_splits, scoring='accuracy')
        return float(np.mean(scores))
    except Exception as e:
        print(f"⚠️ Species classification failed with Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_metrics(emb, lab, src, le_act, name, out_dir):

    # === test のみで評価 ===
    mask_test = (src == "test")
    X_test = emb[mask_test].numpy()
    y_test = lab[mask_test]

    n_clusters = len(np.unique(y_test))
    if n_clusters < 2:
        return 0.0, 0.0, 0.0

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    )
    pred = clustering.fit_predict(X_test)
    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)
    
    try:
        sil = silhouette_score(X_test, y_test, metric='cosine')
    except:
        sil = 0.0
    
    return ari, nmi, sil


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


# =================================
# メイン
# =================================
def main(run_dir: Path, pooling_mode: str = "both"):
    setup_environment()

    run_dir = Path(run_dir)
    ablation_root = run_dir / "ablation"

    # ---- baseline_config.json 読み込み ----
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

    # デフォルト値
    DATATYPE = params.get("datatype", "animalkingdom")
    
    # pooling_mode に応じて POOLING_MODES を決定
    if pooling_mode == "true":
        POOLING_MODES = [True]
    elif pooling_mode == "false":
        POOLING_MODES = [False]
    else:
        POOLING_MODES = [True, False]

    # train_label_paths / test_label_paths を config から取得
    train_label_paths = params.get("train_label_paths", None)
    test_label_paths = params.get("test_label_paths", None)

    if train_label_paths is None:
        raise RuntimeError("[eval] 'train_label_paths' が必要です。")
    if test_label_paths is None:
        raise RuntimeError("[eval] 'test_label_paths' が必要です。")
    
    # リスト化
    if isinstance(test_label_paths, str):
        test_label_paths = [test_label_paths]

    # モデルファイルのリスト取得
    model_paths = list(ablation_root.glob("**/*.pth"))
    if not model_paths:
        print("⚠️ No model files found in ablation directory.")
        return

    # 各 pooling モードごとにループ
    for POOLING in POOLING_MODES:
        pooling_str = "pooling_true" if POOLING else "pooling_false"
        print(f"\n{'='*60}")
        print(f"🔄 Evaluating with {pooling_str} ...")
        print(f"{'='*60}")

        # 各 test set ごとにループ
        for test_csv in test_label_paths:
            test_stem = Path(test_csv).stem
            eval_root = run_dir / "eval" / test_stem / pooling_str
            eval_root.mkdir(parents=True, exist_ok=True)
            
            jsonl_root = eval_root / "jsonl"
            jsonl_root.mkdir(exist_ok=True)

            print(f"\n🚀 Evaluating on Test Set: {test_stem} ({pooling_str}) ...")

            # データロード (このテストセット用)
            full_df, le_act, features = load_data_for_eval(train_label_paths, test_csv, POOLING, default_datatype=DATATYPE)

            results = []

            for mp in tqdm(model_paths, desc=f"Models ({test_stem})"):
                p = params.copy()
                p["model_path"] = mp

                rel = mp.relative_to(ablation_root).parts
                key = rel[0]          # train_mode / adversarial / flow_preprocessing
                val = rel[1]          # gated / on / centered ...

                if key in {"train_mode", "adversarial", "flow_preprocessing"}:
                    p[key] = val

                if p.get("train_mode") is None:
                    continue

                model = build_and_load_model(p)
                if model is None:
                    continue

                emb, lab, src, df_meta = extract_embeddings(full_df, features, model, p)
                if emb is None:
                    continue

                # =============================
                # JSONL 保存 (jsonlフォルダへ)
                # =============================
                # train + test 両方を保存
                if len(emb) > 0:
                    save_jsonl(jsonl_root / f"{mp.stem}.jsonl",
                            df_meta, lab, src, emb)
                
                # メトリクス計算は test のみ
                mask_test = (src == "test")
                e_test = emb[mask_test]
                lab_test = lab[mask_test]

                if len(e_test) > 0:
                    # Metrics
                    lab_tensor = torch.tensor(lab_test, dtype=torch.long)
                    emb_tensor = e_test  # 既にTensor (cpu)

                    stats = compute_distance_stats_epoch(emb_tensor, lab_tensor)
                    
                    # ARI, NMI も計算
                    ari, nmi, _ = evaluate_metrics(emb, lab, src, le_act, mp.stem, eval_root)
                    
                    # Species 分類精度を計算 (低い = 種情報が除去されている)
                    species_clf_acc = compute_species_clf_accuracy(emb, df_meta, src)

                    res_dict = {
                        "name": mp.stem,
                        "train_mode": p.get("train_mode"),
                        "pooling": POOLING,
                        "ari": ari,
                        "nmi": nmi,
                        "species_clf_acc": species_clf_acc if species_clf_acc is not None else 0.0,
                    }
                    # stats の内容をマージ (intra_mean, intra_var, inter_mean, inter_var, silhouette)
                    if stats:
                        res_dict.update(stats)
                    else:
                        # 計算できなかった場合（クラス不足など）
                        res_dict.update({
                            "intra_mean": 0, "intra_var": 0,
                            "inter_mean": 0, "inter_var": 0,
                            "silhouette": 0
                        })

                    results.append(res_dict)

            # CSV 出力
            if results:
                pd.DataFrame(results).to_csv(eval_root / "eval_summary.csv", index=False)
                
                # Markdown 出力
                md_path = eval_root / "eval_summary.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# Evaluation Summary ({test_stem} - {pooling_str})\n\n")
                    # ヘッダー作成
                    headers = ["Model", "Train Mode", "Pooling", "ARI", "NMI", "Species Clf Acc ↓", "Silhouette", 
                               "Intra Mean", "Intra Var", "Inter Mean", "Inter Var"]
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
                    
                    for r in results:
                        row_vals = [
                            r['name'], r['train_mode'], str(r['pooling']),
                            f"{r['ari']:.4f}", f"{r['nmi']:.4f}", 
                            f"{r.get('species_clf_acc', 0):.4f}",
                            f"{r.get('silhouette', 0):.4f}",
                            f"{r.get('intra_mean', 0):.4f}", f"{r.get('intra_var', 0):.4f}",
                            f"{r.get('inter_mean', 0):.4f}", f"{r.get('inter_var', 0):.4f}"
                        ]
                        f.write("| " + " | ".join(row_vals) + " |\n")

                print(f"✅ Saved results for {test_stem} ({pooling_str}) to {eval_root}")

    # --- α ログの解析 & 可視化 (GatedFusion用) ---
    analyze_alpha_logs(run_dir, run_dir / "eval")  # 共通のevalフォルダに出力

    print("Done.")

# =================================
# α ログの解析（mean / var を epoch ごとに可視化）
# =================================
def analyze_alpha_logs(run_dir: Path, out_dir: Path):
    alpha_dir = run_dir / "alpha_logs"
    if not alpha_dir.exists():
        return

    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
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
            continue

        try:
            alpha_epoch = np.load(npy_path)  # shape: (N, D)
            alpha_epoch = alpha_epoch.astype(np.float32)
        except Exception:
            continue

        mean_alpha = float(alpha_epoch.mean())
        var_alpha = float(alpha_epoch.var())

        stats.append((epoch, mean_alpha, var_alpha))

    if not stats:
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

    # グラフ保存
    try:
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
        print(f"✅ Saved alpha stats to {out_dir}")
    except Exception as e:
        print(f"⚠️ Failed to plot alpha stats: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", type=str, help="Path to run directory")
    parser.add_argument("--pooling", type=str, choices=["true", "false", "both"], default="both", help="Pooling mode to evaluate")
    args = parser.parse_args()

    if args.run_dir:
        main(Path(args.run_dir), pooling_mode=args.pooling)
    else:
        dirs = [d for d in Path("train_result").glob("**/run_*") if d.is_dir()]
        dirs = sorted(dirs, reverse=True)
        if len(dirs) == 0:
            raise RuntimeError("No run_* directory found under train_result/")
        main(dirs[0], pooling_mode=args.pooling)
