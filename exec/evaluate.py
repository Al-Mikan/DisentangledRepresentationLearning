import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
# Optional UMAP import with fallback handled later
try:
    from umap import UMAP  # type: ignore
    _HAS_UMAP = True
except Exception:
    UMAP = None  # type: ignore
    _HAS_UMAP = False

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


def setup_environment():
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "2"  # Windows + MKL のメモリリーク対策


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
    # VMAE features (train and test if available)
    if config['vmae_json'].exists():
        try:
            features["vmae"].update(json.loads(config['vmae_json'].read_text()))
        except Exception as e:
            print(f"⚠️ Failed to read VMAE train json: {e}")
    vmae_test_json = config.get('vmae_json_test')
    if vmae_test_json and Path(vmae_test_json).exists():
        try:
            features["vmae"].update(json.loads(Path(vmae_test_json).read_text()))
        except Exception as e:
            print(f"⚠️ Failed to read VMAE test json: {e}")

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

    # Be resilient to older PyTorch without weights_only
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=DEVICE)
    fused_dim = params.get('fused_dim', 512)
    D = fused_dim if params['train_mode'] == 'gated' else (2048 if params['train_mode'] == 'flow' else 768)

    # Fusion
    if params['train_mode'] == 'gated':
        models['fusion'] = GatedFusion(2048, 768, fused_dim).to(DEVICE).eval()
        fusion_state = {k.replace('fusion.', ''): v for k, v in state_dict.items() if k.startswith('fusion.')}
        if fusion_state:
            models['fusion'].load_state_dict(fusion_state)

    # Encoder: detect prefix and architecture from state_dict keys robustly
    possible_prefixes = ['action_encoder.', 'net.']
    chosen_prefix: Optional[str] = None
    for p in possible_prefixes:
        if any(k.startswith(p) for k in state_dict.keys()):
            chosen_prefix = p
            break
    if chosen_prefix is None:
        # As a fallback, try without prefix (keys like 'encoder.weight')
        chosen_prefix = ''

    enc_state = {k.replace(chosen_prefix, ''): v for k, v in state_dict.items() if k.startswith(chosen_prefix)}
    if len(enc_state) == 0:
        print(f"⚠️ Could not find encoder weights with prefixes {possible_prefixes + ['<root>']} in {model_path}")
        return None

    # 2) Infer architecture from keys
    keys = list(enc_state.keys())
    has_act = any(k.startswith('act_embed') for k in keys)
    has_mlp_pattern = any(k.startswith('act_embed.0') or k.startswith('encoder.0') for k in keys)
    if has_act:
        # Action encoder
        if has_mlp_pattern or any('.0.' in k for k in keys):
            encoder = ActionMLPNet(D, 256, 256).to(DEVICE).eval()
        else:
            encoder = ActionLinearNet(D, 256).to(DEVICE).eval()
    else:
        # Simple encoder
        if has_mlp_pattern or any('.0.' in k for k in keys):
            encoder = SimpleMLPNet(D, 256, 256).to(DEVICE).eval()
        else:
            encoder = SimpleLinearNet(D, 256).to(DEVICE).eval()

    # 3) Load weights into the inferred encoder
    try:
        encoder.load_state_dict(enc_state, strict=True)
    except Exception as e:
        # Try non-strict as a last resort to tolerate minor naming differences
        try:
            encoder.load_state_dict(enc_state, strict=False)
            print(f"ℹ️ Loaded encoder with non-strict mode due to: {e}")
        except Exception as ee:
            print(f"❌ Failed to load encoder state_dict: {ee}")
            return None

    # Expose a generic key and also original-style if applicable
    models['encoder'] = encoder
    if chosen_prefix.startswith('action_encoder'):
        models['action_encoder'] = encoder
    else:
        models['net'] = encoder
    return models


# =============================
# 埋め込み抽出
# =============================
def extract_embeddings(df, features, models, params):
    emb_list, labels, sources = [], [], []
    if 'encoder' in models:
        encoder = models['encoder']
    elif 'action_encoder' in models:
        encoder = models['action_encoder']
    else:
        encoder = models['net']
    fusion = models['fusion'] if 'fusion' in models else None
    mode = params['train_mode']

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
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
    if len(emb_list) == 0:
        return None, None, None
    return torch.stack(emb_list), np.array(labels), np.array(sources)


# =============================
# 評価 + 可視化
# =============================
def evaluate_and_visualize(embeddings, labels, sources, le_act, name, out_dir, metric='cosine'):
    out_dir.mkdir(parents=True, exist_ok=True)
    if embeddings is None or labels is None or sources is None:
        print(f"⚠️ No embeddings extracted for {name}")
        return np.nan, np.nan

    # ✅ test部分だけ抽出
    test_mask = (sources == 'test')
    X_test, y_test = embeddings[test_mask], labels[test_mask]
    if len(y_test) == 0:
        print(f"⚠️ No test samples for {name}")
        return np.nan, np.nan

    # クラスタ数・クラスタリング
    true_k = int(len(np.unique(y_test)))
    n_samples = int(len(y_test))
    n_clusters = max(1, min(true_k, n_samples))
    if n_clusters < 2:
        print(f"⚠️ Not enough clusters (n_clusters={n_clusters}) for {name}")
        return np.nan, np.nan

    X_np = X_test.numpy()
    pred = None
    if metric == 'cosine':
        try:
            clustering_model = AgglomerativeClustering(
                n_clusters=n_clusters, metric='cosine', linkage='average'
            )
            pred = clustering_model.fit_predict(X_np)
        except TypeError:
            clustering_model = AgglomerativeClustering(
                n_clusters=n_clusters, affinity='cosine', linkage='average'
            )
            pred = clustering_model.fit_predict(X_np)
        except Exception as e:
            print(f"⚠️ Agglomerative clustering failed ({e}); falling back to KMeans")

    if pred is None:
        try:
            pred = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_np)
        except Exception as e:
            print(f"❌ KMeans failed: {e}")
            return np.nan, np.nan

    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)
    print(f"📊 {name}: ARI={ari:.4f}, NMI={nmi:.4f}")

    # ✅ 可視化もtestのみ
    try:
        proj = TSNE(
            n_components=2,
            init='random',
            random_state=42,
            metric='cosine' if metric == 'cosine' else 'euclidean',
            perplexity=30,
            learning_rate='auto',
        ).fit_transform(X_np)
    except Exception as e:
        print(f"⚠️ t-SNE failed ({e}); skipping plot for {name}")
        return ari, nmi

    fig, ax = plt.subplots(figsize=(10, 8))
    for cid in np.unique(y_test):
        cname = le_act.classes_[cid]
        ax.scatter(
            proj[y_test == cid, 0], proj[y_test == cid, 1],
            s=20, label=cname
        )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.title(f"{name} | NMI={nmi:.3f} ARI={ari:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_test_tsne.png", dpi=150)
    plt.close()

    return ari, nmi


# =============================
# メイン処理
# =============================
def main(DATE_TAG):
    setup_environment()

    DATATYPE = "animalkingdom"
    ABLATION_CSV = Path(f"./train_result/{DATE_TAG}/ablations/ablation_results.csv")

    config = {
        'train_csv': Path(f"./label/{DATATYPE}/train/labels.csv"),
        'test_csv': Path(f"./label/{DATATYPE}/test/labels_test.csv"),
        'vmae_json': Path(f"./vector/{DATATYPE}/train/vectors_sliding_base.json"),
        'vmae_json_test': Path(f"./vector/{DATATYPE}/test/vectors_sliding_base.json"),
        'x3d_dir': Path(f"./x3d_output/{DATATYPE}"),
        'x3d_dir_centered': Path(f"./x3d_output_centered/{DATATYPE}"),
    }

    if not ABLATION_CSV.exists():
        # Auto-pick the most recent date directory (and run subdir) under train_result
        tr_root = Path("./train_result")
        if tr_root.exists():
            date_dirs = sorted([p for p in tr_root.iterdir() if p.is_dir()], reverse=True)
            for cand in date_dirs:
                # 1) direct ablations under date
                alt_csv = cand / "ablations" / "ablation_results.csv"
                if alt_csv.exists():
                    DATE_TAG = cand.name
                    ABLATION_CSV = alt_csv
                    print(f"ℹ️ Using latest ablation CSV: {ABLATION_CSV}")
                    break
                # 2) within run_* subdirectories
                run_dirs = sorted([p for p in cand.iterdir() if p.is_dir() and p.name.startswith("run_")], reverse=True)
                found = False
                for run in run_dirs:
                    alt_csv2 = run / "ablations" / "ablation_results.csv"
                    if alt_csv2.exists():
                        DATE_TAG = cand.name
                        ABLATION_CSV = alt_csv2
                        print(f"ℹ️ Using latest run ablation CSV: {ABLATION_CSV}")
                        found = True
                        break
                if found:
                    break
        if not ABLATION_CSV.exists():
            print(f"❌ CSV not found: {ABLATION_CSV}")
            return

    ablation_df = pd.read_csv(ABLATION_CSV)
    full_df, le_act, features = load_data_for_eval(config)
    eval_root = Path(f"./eval_result/{DATE_TAG}")
    # If the ablation CSV resides under a run_* subdir, mirror that structure for eval outputs
    try:
        parent_of_abl = ABLATION_CSV.parent.parent  # .../<maybe run_XXX>/ablations/ablation_results.csv
        if parent_of_abl.name.startswith("run_"):
            eval_root = eval_root / parent_of_abl.name
    except Exception:
        pass
    img_dir = eval_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for _, row in ablation_df.iterrows():
        name = row["name"]
        model_path = Path(row["model_path"].replace("\\", "/"))
        if not model_path.exists():
            print(f"⚠️ Missing model for {name}")
            continue

        # Prefer explicit columns from ablation CSV; fallback to name tokens
        tokens = set(Path(model_path).stem.split('_'))
        train_mode = row["train_mode"] if "train_mode" in ablation_df.columns else (
            "gated" if "gated" in tokens else ("flow" if "flow" in tokens else "mae")
        )
        if "use_mlp" in ablation_df.columns:
            use_mlp = bool(row["use_mlp"])
        else:
            use_mlp = ("mlp" in tokens) if ("mlp" in tokens or "nomlp" in tokens) else False
            if "nomlp" in tokens:
                use_mlp = False
        if "use_adversarial" in ablation_df.columns:
            use_adversarial = bool(row["use_adversarial"])
        else:
            use_adversarial = ("adv" in tokens)
        flow_preprocessing = (
            str(row["flow_preprocessing"]).strip().lower()
            if "flow_preprocessing" in ablation_df.columns and pd.notna(row["flow_preprocessing"])
            else ("centered" if "centered" in tokens else "normal")
        )

        params = {
            "model_path": model_path,
            "train_mode": train_mode,
            "use_mlp": use_mlp,
            "use_adversarial": use_adversarial,
            "flow_preprocessing": flow_preprocessing,
            "loss_type": "improved",  # ← 固定（評価には直接影響しない）
            "fused_dim": 512,
        }

        print(f"\n🧩 Evaluating {name} ({params['train_mode']}, adv={params['use_adversarial']}, mlp={params['use_mlp']})")
        models = build_and_load_model(params)
        if models is None:
            continue

        embeddings, labels, sources = extract_embeddings(full_df, features, models, params)
        ari, nmi = evaluate_and_visualize(embeddings, labels, sources, le_act, name, img_dir)

        # Pull optional columns safely
        val_loss = row["val_loss"] if "val_loss" in ablation_df.columns else np.nan
        metric_combined = row["metric_combined"] if "metric_combined" in ablation_df.columns else np.nan

        results.append({
            "name": name,
            "train_mode": params["train_mode"],
            "use_mlp": params["use_mlp"],
            "use_adversarial": params["use_adversarial"],
            "flow_preprocessing": params["flow_preprocessing"],
            "val_loss": val_loss,
            "val_metric_combined": metric_combined,
            "test_ari": ari,
            "test_nmi": nmi,
            "model_path": str(model_path),
        })

    results_df = pd.DataFrame(results)
    out_csv = eval_root / "eval_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved evaluation summary to {out_csv}")


if __name__ == "__main__":
    DATA_TAG = "2025-11-04/run_002"  # ← train_resultの日付と合わせる（自動検出フォールバックあり）
    main(DATA_TAG)