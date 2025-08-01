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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import optuna
from umap import UMAP

# 必要なモデルクラスをインポート
from model import (
    GatedFusion, ActionLinearNet, ActionMLPNet, SimpleLinearNet, SimpleMLPNet
)

# ============================================
# ユーティリティ & 評価関数
# ============================================

def setup_environment():
    """環境設定を初期化"""
    torch.set_grad_enabled(False)
    os.environ["OMP_NUM_THREADS"] = "1"

def load_data_for_eval(config: Dict) -> Tuple[pd.DataFrame, LabelEncoder, Dict]:
    """評価に必要なラベルと特徴量をすべて読み込む"""
    print("Loading labels and features...")
    train_df = pd.read_csv(config['train_csv'])
    test_df = pd.read_csv(config['test_csv'])
    train_df['source'] = 'train'
    test_df['source'] = 'test'
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["video_path"] = full_df["video_path"].str.replace("\\", "/").str.strip()
    
    le_act = LabelEncoder().fit(full_df["action"])
    full_df["act_id"] = le_act.transform(full_df["action"])
    
    features = {"flow": {}, "flow_centered": {}, "vmae": {}}
    if config['vmae_json'].exists():
        features["vmae"] = json.loads(config['vmae_json'].read_text())
    
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

def build_and_load_model(params: Dict):
    """Optunaのパラメータに基づいてモデルを構築し、重みをロードする"""
    models = nn.ModuleDict()
    model_path = Path(params['model_path'])
    if not model_path.exists():
        print(f"⚠️ Model not found: {model_path}")
        return None
    state_dict = torch.load(model_path, weights_only=True)
    
    fused_dim = params.get('fused_dim', 512)
    D = fused_dim if params['train_mode'] == 'gated' else (2048 if params['train_mode'] == 'flow' else 768)
    
    if params['train_mode'] == 'gated':
        models['fusion'] = GatedFusion(2048, 768, fused_dim).cuda().eval()
        fusion_state_dict = {k.replace('fusion.', ''): v for k, v in state_dict.items() if k.startswith('fusion.')}
        if fusion_state_dict:
            models['fusion'].load_state_dict(fusion_state_dict)

    model_key = 'action_encoder' if params['use_adversarial'] else 'net'
    model_cls = (ActionMLPNet if params['use_mlp'] else ActionLinearNet) if params['use_adversarial'] else (SimpleMLPNet if params['use_mlp'] else SimpleLinearNet)
    encoder = (model_cls(D, 256, 256) if params['use_mlp'] else model_cls(D, 256)).cuda().eval()
    
    encoder_prefix = f"{model_key}."
    encoder_state_dict = {k.replace(encoder_prefix, ''): v for k, v in state_dict.items() if k.startswith(encoder_prefix)}
    
    if encoder_state_dict:
        encoder.load_state_dict(encoder_state_dict)
        models[model_key] = encoder
    else:
        print(f"⚠️ Keys with prefix '{encoder_prefix}' not found in checkpoint for {model_path}.")
        return None
        
    return models

def extract_embeddings(df, features, models, params):
    emb_list, labels, sources = [], [], []
    model_key = 'action_encoder' if params['use_adversarial'] else 'net'
    encoder = models[model_key]
    fusion = models['fusion'] if 'fusion' in models else None
    mode = params['train_mode']

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        path = row["video_path"]
        
        if mode == 'gated':
            flow_key = 'flow_centered' if params.get('flow_preprocessing') == 'centered' else 'flow'
            if path not in features[flow_key] or path not in features['vmae']: continue
            x3d_tensor = torch.tensor(features[flow_key][path]).unsqueeze(0).float().cuda()
            vmae_tensor = torch.tensor(features['vmae'][path]).unsqueeze(0).float().cuda()
            x, _ = fusion(x3d_tensor, vmae_tensor)
        else:
            feature_key = 'vmae' if mode == 'mae' else ('flow_centered' if params.get('flow_preprocessing') == 'centered' else 'flow')
            if path not in features[feature_key]: continue
            x = torch.tensor(features[feature_key][path]).unsqueeze(0).float().cuda()

        a_vec = encoder(x)
        emb_list.append(nn.functional.normalize(a_vec, dim=-1).squeeze(0).cpu())
        labels.append(row["act_id"])
        sources.append(row["source"])
        
    return torch.stack(emb_list), np.array(labels), np.array(sources)

def evaluate_and_visualize(embeddings, labels, sources, le_act, name, out_dir, metric_for_eval='cosine', vis_methods=['tsne', 'umap']):
    # --- ✨修正点: 訓練データとテストデータに分割 ---
    train_mask = (sources == 'train')
    test_mask = (sources == 'test')
    
    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_test, y_test = embeddings[test_mask], labels[test_mask]

    # --- テストデータのみでクラスタリング評価 ---
    X_test_np = X_test.numpy()
    true_k = len(np.unique(y_test))
    
    if metric_for_eval == 'cosine':
        clustering_model = AgglomerativeClustering(n_clusters=true_k, metric='cosine', linkage='average')
        pred = clustering_model.fit_predict(X_test_np)
    else:
        pred = KMeans(n_clusters=true_k, random_state=42, n_init='auto').fit_predict(X_test_np)

    ari = adjusted_rand_score(y_test, pred)
    nmi = normalized_mutual_info_score(y_test, pred)
    print(f"[TEST SET] ARI={ari:.4f}, NMI={nmi:.4f} (metric: {metric_for_eval})")

    # --- 全データで可視化 ---
    for method in vis_methods:
        print(f"Visualizing with {method.upper()}...")
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1), metric=metric_for_eval, max_iter=1000)
        elif method == 'umap':
            reducer = UMAP(n_components=2, random_state=42, metric=metric_for_eval)
        else: continue
            
        proj = reducer.fit_transform(embeddings.numpy())
        
        proj_train = proj[train_mask]
        proj_test = proj[test_mask]
        
        title = f"{method.upper()} for {name}\nTest NMI: {nmi:.4f}, Test ARI: {ari:.4f}"
        out_path = out_dir / f"{name}_{method}.png"
        
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(title, fontsize=16)
        cmap = plt.get_cmap("tab20")
        
        for class_id in np.unique(labels):
            class_name = le_act.classes_[class_id]
            color = cmap(class_id % 20)
            
            # 訓練データをプロット
            train_class_mask = (y_train == class_id)
            if np.any(train_class_mask):
                ax.scatter(proj_train[train_class_mask, 0], proj_train[train_class_mask, 1], color=color, marker='o', s=20, alpha=0.3, label=f"{class_name} (Train)")
            
            # テストデータをプロット
            test_class_mask = (y_test == class_id)
            if np.any(test_class_mask):
                ax.scatter(proj_test[test_class_mask, 0], proj_test[test_class_mask, 1], color=color, marker='^', s=80, alpha=1.0, edgecolors='black', linewidths=0.5, label=f"{class_name} (Test)")

        handles, labels_list = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels_list, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"✅ {method.upper()} visualization saved to {out_path}")
    
    return ari, nmi

# ============================================
# メイン実行ブロック
# ============================================
def main():
    setup_environment()
    
    STUDY_DB_PATH = "sqlite:///optuna_study.db"
    STUDY_NAME = "disentangle-supcon-study-v1" # 適切なStudy名に変更
    TOP_N_TRIALS = 5
    DATATYPE = 'animalkingdom'

    config = {
        'train_csv': Path(f"./label/{DATATYPE}/train/train.csv"),
        'test_csv': Path(f"./label/{DATATYPE}/test/labels_test.csv"),
        'vmae_json': Path(f"./vector/{DATATYPE}/train/vectors_sliding_base.json"),
        'x3d_dir': Path(f"x3d_output/{DATATYPE}"),
        'x3d_dir_centered': Path(f"x3d_output_centered/{DATATYPE}"),
    }
    
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STUDY_DB_PATH)
        trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        top_trials = sorted(trials, key=lambda t: t.value)[:TOP_N_TRIALS]
        print(f"Found {len(top_trials)} best trials to evaluate.")
    except Exception as e:
        print(f"❌ Could not load Optuna study: {e}")
        return

    full_df, le_act, features = load_data_for_eval(config)

    final_results = []
    for rank, trial in enumerate(top_trials, 1):
        params = trial.params
        run_name = f"trial_{trial.number}_{params['train_mode']}_{params.get('loss_type', 'default')}"
        print(f"\n===== Evaluating Rank {rank} | {run_name} =====")
        print(f"Params: {params}")
        
        # train.pyで保存したモデルパスを復元
        params['model_path'] = Path(f"./models/{params['train_mode']}/{DATATYPE}/{params['loss_type']}/{run_name}_best.pth")

        models = build_and_load_model(params)
        if models is None: continue
        
        params['fused_dim'] = 512
        params['feature_dim'] = 256

        a_vecs, labels, sources = extract_embeddings(full_df, features, models, params)
        
        out_dir = Path("results") / STUDY_NAME
        eval_metric = 'cosine' if params.get('loss_type') in ['cosine', 'supcon'] else 'euclidean'
        ari, nmi = evaluate_and_visualize(a_vecs, labels, sources, le_act, run_name, out_dir, metric_for_eval=eval_metric)
        
        result_summary = {'rank': rank, 'trial_number': trial.number, 'val_loss': trial.value, 'test_ari': ari, 'test_nmi': nmi, **params}
        final_results.append(result_summary)
        
    results_df = pd.DataFrame(final_results)
    print("\n\n===== FINAL EVALUATION SUMMARY =====")
    display_cols = ['rank', 'trial_number', 'val_loss', 'test_nmi', 'test_ari', 'train_mode', 'use_adversarial', 'loss_type', 'lr', 'flow_preprocessing']
    display_cols = [col for col in display_cols if col in results_df.columns]
    if not results_df.empty and display_cols:
        print(results_df[display_cols])
    
    results_df.to_csv(f"results/{STUDY_NAME}_summary.csv", index=False)
    print(f"\n✅ Summary saved to results/{STUDY_NAME}_summary.csv")

if __name__ == "__main__":
    main()