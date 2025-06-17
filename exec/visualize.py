import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import DisentangleNet, DisentangleNet2
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MODES = ["simple", "adaptive3d", "adaptive1d","sliding"]
LOSSES = ["triplet", "improved"]
USE_TEST = True


# --- 可視化対象のラベルを限定する（None の場合は全ラベル表示）
VISIBLE_ACTIONS = None  # 例: ["walking", "eating"]
VISIBLE_SPECIES = None  # 例: ["polar bear", "zebra"]

# --- ラベル読み込み ---
df = pd.read_csv('labels.csv')
df_test = pd.read_csv('labels_test.csv')
df['video_path'] = df['video_path'].str.replace('\\', '/')
df_test['video_path'] = df_test['video_path'].str.replace('\\', '/')
if USE_TEST:
    df = pd.concat([df, df_test], ignore_index=True)

# --- ラベルエンコード ---
le_act = LabelEncoder().fit(df['action'])
le_sp  = LabelEncoder().fit(df['species'])
df['act_id'] = le_act.transform(df['action'])
df['sp_id']  = le_sp.transform(df['species'])

A = len(le_act.classes_)
S = len(le_sp.classes_)

# --- 埋め込み抽出関数 ---
def get_embeddings(vecs_dict, df, model):
    a_vecs, s_vecs, a_labels, s_labels = [], [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row['video_path']
            if path not in vecs_dict:
                continue
            z = torch.tensor(vecs_dict[path]).unsqueeze(0).float().cuda()
            a_vec, s_vec = model(z)
            a_vecs.append(a_vec.squeeze(0).cpu())
            s_vecs.append(s_vec.squeeze(0).cpu())
            a_labels.append(row['act_id'])
            s_labels.append(row['sp_id'])
    return torch.stack(a_vecs), torch.stack(s_vecs), a_labels, s_labels

# --- 評価関数 ---
def evaluate_clustering(vecs, true_labels, name="",metric_file=None):
    n_clusters = len(np.unique(true_labels))

    # ✅ KMeans++ 初期化を明示
    preds = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit_predict(vecs)
    # preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)
    ari = adjusted_rand_score(true_labels, preds)
    nmi = normalized_mutual_info_score(true_labels, preds)
    print(f"\n📊 {name} のクラスタリング評価")
    print(f"  🔹 ARI  : {ari:.4f}")
    print(f"  🔹 NMI  : {nmi:.4f}")

    # テキストファイルにも保存
    if metric_file:
        with open(metric_file, 'a') as f:
            f.write(f"{name}\n")
            f.write(f"  ARI = {ari:.4f}\n")
            f.write(f"  NMI = {nmi:.4f}\n\n")

# --- 可視化関数 ---
def plot_embedding(ax, proj, labels, label_encoder, title, show_legend=True):
    ax.set_title(title)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')
    for i, label in enumerate(unique_labels):
        idx = np.array(labels) == label
        label_name = label_encoder.inverse_transform([label])[0]
        ax.scatter(proj[idx, 0], proj[idx, 1], s=5, color=cmap(i % 20), label=label_name)
    if show_legend:
        ax.legend(fontsize=6, markerscale=3)

# --- 結果ディレクトリ作成 ---
metric_file = "result/metrics.txt"
if os.path.exists(metric_file):
    os.remove(metric_file) 

os.makedirs("result/action", exist_ok=True)
os.makedirs("result/species", exist_ok=True)

# --- 全パターンループ ---
for mode in MODES:
    for loss in LOSSES:
        print(f"\n=== 処理中: mode={mode}, loss={loss} ===")

        # ベクトル読み込み
        vec_path = f'exec/vectors_{mode}.json'
        vec_test_path = f'exec/vectors_{mode}_test.json'
        with open(vec_path) as f:
            vecs = json.load(f)
        if USE_TEST:
            with open(vec_test_path) as f:
                vecs_test = json.load(f)
            vecs.update(vecs_test)

        # モデル読み込み
        model_path = f'./model/disentangled_{mode}_{loss}.pth'
        if loss == "cross":
            net = DisentangleNet2(D=768, H=256, A=A, S=S).cuda()
        else:
            net = DisentangleNet(D=768, H=256).cuda()
        
        net.load_state_dict(torch.load(model_path))
        net.eval()

        # 有効データのみフィルタ
        df_valid = df[df['video_path'].apply(lambda p: p in vecs)]

        # 可視化対象の action / species があればさらにフィルタ
        if VISIBLE_ACTIONS is not None:
            df_valid = df_valid[df_valid['action'].isin(VISIBLE_ACTIONS)]
        if VISIBLE_SPECIES is not None:
            df_valid = df_valid[df_valid['species'].isin(VISIBLE_SPECIES)]

        # 埋め込み取得
        a_vecs, s_vecs, a_labels, s_labels = get_embeddings(vecs, df_valid, net)

        # 評価
        evaluate_clustering(a_vecs, a_labels, f"{mode}/{loss} - Action", metric_file)
        evaluate_clustering(s_vecs, s_labels, f"{mode}/{loss} - Species", metric_file)

        # t-SNE
        a_proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(a_vecs)
        s_proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(s_vecs)

        # 描画と保存（フォルダ分けて）
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        plot_embedding(ax1, a_proj, a_labels, le_act, "Action Embedding", show_legend=True)
        save_path1 = f"result/action/{mode}_{loss}.png"
        plt.savefig(save_path1, bbox_inches='tight')
        plt.close()

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        plot_embedding(ax2, s_proj, s_labels, le_sp, "Species Embedding", show_legend=False)
        save_path2 = f"result/species/{mode}_{loss}.png"
        plt.savefig(save_path2, bbox_inches='tight')
        plt.close()

        print(f"📷 Action 結果 → {save_path1}")
        print(f"📷 Species 結果 → {save_path2}")
