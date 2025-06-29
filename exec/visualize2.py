import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import (
    DisentangleEmbedOnlySimple,
    DisentangleEmbedOnlyMLP
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP
import pyperclip
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import mplcursors


# === 設定 ===
MODES = ["sliding"]
LOSSES = ["triplet", "improved"]
SUFFIXES = ["linear_grl", "mlp_grl", "linear_nogrl", "mlp_nogrl"]
VISUALIZE = ["tsne", "umap"]
VMAE_VERSION = "base"

DATA_MODE = "all"   # "train", "test", "all"
USE_24FPS = False
USE_UCF = False
INTERACTIVE = True
DATATYPE = "wolf"  # "animalkingdom", "wolf"


# === データ読み込み ===
if DATATYPE == "wolf":
    df_train = pd.read_csv('./label/wolf/train/labels.csv')
    df_test = pd.read_csv('./label/wolf/test/labels_test.csv')
else:
    df_train = pd.read_csv('./label/animalkingdom/train/labels.csv')
    df_test = pd.read_csv('./label/animalkingdom/test/labels_test.csv')
df_train['video_path'] = df_train['video_path'].str.replace('\\', '/')
df_test['video_path'] = df_test['video_path'].str.replace('\\', '/')

if DATA_MODE == "train":
    df = df_train.copy()
elif DATA_MODE == "test":
    df = df_test.copy()
elif DATA_MODE == "all":
    df = pd.concat([df_train, df_test], ignore_index=True)
else:
    raise ValueError(f"Unknown DATA_MODE: {DATA_MODE}")

if USE_UCF:
    df_ucf = pd.read_csv('./label/ucf/test/labels_ucf.csv')
    df_ucf['video_path'] = df_ucf['video_path'].str.replace('\\', '/').apply(os.path.basename)
    df = pd.concat([df, df_ucf], ignore_index=True)

print("✅ 使用中の DATA_MODE:", DATA_MODE)
print("✅ 全データに含まれるユニークな action:")
print(df['action'].value_counts())

os.makedirs("result", exist_ok=True)
os.makedirs(f"result/{DATA_MODE}", exist_ok=True)
for method in VISUALIZE:
    os.makedirs(f"result/{DATA_MODE}/{method}", exist_ok=True)

metric_file = f"result/{DATATYPE}/{DATA_MODE}/metrics.txt"
if os.path.exists(metric_file):
    os.remove(metric_file)

# === ラベルエンコード ===
le_act = LabelEncoder().fit(df['action'])
df['act_id'] = le_act.transform(df['action'])

# === Embedding 抽出 ===
def get_action_embeddings(vecs, df, model):
    vecs_list, labels, paths = [], [], []
    with torch.no_grad():
        for _, row in df.iterrows():
            path = row['video_path']
            if path not in vecs:
                continue
            z = torch.tensor(vecs[path]).unsqueeze(0).float().cuda()
            a_vec, _ = model(z)
            vecs_list.append(a_vec.squeeze(0).cpu())
            labels.append(row['act_id'])
            paths.append(path)
    return torch.stack(vecs_list), labels, paths


# === クラスタリング評価 ===
def evaluate_clustering(vecs, labels, name):
    n_clusters = len(np.unique(labels))
    preds = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)
    ari = adjusted_rand_score(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds)
    print(f"📊 {name} ARI: {ari:.4f} | NMI: {nmi:.4f}")
    with open(metric_file, 'a') as f:
        f.write(f"{name}\n ARI = {ari:.4f} | NMI = {nmi:.4f}\n")


# === プロット ===
def plot_embedding(proj, labels, label_encoder, paths, title, out_path, interactive):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')

    scatter_plots = []
    scatter_indices = []  # 各 scatter がどのインデックスを使ってるか保持

    for label in unique_labels:
        mask = np.array(labels) == label
        color = cmap(label % 20)
        sc = ax.scatter(proj[mask, 0], proj[mask, 1],
                        s=5, c=[color], label=label_encoder.classes_[label])
        scatter_plots.append(sc)
        scatter_indices.append(np.where(mask)[0])  # ← ここで全体 index を保持

    legend = ax.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left',
                       borderaxespad=0., fontsize=8, title="Actions")

    fig.tight_layout()

    if interactive:
        cursor = mplcursors.cursor(scatter_plots, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            sc_idx = scatter_plots.index(sel.artist)  # どの scatter か特定
            local_idx = sel.index                     # その scatter 内のインデックス
            true_idx = scatter_indices[sc_idx][local_idx]  # 全体 index に変換
            sel.annotation.set_text(f"{label_encoder.classes_[labels[true_idx]]}\n{paths[true_idx]}")
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
            sel.annotation.set_fontsize(8)

        def on_click(event):
            if event.inaxes != ax:
                return
            for sc, idxs in zip(scatter_plots, scatter_indices):
                cont, ind = sc.contains(event)
                if cont:
                    local_idx = ind["ind"][0]
                    true_idx = idxs[local_idx]
                    path = paths[true_idx]
                    pyperclip.copy(path)
                    print(f"📋 Copied: {path}")
                    break
        fig.canvas.mpl_connect("button_press_event", on_click)
        def on_select(verts):
            mask = Path(verts).contains_points(proj)
            selected = [paths[i] for i, m in enumerate(mask) if m]
            if selected:
                joined = "\n".join(selected)
                pyperclip.copy(joined)
                print("\n-----\n✏️ Selected paths:\n", joined, "\n-----")
        LassoSelector(ax, on_select)

    plt.savefig(out_path, bbox_inches='tight', bbox_extra_artists=[legend])
    # plt.show()


# === メイン ===
for mode in MODES:
    for loss in LOSSES:
        for suffix in SUFFIXES:
            suffix_full = f"{mode}_{loss}_{suffix}{suffix_24fps}"
            print(f"=== Processing: {suffix_full} ===")

            suffix_24fps = "_24fps" if USE_24FPS else ""

            vecs = {}
            if DATATYPE == "wolf":
                train_vec = f'vector/{DATATYPE}/train/vectors_{mode}{suffix_24fps}.json'
                test_vec = f'vector/{DATATYPE}/test/vectors_{mode}{suffix_24fps}.json'
            else:
                train_vec = f'vector/{DATATYPE}/train/vectors_{mode}{suffix_24fps}.json'
                test_vec = f'vector/{DATATYPE}/train/vectors_{mode}{suffix_24fps}.json'
            if os.path.exists(train_vec):
                with open(train_vec) as f:
                    vecs.update(json.load(f))

            if DATA_MODE in ["test", "all"] and os.path.exists(test_vec):
                with open(test_vec) as f:
                    vecs.update(json.load(f))

            if USE_UCF:
                with open(f'vector/{DATATYPE}/train/vectors_{mode}_ucf_labels.json') as f:
                    vecs.update(json.load(f))

            if "nonlinear" in suffix:
                net = DisentangleEmbedOnlyMLP().cuda()
            else:
                net = DisentangleEmbedOnlySimple().cuda()

            net.load_state_dict(torch.load(f'./model/{DATATYPE}/{VMAE_VERSION}/{loss}/{suffix}.pth'), strict=False)
            net.eval()

            # === vecs に存在するものだけを valid とする ===
            df_valid = df[df['video_path'].isin(vecs)].copy()
            a_vecs, a_labels, paths = get_action_embeddings(vecs, df_valid, net)

            evaluate_clustering(a_vecs, a_labels, f"{suffix_full} Action")

            for method in VISUALIZE:
                if method == "tsne":
                    proj = TSNE(n_components=2, random_state=0).fit_transform(a_vecs)
                else:
                    proj = UMAP().fit_transform(a_vecs)

                out_dir = f"result/{DATATYPE}/{DATA_MODE}/{method}"
                os.makedirs(out_dir, exist_ok=True)
                out_file = f"{out_dir}/{suffix_full}.png"

                plot_embedding(proj, a_labels, le_act, paths,
                           f"Action {method.upper()}", out_file, INTERACTIVE)
