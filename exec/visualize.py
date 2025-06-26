import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from model import DisentangleNet, DisentangleNet2
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pyperclip
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from umap import UMAP

MODES = ["sliding"]
LOSSES = ["triplet", "improved"]
USE_TEST = True
USE_24FPS = True
USE_UCF = True
VISIBLE_ACTIONS = None
VISIBLE_SPECIES = None
TEST_ONLY = True


# --- ラベル読み込み ---
df = pd.read_csv('labels.csv')
if USE_24FPS:
    df_test = pd.read_csv('labels_test_24fps.csv')
elif USE_TEST:
    df_test = pd.read_csv('labels_test.csv')
df['video_path'] = df['video_path'].str.replace('\\', '/')
df_test['video_path'] = df_test['video_path'].str.replace('\\', '/')
if USE_TEST and TEST_ONLY:
    df = df_test.copy()
elif USE_TEST:
    df = pd.concat([df, df_test], ignore_index=True)
if USE_UCF:
    df_ucf = pd.read_csv('labels_ucf.csv')
    df_ucf['video_path'] = df_ucf['video_path'].str.replace('\\', '/')
    df_ucf['video_path'] = df_ucf['video_path'].apply(lambda p: os.path.basename(p))
    df = pd.concat([df, df_ucf], ignore_index=True)

# --- ラベルエンコード ---
le_act = LabelEncoder().fit(df['action'])
le_sp = LabelEncoder().fit(df['species'])
df['act_id'] = le_act.transform(df['action'])
df['sp_id'] = le_sp.transform(df['species'])
A = len(le_act.classes_)
S = len(le_sp.classes_)

# --- 埋め込み抽出（video_path付き） ---
def get_embeddings_with_paths(vecs_dict, df, model):
    a_vecs, s_vecs, a_labels, s_labels, paths = [], [], [], [], []
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
            paths.append(path)
    return torch.stack(a_vecs), torch.stack(s_vecs), a_labels, s_labels, paths

# --- クラスタリング評価 ---
def evaluate_clustering(vecs, true_labels, name="", metric_file=None):
    n_clusters = len(np.unique(true_labels))
    preds = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit_predict(vecs)
    ari = adjusted_rand_score(true_labels, preds)
    nmi = normalized_mutual_info_score(true_labels, preds)
    print(f"\n📊 {name} のクラスタリング評価")
    print(f"  🔹 ARI  : {ari:.4f}")
    print(f"  🔹 NMI  : {nmi:.4f}")
    if metric_file:
        with open(metric_file, 'a') as f:
            f.write(f"{name}\n  ARI = {ari:.4f}\n  NMI = {nmi:.4f}\n\n")

# --- インタラクティブ可視化 ---
def interactive_plot_embedding(proj, labels, label_encoder, paths, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(title)
    label_names = label_encoder.classes_
    colors = plt.get_cmap('tab20')(np.array(labels) % 20)
    sc = ax.scatter(proj[:, 0], proj[:, 1], s=5, c=colors)


    # hover 表示
    cursor = mplcursors.cursor(sc, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        sel.annotation.set_text(f"{label_names[labels[idx]]}\n{paths[idx]}")
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.8)
        sel.annotation.set_fontsize(8)

    # クリック時コピー（マウスイベントだけに登録）
    def on_click(event):
        if event.inaxes != ax:
            return
        cont, ind = sc.contains(event)
        if cont:
            idx = ind["ind"][0]
            path = paths[idx]
            pyperclip.copy(path)
            print(f"📋 Copied to clipboard: {path}")

    # イベント登録は cursor ではなく fig.canvas に対して行う！
    fig.canvas.mpl_connect("button_press_event", on_click)


    # 自由選択時に一致パスを出力＋クリップボードコピー
    def on_select(verts):
        path_obj = Path(verts)
        selected = path_obj.contains_points(proj)
        selected_paths = [paths[i] for i, flag in enumerate(selected) if flag]
        if selected_paths:
            joined = '\n'.join(selected_paths)
            pyperclip.copy(joined)
            print("\n-----")
            print("\n✏️ Lasso selected paths (copied):")
            for p in selected_paths:
                print(" •", p)
            print("-----\n")  # 区切り線を閉じる

    lasso = LassoSelector(ax, on_select)
    plt.show()

# --- 結果ディレクトリ作成 ---
metric_file = "result/metrics.txt"
os.makedirs("result/action", exist_ok=True)
os.makedirs("result/species", exist_ok=True)
if os.path.exists(metric_file):
    os.remove(metric_file)

# --- メインループ ---
for mode in MODES:
    for loss in LOSSES:
        print(f"\n=== 処理中: mode={mode}, loss={loss} ===")
        suffix = "_24fps" if USE_24FPS else ""
        vec_path = f'exec/vectors_{mode}.json'
        vec_test_path = f'exec/vectors_{mode}_test{suffix}.json'
        with open(vec_path) as f:
            vecs = json.load(f)
        if USE_TEST:
            with open(vec_test_path) as f:
                vecs_test = json.load(f)
            vecs.update(vecs_test)
        if USE_UCF:
            with open(f'exec/vectors_{mode}_ucf_labels.json') as f:
                vecs_ucf = json.load(f)
            vecs.update(vecs_ucf)

        model_path = f'./model/disentangled_{mode}_{loss}.pth'
        net = DisentangleNet2(D=768, H=256, A=A, S=S).cuda() if loss == "cross" else DisentangleNet(D=768, H=256).cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()

        df_valid = df[df['video_path'].isin(vecs.keys())].copy()
        a_vecs, s_vecs, a_labels, s_labels, video_paths = get_embeddings_with_paths(vecs, df_valid, net)

        evaluate_clustering(a_vecs, a_labels, f"{mode}/{loss} - Action", metric_file)
        evaluate_clustering(s_vecs, s_labels, f"{mode}/{loss} - Species", metric_file)

        a_proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(a_vecs.numpy())
        interactive_plot_embedding(a_proj, a_labels, le_act, video_paths, "🎬 Action Embedding")
