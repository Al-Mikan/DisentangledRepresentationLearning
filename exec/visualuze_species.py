# ───────────────────────────────── visualize.py ─────────────────────────────
import os, json, pickle
import numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from model import DisentangleEmbedOnlySimple, DisentangleEmbedOnlyMLP
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP

# -------------- 設定 --------------------------------------------------------
MODES      = ["sliding"]
LOSSES     = ["triplet", "improved"]
SUFFIXES   = ["nonlinear_grl", "linear_grl", "nonlinear_nogrl", "linear_nogrl"]
VISUALIZE  = ["tsne", "umap"]
DATA_MODE  = "all"        # train + test を必ず表示
USE_24FPS  = True
USE_UCF    = False
RESULT_DIR = "result2"    # 出力先

# -------------- ① データ読み込み -------------------------------------------
df_train = pd.read_csv("labels.csv")
df_test  = pd.read_csv("labels_test_24fps.csv" if USE_24FPS else "labels_test.csv")
for _df in (df_train, df_test):
    _df["video_path"] = _df["video_path"].str.replace("\\", "/")

df = pd.concat([df_train, df_test], ignore_index=True)   # all= train+test
if USE_UCF:
    df_ucf = pd.read_csv("labels_ucf.csv")
    df_ucf["video_path"] = df_ucf["video_path"].str.replace("\\", "/").apply(os.path.basename)
    df = pd.concat([df, df_ucf], ignore_index=True)

# -------------- ② 学習時の LabelEncoder をロード ---------------------------
with open("model/label_encoder_action.pkl", "rb") as f:
    le_train = pickle.load(f)        # <- 学習時に pickle.dump しておく

# “未知 action” → act_id = -1 にする
def encode_or_unknown(act_name:str) -> int:
    if act_name in le_train.classes_:
        return int(le_train.transform([act_name])[0])
    return -1

df["act_id"] = df["action"].apply(encode_or_unknown)
print("🆕 学習時 unknown の label 数 =", (df["act_id"]==-1).sum())

# -------------- ③ 各種ユーティリティ ---------------------------------------
os.makedirs(RESULT_DIR, exist_ok=True)
METRIC_PATH = f"{RESULT_DIR}/metrics.txt"; open(METRIC_PATH,"w").close()

def get_action_embeddings(vecs, df_sub, model):
    outs, labels, paths = [], [], []
    with torch.no_grad():
        for path, act_id in zip(df_sub["video_path"], df_sub["act_id"]):
            if path not in vecs: continue
            z = torch.tensor(vecs[path]).unsqueeze(0).float().cuda()
            a_vec,_ = model(z)
            outs.append(a_vec.squeeze(0).cpu())
            labels.append(act_id); paths.append(path)
    return torch.stack(outs), labels, paths

def clustering_report(X, y, tag):
    k = len(set([i for i in y if i!=-1]))   # unknown (-1) はクラスタ数に数えない
    pred = KMeans(k, random_state=0).fit_predict(X)
    ari = adjusted_rand_score(y, pred); nmi = normalized_mutual_info_score(y, pred)
    with open(METRIC_PATH,"a") as f: f.write(f"{tag}: ARI={ari:.4f}, NMI={nmi:.4f}\n")
    print(f"📊 {tag}:  ARI={ari:.4f} | NMI={nmi:.4f}")

def plot_embed(Z2d, y, out_png, title):
    fig, ax = plt.subplots(figsize=(8,6)); cmap = plt.get_cmap("tab20")
    unk_mask = np.array(y)==-1

    # 既知 action
    for act_id in sorted(set([i for i in y if i!=-1])):
        m = np.array(y)==act_id
        ax.scatter(Z2d[m,0],Z2d[m,1],s=6,
                   color=cmap(act_id%20), label=le_train.inverse_transform([act_id])[0])
    # 未知 action
    if unk_mask.any():
        ax.scatter(Z2d[unk_mask,0], Z2d[unk_mask,1], s=30, c="black",
                   marker="^", label="UNKNOWN")

    ax.set_title(title)
    ax.legend(markerscale=2,bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout(); fig.savefig(out_png); plt.close()

# -------------- ④ メインループ ---------------------------------------------
for mode in MODES:
  for loss in LOSSES:
    for suf in SUFFIXES:
        tag = f"{mode}_{loss}_{suf}"
        print(f"\n=== {tag} ===")

        # ---- ベクトル読み込み (train+test) -----------------
        vecs = {}
        for fp in [f"exec/vectors_{mode}.json",
                   f"exec/vectors_{mode}_test.json",
                   (f"exec/vectors_{mode}_ucf_labels.json" if USE_UCF else "")]:
            if fp and os.path.exists(fp):
                with open(fp) as f: vecs.update(json.load(f))

        # ---- ネットワーク -------------------------------
        net = (DisentangleEmbedOnlyMLP() if "nonlinear" in suf else DisentangleEmbedOnlySimple()).cuda().eval()
        net.load_state_dict(torch.load(f"model/disentangled_{tag}.pth"), strict=False)

        df_sub                = df[df["video_path"].isin(vecs)]
        X, y, _paths          = get_action_embeddings(vecs, df_sub, net)
        clustering_report(X, y, tag)

        for meth in VISUALIZE:
            Z2 = TSNE(2,random_state=0).fit_transform(X) if meth=="tsne" else UMAP().fit_transform(X)
            out_png = f"{RESULT_DIR}/{meth}/{tag}_{meth}.png"
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            plot_embed(Z2, y, out_png, f"{meth.upper()}  –  {tag}")

print("\n✅ 完了: Unknown ラベルは ^ (黒) でプロットされます")
