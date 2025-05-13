import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# --- モデル定義 ---
D, H = 768, 256
class DisentangleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp_embed     = nn.Linear(D, H, bias=False)
        self.act_embed    = nn.Linear(D, H, bias=False)

    def forward(self, z):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        return a_vec, s_vec

# --- モデル読み込み ---
net = DisentangleNet().cuda()
net.load_state_dict(torch.load("disentangled_triplet.pth"))
net.eval()

# --- データ読み込み ---
vecs = json.load(open('video_vectors.json'))
df = pd.read_csv('filtered_labels.csv')
df['video_path'] = df['video_path'].str.replace('\\', '/')
df = df[df['video_path'].apply(lambda p: p in vecs)]

le_act = LabelEncoder().fit(df['action'])
le_sp  = LabelEncoder().fit(df['species'])
df['act_id'] = le_act.transform(df['action'])
df['sp_id']  = le_sp.transform(df['species'])

# --- Dataset / DataLoader ---
class VecDataset(Dataset):
    def __getitem__(self, idx):
        row = df.iloc[idx]
        x = torch.tensor(vecs[row['video_path']]).float()
        return x, row['act_id'], row['sp_id']
    def __len__(self): return len(df)

loader = DataLoader(VecDataset(), batch_size=64, shuffle=False)

# --- ベクトル抽出 ---
a_vecs, s_vecs, a_labels, s_labels = [], [], [], []
with torch.no_grad():
    for z, a, s in loader:
        z = z.cuda()
        a_vec, s_vec = net(z)
        a_vecs.append(a_vec.cpu())
        s_vecs.append(s_vec.cpu())
        a_labels += a.tolist()
        s_labels += s.tolist()

a_vecs = torch.cat(a_vecs).numpy()
s_vecs = torch.cat(s_vecs).numpy()

# --- t-SNE ---
a_proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(a_vecs)
s_proj = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(s_vecs)

# --- プロット ---
plt.figure(figsize=(14, 6))

# 行動ベクトル（a_vec）
plt.subplot(1, 2, 1)
plt.title("Action Embedding")
unique_a = np.unique(a_labels)
cmap = plt.get_cmap('tab20')
colors = [cmap(i % 20) for i in range(len(unique_a))]
for i, label in enumerate(unique_a):
    idx = np.array(a_labels) == label
    label_name = le_act.inverse_transform([label])[0]
    plt.scatter(a_proj[idx, 0], a_proj[idx, 1], color=colors[i], s=5, label=label_name)
plt.legend(fontsize=6, markerscale=3, loc='best')

# 種ベクトル（s_vec）
plt.subplot(1, 2, 2)
plt.title("Species Embedding")
unique_s = np.unique(s_labels)
colors = [cmap(i % 20) for i in range(len(unique_s))]
for i, label in enumerate(unique_s):
    idx = np.array(s_labels) == label
    label_name = le_sp.inverse_transform([label])[0]
    plt.scatter(s_proj[idx, 0], s_proj[idx, 1], color=colors[i], s=5, label=label_name)
plt.legend(fontsize=6, markerscale=3, loc='best')

plt.tight_layout()
plt.savefig("result/embedding_tsne_with_legend.png", bbox_inches='tight')
plt.show()
