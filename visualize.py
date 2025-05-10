# visualize.py
import torch, json, pandas as pd, umap.umap_ as umap, matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

net = DisentangleNet().cuda(); net.load_state_dict(torch.load('disentangled.pth'))

df   = pd.read_csv('labels.csv')
vecs = json.load(open('video_vectors.json'))
acts = LabelEncoder().fit_transform(df['action'])

a_vecs = []
for path in df['video_path']:
    z = torch.tensor(vecs[os.path.join('data_root', path)]).cuda()
    with torch.no_grad():
        _,_, a_vec,_,_ = net(z)
    a_vecs.append(a_vec.cpu().numpy())

emb2d = umap.UMAP().fit_transform(a_vecs)
plt.figure(figsize=(8,7))
scatter = plt.scatter(emb2d[:,0], emb2d[:,1], c=acts, cmap='tab20', s=8)
plt.legend(handles=scatter.legend_elements()[0], labels=list(set(df['action'])), fontsize=8, bbox_to_anchor=(1.05,1))
plt.title("Action‑only latent space (species‑invariant)")
plt.tight_layout(); plt.savefig('action_tsne.png'); plt.show()
