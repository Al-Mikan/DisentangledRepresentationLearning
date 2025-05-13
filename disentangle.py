# disentangle.py
import torch, torch.nn as nn, pandas as pd, json, os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

vecs = json.load(open('video_vectors.json'))
df   = pd.read_csv('filtered_labels.csv')

df['video_path'] = df['video_path'].str.replace('\\', '/')

# JSONに存在する動画パスだけにフィルタ
df = df[df['video_path'].apply(lambda p: p in vecs)]

print(f"✅ 使用可能なサンプル数: {len(df)} 件")

# ラベルエンコード
le_act  = LabelEncoder().fit(df['action'])
le_sp   = LabelEncoder().fit(df['species'])
df['act_id'] = le_act.transform(df['action'])
df['sp_id']  = le_sp.transform(df['species'])

class VecDataset(Dataset):
    def __getitem__(self, idx):
        row = df.iloc[idx]
        x = torch.tensor(vecs[row['video_path']]).float()
        return x, row['act_id'], row['sp_id']
    def __len__(self): return len(df)

loader = DataLoader(VecDataset(), batch_size=64, shuffle=True)

D = 768              # VideoMAE ベクトル次元
H = 256              # 隠れ層
A = len(le_act.classes_)
S = len(le_sp.classes_)

class DisentangleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()  # すでに抽出済みなので恒等
        self.action_head  = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, A))
        self.species_head = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, S))
        # 行動・種ベクトルを得るための中間層
        self.act_embed    = nn.Linear(D, H, bias=False)
        self.sp_embed     = nn.Linear(D, H, bias=False)
    def forward(self, z):
        a_logits = self.action_head(z)
        s_logits = self.species_head(z)
        a_vec = self.act_embed(z)   # H-d 行動表現
        s_vec = self.sp_embed(z)    # H-d 種表現
        res   = z - self.act_embed.weight.T @ a_vec.T - self.sp_embed.weight.T @ s_vec.T
        return a_logits, s_logits, a_vec, s_vec, res.T

net = DisentangleNet().cuda()
opt = torch.optim.Adam(net.parameters(), 1e-4)
ce  = nn.CrossEntropyLoss()
ortho = lambda u,v: ((u*v).sum(dim=1)**2).mean()   # 直交正則化

for epoch in range(20):
    for z, a, s in loader:
        z,a,s = z.cuda(), a.cuda(), s.cuda()
        a_log, s_log, a_vec, s_vec, _ = net(z)
        loss = ce(a_log, a) + ce(s_log, s) + 0.1*ortho(a_vec, s_vec)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f'epoch {epoch}: loss={loss.item():.3f}')
torch.save(net.state_dict(), 'disentangled.pth')
