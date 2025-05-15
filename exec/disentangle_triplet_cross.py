import torch, torch.nn as nn, pandas as pd, json
import os, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. データ読み込み
vecs = json.load(open('exec/video_vectors.json'))
df = pd.read_csv('labels.csv')
df['video_path'] = df['video_path'].str.replace('\\', '/')
df = df[df['video_path'].apply(lambda p: p in vecs)]
print(f"✅ 使用可能なサンプル数: {len(df)} 件")

le_act = LabelEncoder().fit(df['action'])
le_sp  = LabelEncoder().fit(df['species'])
df['act_id'] = le_act.transform(df['action'])
df['sp_id']  = le_sp.transform(df['species'])

# 2. データセット定義
class VecDataset(Dataset):
    def __getitem__(self, idx):
        row = df.iloc[idx]
        x = torch.tensor(vecs[row['video_path']]).float()
        return x, row['act_id'], row['sp_id']
    def __len__(self): return len(df)

loader = DataLoader(VecDataset(), batch_size=128, shuffle=True)

D, H = 768, 256
A = len(le_act.classes_)
S = len(le_sp.classes_)

# 3. モデル定義
class DisentangleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sp_embed     = nn.Linear(D, H, bias=False)
        self.act_embed    = nn.Linear(D, H, bias=False)
        self.act_classifier = nn.Linear(H, A)
        self.sp_classifier  = nn.Linear(H, S)

    def forward(self, z):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        a_logits = self.act_classifier(a_vec)
        s_logits = self.sp_classifier(s_vec)
        return a_vec, s_vec, a_logits, s_logits

net = DisentangleNet().cuda()
opt = torch.optim.Adam(net.parameters(), 1e-4)
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
ce_act = nn.CrossEntropyLoss()
ce_sp  = nn.CrossEntropyLoss()
ortho = lambda u,v: ((u*v).sum(dim=1)**2).mean()

# 4. Tripletペア作成
def make_triplets(vectors, labels):
    anchors, positives, negatives = [], [], []
    labels = labels.cpu().numpy()
    for i in range(len(vectors)):
        anchor = vectors[i]
        label = labels[i]
        pos_idx = np.where(labels == label)[0]
        neg_idx = np.where(labels != label)[0]
        pos_idx = [j for j in pos_idx if j != i]
        if not pos_idx or not len(neg_idx): continue
        j = np.random.choice(pos_idx)
        k = np.random.choice(neg_idx)
        anchors.append(anchor)
        positives.append(vectors[j])
        negatives.append(vectors[k])
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

# 5. 学習ループとロス記録
n_epochs = 500
patience = 50
best_loss = float('inf')
no_improve_count = 0

log = {
    'triplet_action': [], 'triplet_species': [],
    'ce_action': [], 'ce_species': [],
    'ortho': [], 'total': []
}

for epoch in range(n_epochs):
    ep_trip_a, ep_trip_s, ep_ce_a, ep_ce_s, ep_ortho, ep_total = 0, 0, 0, 0, 0, 0
    steps = 0

    for z, a, s in loader:
        z, a, s = z.cuda(), a.cuda().long(), s.cuda().long()
        a_vec, s_vec, a_logits, s_logits = net(z)

        anc_a, pos_a, neg_a = make_triplets(a_vec, a)
        anc_s, pos_s, neg_s = make_triplets(s_vec, s)

        if len(anc_a) == 0 or len(anc_s) == 0: continue

        loss_trip_a = triplet_loss_fn(anc_a, pos_a, neg_a)
        loss_trip_s = triplet_loss_fn(anc_s, pos_s, neg_s)
        loss_ce_a   = ce_act(a_logits, a)
        loss_ce_s   = ce_sp(s_logits, s)
        loss_ortho  = ortho(a_vec, s_vec)

        # 重み付き合計損失
        loss = (
            loss_trip_a +
            loss_trip_s +
            0.5 * loss_ce_a +
            0.5 * loss_ce_s +
            0.1 * loss_ortho
        )

        opt.zero_grad(); loss.backward(); opt.step()

        ep_trip_a += loss_trip_a.item()
        ep_trip_s += loss_trip_s.item()
        ep_ce_a   += loss_ce_a.item()
        ep_ce_s   += loss_ce_s.item()
        ep_ortho  += loss_ortho.item()
        ep_total  += loss.item()
        steps += 1

    avg_total = ep_total / steps
    log['triplet_action'].append(ep_trip_a / steps)
    log['triplet_species'].append(ep_trip_s / steps)
    log['ce_action'].append(ep_ce_a / steps)
    log['ce_species'].append(ep_ce_s / steps)
    log['ortho'].append(ep_ortho / steps)
    log['total'].append(avg_total)

    print(f"epoch {epoch:03d}: loss={avg_total:.3f}, trip_a={ep_trip_a/steps:.3f}, trip_s={ep_trip_s/steps:.3f}, ce_a={ep_ce_a/steps:.3f}, ce_s={ep_ce_s/steps:.3f}, ortho={ep_ortho/steps:.3f}")

    # --- Early Stopping ---
    if avg_total < best_loss:
        best_loss = avg_total
        no_improve_count = 0
        torch.save(net.state_dict(), 'disentangled_triplet_cross.pth')
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"🛑 {patience}エポック連続でloss改善がないため、学習を終了します。")
            break

# 6. ロスの可視化
plt.figure(figsize=(10, 6))
plt.plot(log['total'], label='Total Loss')
plt.plot(log['triplet_action'], label='Triplet Loss (Action)')
plt.plot(log['triplet_species'], label='Triplet Loss (Species)')
plt.plot(log['ce_action'], label='CrossEntropy (Action)')
plt.plot(log['ce_species'], label='CrossEntropy (Species)')
plt.plot(log['ortho'], label='Orthogonality')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("./exec/training_cross_losses.png")
plt.show()
