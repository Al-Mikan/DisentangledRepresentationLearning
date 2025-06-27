# disentangle_model.py

import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# === GRL ===
from torch.autograd import Function

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None

def grad_reverse(x, lambda_grl=1.0):
    return GradientReverseLayer.apply(x, lambda_grl)

# -----------------------------
# モデル定義
# -----------------------------
class DisentangleNet(nn.Module):
    def __init__(self, D=768, H=256, A=10, S=10):
        super().__init__()
        self.act_embed = nn.Linear(D, H, bias=False)
        self.sp_embed = nn.Linear(D, H, bias=False)
        self.action_disc = nn.Linear(H, A)  # 種から行動を当てる Discriminator
        self.species_disc = nn.Linear(H, S)  # 行動から種を当てる Discriminator

    def forward(self, z, grl_lambda=1.0):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)

        # GRL を通す
        s_pred_from_a = self.species_disc(grad_reverse(a_vec, grl_lambda))
        a_pred_from_s = self.action_disc(grad_reverse(s_vec, grl_lambda))

        return a_vec, s_vec, s_pred_from_a, a_pred_from_s
    
class DisentangleNetNonlinear(nn.Module):
    """
    🔥 非線形ヘッド版
    act_embed と sp_embed に2層MLPを使う
    """
    def __init__(self, D=768, H=256, A=10, S=10, hidden=512):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, H)
        )
        self.sp_embed = nn.Sequential(
            nn.Linear(D, hidden),
            nn.ReLU(),
            nn.Linear(hidden, H)
        )
        self.action_disc = nn.Linear(H, A)
        self.species_disc = nn.Linear(H, S)

    def forward(self, z, grl_lambda=1.0):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        s_pred_from_a = self.species_disc(grad_reverse(a_vec, grl_lambda))
        a_pred_from_s = self.action_disc(grad_reverse(s_vec, grl_lambda))
        return a_vec, s_vec, s_pred_from_a, a_pred_from_s
    

class DisentangleNet2(nn.Module):
    def __init__(self, A, S, D=768, H=256):
        super().__init__()
        self.act_embed = nn.Linear(D, H, bias=False)
        self.sp_embed = nn.Linear(D, H, bias=False)
        self.act_classifier = nn.Linear(H, A)
        self.sp_classifier = nn.Linear(H, S)
        self.action_disc = nn.Linear(H, A)
        self.species_disc = nn.Linear(H, S)

    def forward(self, z, grl_lambda=1.0):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        a_logits = self.act_classifier(a_vec)
        s_logits = self.sp_classifier(s_vec)

        # GRL で逆学習
        s_pred_from_a = self.species_disc(grad_reverse(a_vec, grl_lambda))
        a_pred_from_s = self.action_disc(grad_reverse(s_vec, grl_lambda))

        return a_vec, s_vec, a_logits, s_logits, s_pred_from_a, a_pred_from_s

# -----------------------------
# データ読み込み・前処理関数
# -----------------------------
def load_data(label_path='labels.csv', vector_path='exec/video_vectors.json'):
    with open(vector_path) as f:
        vecs = json.load(f)

    df = pd.read_csv(label_path)
    df['video_path'] = df['video_path'].str.replace('\\', '/')
    df = df[df['video_path'].apply(lambda p: p in vecs)]

    le_act = LabelEncoder().fit(df['action'])
    le_sp = LabelEncoder().fit(df['species'])
    df['act_id'] = le_act.transform(df['action'])
    df['sp_id'] = le_sp.transform(df['species'])

    print(f"✅ 使用可能なサンプル数: {len(df)} 件")
    return df, vecs, le_act, le_sp

# -----------------------------
# Dataset 定義
# -----------------------------
class VecDataset(Dataset):
    def __init__(self, df, vecs):
        self.df = df
        self.vecs = vecs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor(self.vecs[row['video_path']]).float()
        return x, row['act_id'], row['sp_id']

    def __len__(self):
        return len(self.df)

# -----------------------------
# DataLoader 作成関数
# -----------------------------
def create_dataloader(df, vecs, batch_size=64, shuffle=True):
    dataset = VecDataset(df, vecs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
