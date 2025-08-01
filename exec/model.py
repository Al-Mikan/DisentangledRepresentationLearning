# model.py

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
# 学習用モデル
# -----------------------------
class DisentangleNetSimple(nn.Module):
    """単層線形ヘッド"""
    def __init__(self, D=768, H=256, A=10, S=10):
        super().__init__()
        self.act_embed = nn.Linear(D, H, bias=False)
        self.sp_embed = nn.Linear(D, H, bias=False)
        self.action_disc = nn.Linear(H, A)
        self.species_disc = nn.Linear(H, S)

    def forward(self, z, grl_lambda=1.0):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        s_pred_from_a = self.species_disc(grad_reverse(a_vec, grl_lambda))
        a_pred_from_s = self.action_disc(grad_reverse(s_vec, grl_lambda))
        return a_vec, s_vec, s_pred_from_a, a_pred_from_s

class DisentangleNetMLP(nn.Module):
    """MLPヘッド版"""
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

class GatedFusion(nn.Module):
    def __init__(self, d_x3d, d_vmae, d_hidden):
        super().__init__()
        self.x3d_fc = nn.Linear(d_x3d, d_hidden)
        self.vmae_fc = nn.Linear(d_vmae, d_hidden)
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Sigmoid()
        )

    def forward(self, x3d, vmae):
        x3d_proj = self.x3d_fc(x3d)
        vmae_proj = self.vmae_fc(vmae)
        concat = torch.cat([x3d_proj, vmae_proj], dim=-1)
        alpha = self.gate(concat)
        fused = alpha * x3d_proj + (1 - alpha) * vmae_proj
        return fused, alpha
# -----------------------------
# 推論用モデル
# -----------------------------
class DisentangleEmbedOnlySimple(nn.Module):
    """推論用: 単層線形"""
    def __init__(self, D=768, H=256):
        super().__init__()
        self.act_embed = nn.Linear(D, H, bias=False)
        self.sp_embed = nn.Linear(D, H, bias=False)

    def forward(self, z):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        return a_vec, s_vec

class DisentangleEmbedOnlyMLP(nn.Module):
    """推論用: MLP版"""
    def __init__(self, D=768, H=256, hidden=512):
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

    def forward(self, z):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        return a_vec, s_vec

# -----------------------------
# データロード系
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

def create_dataloader(df, vecs, batch_size=64, shuffle=True):
    dataset = VecDataset(df, vecs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

