# disentangle_model.py

import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# モデル定義
# -----------------------------
class DisentangleNet(nn.Module):
    def __init__(self, D=768, H=256):
        super().__init__()
        self.sp_embed = nn.Linear(D, H, bias=False)
        self.act_embed = nn.Linear(D, H, bias=False)

    def forward(self, z):
        return self.act_embed(z), self.sp_embed(z)

class DisentangleNet2(nn.Module):
    def __init__(self, A, S, D=768, H=256):
        super().__init__()
        self.sp_embed = nn.Linear(D, H, bias=False)
        self.act_embed = nn.Linear(D, H, bias=False)
        self.act_classifier = nn.Linear(H, A)
        self.sp_classifier = nn.Linear(H, S)

    def forward(self, z):
        a_vec = self.act_embed(z)
        s_vec = self.sp_embed(z)
        a_logits = self.act_classifier(a_vec)
        s_logits = self.sp_classifier(s_vec)
        return a_vec, s_vec, a_logits, s_logits

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
