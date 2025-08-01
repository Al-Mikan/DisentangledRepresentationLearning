# model.py

import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
# -----------------------------
# 1. Action Embedding Models
# -----------------------------
class SimpleLinearNet(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.action_head = nn.Linear(input_dim, feature_dim, bias=False)

    def forward(self, x):
        return self.action_head(x)

class SimpleMLPNet(nn.Module):
    def __init__(self, input_dim=768, feature_dim=256, hidden_dim=512):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        return self.act_embed(x)

# -----------------------------
# 2. Adversarial Discriminator Setup
# -----------------------------
class ActionLinearNet(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim, bias=False)

    def forward(self, x):
        return self.encoder(x)
    
class ActionMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class SpeciesDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_species):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(),
            nn.Linear(128, num_species)
        )

    def forward(self, feat):
        return self.classifier(feat)
# -----------------------------
# 3. 推論用モデル（a_vec, s_vec 両方を抽出）
# -----------------------------
class DisentangleEmbedLinear(nn.Module):
    def __init__(self, D=768, H=256):
        super().__init__()
        self.act_embed = nn.Linear(D, H, bias=False)

    def forward(self, z):
        a_vec = self.act_embed(z)
        return a_vec

class DisentangleEmbedMLP(nn.Module):
    def __init__(self, D=768, H=256, hidden=512):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(D, hidden), nn.ReLU(),
            nn.Linear(hidden, H)
        )
    def forward(self, z):
        a_vec = self.act_embed(z)
        return a_vec

# -----------------------------
# 4. Gated Fusion
# -----------------------------
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
