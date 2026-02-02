# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# -----------------------------
# 1. Action Embedding Models
# -----------------------------

class SimpleMLPNet(nn.Module):
    def __init__(self, input_dim=768, feature_dim=256, hidden_dim=512, p_drop=0.3):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),      
            nn.ReLU(),
            nn.Dropout(p_drop),               
            nn.Linear(hidden_dim, feature_dim, bias=False),
        )
        self._init_weights()

    def forward(self, x):
        x = self.act_embed(x)
        x = F.normalize(x, p=2, dim=1)
        return x
        # return F.normalize(x, dim=-1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

# -----------------------------
# 2. Adversarial Discriminator Setup
# -----------------------------
class ActionMLPNet(nn.Module):
    def __init__(self, input_dim=512, feature_dim=256, hidden_dim=512, p_drop=0.2):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LeakyReLU(0.1),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, feature_dim, bias=True),
        )
        self._init_weights()

    def forward(self, x):
        x = self.act_embed(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def _init_weights(self):
        # 最後の層（feature_dimを出力する層）を特定するためにループ
        for i, m in enumerate(self.act_embed):
            if isinstance(m, nn.Linear):
                # 最後のLinear層かどうか判定
                is_last = (i == len(self.act_embed) - 1)
                
                if is_last:
                    # 埋め込み層は直交初期化で散らす
                    nn.init.orthogonal_(m.weight)
                else:
                    # 中間層はKaiming初期化
                    nn.init.kaiming_uniform_(
                        m.weight, 
                        mode='fan_in', 
                        nonlinearity='leaky_relu', 
                        a=0.1
                    )
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class SpeciesDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_species):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_species)
        )
        self._init_weights()

    def forward(self, feat):
        return self.classifier(feat)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# -----------------------------
# 4. Gated Fusion
# -----------------------------
class GatedFusion(nn.Module):
    def __init__(self, d_x3d=2048, d_vmae=768, d_hidden=512):
        super().__init__()
        
        # 入力正規化 (Pre-Norm)
        self.in_ln_x3d  = nn.LayerNorm(d_x3d)
        self.in_ln_vmae = nn.LayerNorm(d_vmae)
        
        self.x3d_fc  = nn.Linear(d_x3d, d_hidden, bias=False)
        self.vmae_fc = nn.Linear(d_vmae, d_hidden, bias=False)
        # self.x3d_ln  = nn.LayerNorm(d_hidden)
        # self.vmae_ln = nn.LayerNorm(d_hidden)
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden, bias=True),
            nn.Sigmoid()
        )
        self._init_weights()

    def forward(self, x3d, vmae):
        # 入力を正規化
        x3d  = self.in_ln_x3d(x3d)
        vmae = self.in_ln_vmae(vmae)
        
        x3d_proj  = self.x3d_fc(x3d)
        vmae_proj = self.vmae_fc(vmae)

        concat = torch.cat([x3d_proj, vmae_proj], dim=-1)
        alpha = self.gate(concat)
        fused = alpha * x3d_proj + (1 - alpha) * vmae_proj

        return fused, alpha

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ConcatFusion(nn.Module):
    """
    単純な Concat Fusion（ベースライン）。
    2つの特徴を連結して Linear で射影するだけ。
    GatedFusion との比較用。
    """
    def __init__(self, d_x3d=2048, d_vmae=768, d_hidden=512):
        super().__init__()
        
        # 入力正規化 (Pre-Norm)
        self.in_ln_x3d  = nn.LayerNorm(d_x3d)
        self.in_ln_vmae = nn.LayerNorm(d_vmae)
        
        # 連結後に射影
        self.fc = nn.Linear(d_x3d + d_vmae, d_hidden, bias=False)
        self._init_weights()

    def forward(self, x3d, vmae):
        # 入力を正規化
        x3d  = self.in_ln_x3d(x3d)
        vmae = self.in_ln_vmae(vmae)
        
        # 単純に連結して射影
        concat = torch.cat([x3d, vmae], dim=-1)
        fused = self.fc(concat)

        # alpha は None を返す（GatedFusion との互換性のため）
        return fused, None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ActionClassifier(nn.Module):
    """
    行動分類用のシンプルな classifier
    - 入力: 行動埋め込みベクトル (B, D)
    - 出力: 行動クラス logits (B, num_actions)
    """

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
    ):
        super().__init__()

        self.net = nn.Linear(in_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) 行動埋め込み
        Returns:
            logits: (B, num_actions)
        """
        return self.net(x)
