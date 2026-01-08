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
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),      
            nn.ReLU(),
            nn.Dropout(p_drop),               
            nn.Linear(hidden_dim, feature_dim, bias=True),
            nn.LayerNorm(feature_dim)   
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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# -----------------------------
# 2. Adversarial Discriminator Setup
# -----------------------------
class ActionMLPNet(nn.Module):
    def __init__(self, input_dim=768, feature_dim=256, hidden_dim=512, p_drop=0.3):
        super().__init__()
        self.act_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),      
            nn.ReLU(),
            nn.Dropout(p_drop),               
            nn.Linear(hidden_dim, feature_dim, bias=True),
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
                nn.init.xavier_uniform_(m.weight)
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
    def __init__(self, d_x3d=2048, d_vmae=768, d_hidden=512, p_drop=0.1):
        super().__init__()
        self.x3d_fc = nn.Linear(d_x3d, d_hidden)
        self.vmae_fc = nn.Linear(d_vmae, d_hidden)
        self.x3d_ln = nn.LayerNorm(d_hidden)
        self.vmae_ln = nn.LayerNorm(d_hidden)
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.LayerNorm(d_hidden), 
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(p_drop)
        self._init_weights()

    def forward(self, x3d, vmae):
        if x3d.dim() >= 2:
            x3d = F.layer_norm(x3d, x3d.shape[1:])
        if vmae.dim() >= 2:
            vmae = F.layer_norm(vmae, vmae.shape[1:])
        x3d_proj = self.dropout(torch.relu(self.x3d_ln(self.x3d_fc(x3d))))
        vmae_proj = self.dropout(torch.relu(self.vmae_ln(self.vmae_fc(vmae))))
        concat = torch.cat([x3d_proj, vmae_proj], dim=-1)
        alpha = self.gate(concat)
        fused = alpha * x3d_proj + (1 - alpha) * vmae_proj
        return fused, alpha

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # ゲート最後の層のバイアスを0に（開始時に中立的なalphaを促進）
        if isinstance(self.gate[0], nn.Linear):
            # gate: Linear(d_hidden*2 -> d_hidden) -> LayerNorm -> Sigmoid
            if self.gate[0].bias is not None:
                nn.init.zeros_(self.gate[0].bias)



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
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # hidden_dim <= 0 のときは Linear 1層
        if hidden_dim <= 0:
            self.net = nn.Linear(in_dim, num_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D) 行動埋め込み
        Returns:
            logits: (B, num_actions)
        """
        return self.net(x)
