import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTripletLoss(nn.Module):
    def __init__(self, tau1=1.0, tau2=0.5, beta=0.5):
        super().__init__()
        self.tau1 = tau1  # inter-class margin (通常は正)
        self.tau2 = tau2  # intra-class margin (通常は小さめ)
        self.beta = beta  # 重み係数

    def forward(self, anchor, positive, negative):
        # L2距離
        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)

        # inter-class constraint: d_ap - d_an + margin < 0 が理想
        inter = torch.clamp(d_ap - d_an + self.tau1, min=0.0)

        # intra-class constraint: d_ap < tau2 が理想
        intra = torch.clamp(d_ap - self.tau2, min=0.0)

        loss = inter.mean() + self.beta * intra.mean()
        return loss
