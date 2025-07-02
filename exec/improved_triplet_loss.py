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
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        d_ap = F.pairwise_distance(anchor, positive, p=2)
        d_an = F.pairwise_distance(anchor, negative, p=2)

        inter = torch.clamp(d_ap - d_an + self.tau1, min=0.0)
        intra = torch.clamp(d_ap - self.tau2, min=0.0)

        loss = inter.mean() + self.beta * intra.mean()
        return loss
