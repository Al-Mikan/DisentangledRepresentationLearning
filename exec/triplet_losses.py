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



class CosineTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        # Cosine similarity: Higher is better (similar)
        sim_pos = F.cosine_similarity(anchor, positive, dim=-1)  # → 1 に近いほど正解
        sim_neg = F.cosine_similarity(anchor, negative, dim=-1)  # → -1 に近いほど間違い

        # Loss: max(0, margin + sim_neg - sim_pos)
        loss = F.relu(self.margin + sim_neg - sim_pos)
        return loss.mean()