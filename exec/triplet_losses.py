import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTripletLoss(nn.Module):
    def __init__(self, tau1=0.3, tau2=0.5, beta=0.1):
        super().__init__()
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta

    def forward(self, embeddings, labels, triplets):
        # triplets が tuple (a, p, n) であることを想定
        if triplets is None or len(triplets[0]) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        a_idx, p_idx, n_idx = triplets
        anchor   = F.normalize(embeddings[a_idx], dim=-1)
        positive = F.normalize(embeddings[p_idx], dim=-1)
        negative = F.normalize(embeddings[n_idx], dim=-1)

        d_ap = F.pairwise_distance(anchor, positive)
        d_an = F.pairwise_distance(anchor, negative)

        # tau1: Margin, tau2: Intra-class compactness
        inter_loss = torch.relu(d_ap - d_an + self.tau1)
        intra_loss = torch.relu(d_ap - self.tau2).pow(2) 

        return inter_loss.mean() + self.beta * intra_loss.mean()



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
    


# === Gradient Reversal Layer (GRL) ===
from torch.autograd import Function

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grl(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)
