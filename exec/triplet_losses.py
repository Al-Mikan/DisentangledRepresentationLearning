import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedTripletLoss(nn.Module):
    def __init__(self, tau1=0.3, tau2=0.5, beta=0.5, lambda_norm=0.01):
        super().__init__()
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta
        self.lambda_norm = lambda_norm

    def forward(self, embeddings, labels, triplets):
        if triplets is None or len(triplets[0]) == 0:
            return torch.zeros((), device=embeddings.device)

        a_idx, p_idx, n_idx = triplets
        a = embeddings[a_idx]
        p = embeddings[p_idx]
        n = embeddings[n_idx]

        d_ap = F.pairwise_distance(a, p)
        d_an = F.pairwise_distance(a, n)

        inter = torch.relu(d_ap - d_an + self.tau1)
        intra = torch.relu(d_ap - self.tau2)

        norm = torch.norm(embeddings, dim=1)
        norm_loss = ((norm - 1.0) ** 2).mean()

        return (
            inter.mean()
            + self.beta * intra.mean()
            + self.lambda_norm * norm_loss
        )



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
