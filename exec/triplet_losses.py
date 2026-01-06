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

class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5, epsilon=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.epsilon = epsilon # マイニング用のマージン

    def forward(self, embeddings, labels):
        # L2正規化（これは必須！OKです）
        embeddings = F.normalize(embeddings, dim=1)
        sim_mat = embeddings @ embeddings.t()

        labels = labels.view(-1, 1)
        eye = torch.eye(len(labels), device=labels.device).bool()

        # マスク作成（ここも完璧です）
        is_pos = (labels == labels.t()) & ~eye
        is_neg = (labels != labels.t()) & ~eye

        loss = []

        for i in range(len(labels)):
            pos_sim = sim_mat[i][is_pos[i]]
            neg_sim = sim_mat[i][is_neg[i]]

            if len(pos_sim) == 0 or len(neg_sim) == 0:
                continue

            # ---- hard mining ----
            neg_max = neg_sim.max()
            pos_min = pos_sim.min()

            hard_pos_sim = pos_sim[pos_sim < neg_max + self.epsilon]
            
            hard_neg_sim = neg_sim[neg_sim > pos_min - self.epsilon]

            if len(hard_pos_sim) == 0 or len(hard_neg_sim) == 0:
                continue

            pos_loss = torch.log(
                1 + torch.sum(torch.exp(-self.alpha * (hard_pos_sim - self.base)))
            ) / self.alpha

            neg_loss = torch.log(
                1 + torch.sum(torch.exp(self.beta * (hard_neg_sim - self.base)))
            ) / self.beta

            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], device=embeddings.device, requires_grad=True)

        return torch.mean(torch.stack(loss))

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
