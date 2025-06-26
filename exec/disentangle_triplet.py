import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from model import load_data, create_dataloader,DisentangleNet, DisentangleNet2
import argparse
from improved_triplet_loss import ImprovedTripletLoss


# 行動の損失の重要度を上げ、種の損失の重要度を下げる
lambda_action = 2.0  # 例えば2倍にする
lambda_species = 0.5 # 例えば半分にする
lambda_ortho = 0.05

def train_one(loss_type, mode):
# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', type=str, required=True, choices=['simple', 'adaptive3d', 'adaptive1d'])
# parser.add_argument('--loss', type=str, default='triplet', choices=['triplet', 'improved', 'cross'], help="Loss function type")
# args = parser.parse_args()

# mode = args.mode
# loss_type = args.loss

    # --- ハイパーパラメータ設定 ---
    n_epochs = 1000
    patience = 50

    # --- データ読み込み ---
    output_path = {
        'simple': './exec/vectors_simple.json',
        'adaptive3d': './exec/vectors_adaptive3d.json',
        'adaptive1d': './exec/vectors_adaptive1d.json',
        'sliding': './exec/vectors_sliding.json',
    }[mode]

    df, vecs, le_act, le_sp = load_data('labels.csv', output_path)
    A, S = len(le_act.classes_), len(le_sp.classes_)
    loader = create_dataloader(df, vecs, batch_size=64, shuffle=True)

    # --- モデル・最適化関数の準備 ---
    if loss_type == 'cross':
        net = DisentangleNet2(D=768, H=256, A=A, S=S).cuda()
    else:
        net = DisentangleNet(D=768, H=256).cuda()

    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    if loss_type == 'triplet':
        triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    elif loss_type == 'improved':
        triplet_loss_fn = ImprovedTripletLoss(tau1=1.0, tau2=0.5, beta=0.5)

    ce_act = nn.CrossEntropyLoss()
    ce_sp = nn.CrossEntropyLoss()
    ortho = lambda u, v: ((u * v).sum(dim=1) ** 2).mean()

    # --- Tripletペア作成 ---
    def make_triplets(vectors, labels):
        anchors, positives, negatives = [], [], []
        labels = labels.cpu().numpy()
        for i in range(len(vectors)):
            anchor = vectors[i]
            label = labels[i]
            pos_idx = np.where(labels == label)[0]
            neg_idx = np.where(labels != label)[0]
            pos_idx = [j for j in pos_idx if j != i]
            if not pos_idx or not len(neg_idx): continue
            j = np.random.choice(pos_idx)
            k = np.random.choice(neg_idx)
            anchors.append(anchor)
            positives.append(vectors[j])
            negatives.append(vectors[k])
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
    
    # --- ハードマイニングのTripletペア作成関数 ---
    def make_triplets_hard_mining(vectors, labels):
        # (現在のmake_triplets関数をコピーして改造)
        anchors, positives, negatives = [], [], []
        
        with torch.no_grad(): # 勾配計算は不要
            # 全ペア間の距離を計算
            all_dists = torch.cdist(vectors, vectors, p=2)

        for i in range(len(vectors)):
            anchor = vectors[i]
            label = labels[i]
            
            # ポジティブサンプルのインデックス
            pos_indices = torch.where(labels == label)[0]
            pos_indices = pos_indices[pos_indices != i] # 自分自身を除く
            
            # ネガティブサンプルのインデックス
            neg_indices = torch.where(labels != label)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue
                
            # ★★★ ハードポジティブを選ぶ ★★★
            # アンカーから最も「遠い」ポジティブサンプル
            hardest_positive_idx = pos_indices[torch.argmax(all_dists[i, pos_indices])]
            
            # ★★★ ハードネガティブを選ぶ ★★★
            # アンカーから最も「近い」ネガティブサンプル
            hardest_negative_idx = neg_indices[torch.argmin(all_dists[i, neg_indices])]
            
            anchors.append(anchor)
            positives.append(vectors[hardest_positive_idx])
            negatives.append(vectors[hardest_negative_idx])
            
        if not anchors:
            return None, None, None
            
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    # --- ログ初期化 ---
    best_loss = float('inf')
    no_improve_count = 0
    log = {
        'triplet_action': [], 'triplet_species': [],
        'ce_action': [], 'ce_species': [],
        'ortho': [], 'total': []
    }

    # --- 学習ループ ---
    for epoch in range(n_epochs):
        ep_trip_a, ep_trip_s, ep_ce_a, ep_ce_s, ep_ortho, ep_total = 0, 0, 0, 0, 0, 0
        steps = 0
        for z, a, s in loader:
            z, a, s = z.cuda(), a.cuda().long(), s.cuda().long()

            if loss_type == 'cross':
                a_vec, s_vec, a_logits, s_logits = net(z)
            else:
                a_vec, s_vec = net(z)
                a_logits, s_logits = None, None

            anc_a, pos_a, neg_a = make_triplets_hard_mining(a_vec, a)
            anc_s, pos_s, neg_s = make_triplets_hard_mining(s_vec, s)
            if len(anc_a) == 0 or len(anc_s) == 0: continue

            loss_trip_a = triplet_loss_fn(anc_a, pos_a, neg_a)
            loss_trip_s = triplet_loss_fn(anc_s, pos_s, neg_s)
            loss_ortho = ortho(a_vec, s_vec)

            loss_ce_a = ce_act(a_logits, a) if loss_type == 'cross' else 0
            loss_ce_s = ce_sp(s_logits, s) if loss_type == 'cross' else 0

            loss = (lambda_action * loss_trip_a) + \
           (lambda_species * loss_trip_s) + \
           (lambda_ortho * loss_ortho)
            if loss_type == 'cross':
                loss += 0.5 * loss_ce_a + 0.5 * loss_ce_s

            opt.zero_grad(); loss.backward(); opt.step()

            ep_trip_a += loss_trip_a.item()
            ep_trip_s += loss_trip_s.item()
            ep_ce_a += loss_ce_a.item() if loss_type == 'cross' else 0
            ep_ce_s += loss_ce_s.item() if loss_type == 'cross' else 0
            ep_ortho += loss_ortho.item()
            ep_total += loss.item()
            steps += 1

        avg_total = ep_total / steps
        log['triplet_action'].append(ep_trip_a / steps)
        log['triplet_species'].append(ep_trip_s / steps)
        log['ce_action'].append(ep_ce_a / steps if loss_type == 'cross' else 0)
        log['ce_species'].append(ep_ce_s / steps if loss_type == 'cross' else 0)
        log['ortho'].append(ep_ortho / steps)
        log['total'].append(avg_total)

        print(f"epoch {epoch:03d}: loss={avg_total:.3f}, trip_a={ep_trip_a/steps:.3f}, trip_s={ep_trip_s/steps:.3f}, ce_a={ep_ce_a/steps if loss_type == 'cross' else 0:.3f}, ce_s={ep_ce_s/steps if loss_type == 'cross' else 0:.3f}, ortho={ep_ortho/steps:.3f}")

        # --- Early Stopping ---
        if avg_total < best_loss:
            best_loss = avg_total
            no_improve_count = 0
            suffix = f"{mode}_{loss_type}"
            model_save_path = f"./model/disentangled_{suffix}.pth"
            torch.save(net.state_dict(),model_save_path)
            
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"🛑 {patience}エポック連続でloss改善がないため、学習を終了します。")
                break

    # --- 保存パスを条件で分岐 ---
    save_path = f"./loss/training_losses_{suffix}.png"

    # --- ロス可視化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(log['total'], label='Total Loss')
    plt.plot(log['triplet_action'], label='Triplet Loss (Action)')
    plt.plot(log['triplet_species'], label='Triplet Loss (Species)')
    if loss_type == 'cross':
        plt.plot(log['ce_action'], label='CrossEntropy (Action)')
        plt.plot(log['ce_species'], label='CrossEntropy (Species)')
    plt.plot(log['ortho'], label='Orthogonality')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    
            # ログファイルも一緒に保存
    os.makedirs("loss", exist_ok=True)
    with open("loss/final_losses_summary.txt", "a") as f:
        f.write(
            f"[{mode} | {loss_type}] epoch={epoch:03d}, "
            f"total={avg_total:.4f}, trip_a={ep_trip_a/steps:.4f}, "
            f"trip_s={ep_trip_s/steps:.4f}, "
            f"ce_a={(ep_ce_a/steps if loss_type == 'cross' else 0):.4f}, "
            f"ce_s={(ep_ce_s/steps if loss_type == 'cross' else 0):.4f}, "
            f"ortho={ep_ortho/steps:.4f}\n")
        



def main():
    all_loss_types = ['triplet', 'improved']
    all_modes = ['simple', 'adaptive3d', 'adaptive1d', 'sliding']

    for mode in all_modes:
        for loss_type in all_loss_types:
            print(f"\n🚀 Start training: mode={mode}, loss={loss_type}")
            train_one(loss_type=loss_type, mode=mode)

if __name__ == "__main__":
    main()