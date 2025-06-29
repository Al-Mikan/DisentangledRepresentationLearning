import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from model import load_data, create_dataloader, DisentangleNetMLP, DisentangleNetSimple
from improved_triplet_loss import ImprovedTripletLoss
import requests

# ====== 定数 ======
LAMBDA_ACTION = 2.0
LAMBDA_SPECIES = 0.5
LAMBDA_ADV = 0.5
WEBHOOK_URL = "https://discord.com/api/webhooks/1388129650185732236/t4sICEWcUnyZmPQe-iYlCZTLxhK1TjF3Ucotxltm59BO4gPZYB2Q-ybzdVCOYa0DXDVn"


# ====== 通知 ======
def send_discord_message(message: str):
    payload = {"content": message}
    response = requests.post(WEBHOOK_URL, json=payload)
    if response.status_code != 204:
        print("❌ Discord通知に失敗:", response.status_code, response.text)


# ====== Hard Triplet ======
def make_triplets_hard(vectors, labels):
    anchors, positives, negatives = [], [], []
    with torch.no_grad():
        dists = torch.cdist(vectors, vectors, p=2)
    for i in range(len(vectors)):
        label = labels[i]
        pos_idx = torch.where(labels == label)[0]
        neg_idx = torch.where(labels != label)[0]
        pos_idx = pos_idx[pos_idx != i]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        hardest_pos = pos_idx[torch.argmax(dists[i, pos_idx])]
        hardest_neg = neg_idx[torch.argmin(dists[i, neg_idx])]
        anchors.append(vectors[i])
        positives.append(vectors[hardest_pos])
        negatives.append(vectors[hardest_neg])
    if not anchors:
        return None, None, None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


# ====== トレーニング ======
def train_one(loss_type, mode, use_grl=True, use_mlp=False, datatype='animalkingdom', vmae_version='base'):
    send_discord_message(
        f"🚀 Training started: mode={mode} | loss={loss_type} | {'mlp' if use_mlp else 'linear'} | GRL={'ON' if use_grl else 'OFF'}"
    )

    n_epochs = 1000
    patience = 50

    # ---- パス決定 ----
    base_name = {
        'simple': 'vectors_simple',
        'adaptive3d': 'vectors_adaptive3d',
        'adaptive1d': 'vectors_adaptive1d',
        'sliding': 'vectors_sliding',
    }[mode]

    if datatype == 'wolf':
        output_path = f'./vector/{datatype}/train/{base_name}_{vmae_version}.json'
        csv_file = f"./label/{datatype}/train/labels.csv"
    elif datatype == 'animalkingdom':
        output_path = f'./vector/{datatype}/train/{base_name}.json'
        csv_file = f"./label/{datatype}/train/labels.csv"
    else:
        raise ValueError(f"Unknown datatype: {datatype}")

    # ---- データロード ----
    df, vecs, le_act, le_sp = load_data(csv_file, output_path)
    A, S = len(le_act.classes_), len(le_sp.classes_)
    loader = create_dataloader(df, vecs, batch_size=64, shuffle=True)

    # ---- モデル ----
    net = (DisentangleNetMLP if use_mlp else DisentangleNetSimple)(D=768, H=256, A=A, S=S).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2) if loss_type == 'triplet' else ImprovedTripletLoss(tau1=1.0, tau2=0.5, beta=0.5)

    ce_act = nn.CrossEntropyLoss()
    ce_sp = nn.CrossEntropyLoss()

    # ---- ログ & パス ----
    path_suffix = f"{'mlp' if use_mlp else 'linear'}_{'grl' if use_grl else 'nogrl'}"
    suffix = f"{datatype}_{vmae_version}_{mode}_{loss_type}_{path_suffix}"

    dir_model = f"./model/{datatype}/{vmae_version}/{loss_type}"
    dir_loss = f"./loss/{datatype}/{vmae_version}/{loss_type}"
    os.makedirs(dir_model, exist_ok=True)
    os.makedirs(dir_loss, exist_ok=True)

    model_path = os.path.join(dir_model, f"{path_suffix}.pth")
    loss_plot_path = os.path.join(dir_loss, f"{path_suffix}.png")
    loss_log_path = os.path.join(dir_loss, "final_losses_summary.txt")

    best_loss = float('inf')
    no_improve = 0
    log = {'triplet_action': [], 'triplet_species': [], 'adv_species': [], 'adv_action': [], 'total': []}

    for epoch in range(n_epochs):
        sum_trip_a = sum_trip_s = sum_adv_s = sum_adv_a = sum_total = 0
        steps = 0

        for z, a, s in loader:
            z, a, s = z.cuda(), a.cuda().long(), s.cuda().long()
            grl_lambda = 1.0 if use_grl else 0.0

            a_vec, s_vec, s_pred_from_a, a_pred_from_s = net(z, grl_lambda=grl_lambda)

            anc_a, pos_a, neg_a = make_triplets_hard(a_vec, a)
            anc_s, pos_s, neg_s = make_triplets_hard(s_vec, s)
            if anc_a is None or anc_s is None:
                continue

            loss_trip_a = triplet_loss_fn(anc_a, pos_a, neg_a)
            loss_trip_s = triplet_loss_fn(anc_s, pos_s, neg_s)
            loss_adv_s = ce_sp(s_pred_from_a, s)
            loss_adv_a = ce_act(a_pred_from_s, a)

            loss = (LAMBDA_ACTION * loss_trip_a) + (LAMBDA_SPECIES * loss_trip_s)
            if use_grl:
                loss += LAMBDA_ADV * (loss_adv_s + loss_adv_a)

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_trip_a += loss_trip_a.item()
            sum_trip_s += loss_trip_s.item()
            sum_adv_s += loss_adv_s.item()
            sum_adv_a += loss_adv_a.item()
            sum_total += loss.item()
            steps += 1

        if steps == 0:
            print(f"⚠️ No valid steps at epoch {epoch}. Stopping early.")
            break

        avg_loss = sum_total / steps
        log['triplet_action'].append(sum_trip_a / steps)
        log['triplet_species'].append(sum_trip_s / steps)
        log['adv_species'].append(sum_adv_s / steps)
        log['adv_action'].append(sum_adv_a / steps)
        log['total'].append(avg_loss)

        print(f"Epoch {epoch:03d} | total={avg_loss:.4f} | trip_a={sum_trip_a/steps:.4f} | trip_s={sum_trip_s/steps:.4f} | adv_s={sum_adv_s/steps:.4f} | adv_a={sum_adv_a/steps:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(net.state_dict(), model_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"🛑 No improvement for {patience} epochs. Stopping.")
                break

    # ---- ロス保存 ----
    plt.figure(figsize=(10, 6))
    plt.plot(log['total'], label='Total')
    plt.plot(log['triplet_action'], label='Triplet Action')
    plt.plot(log['triplet_species'], label='Triplet Species')
    plt.plot(log['adv_species'], label='Adv Species')
    plt.plot(log['adv_action'], label='Adv Action')
    plt.title(f"Training Loss: {suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)

    with open(loss_log_path, "a") as f:
        f.write(
            f"[{suffix}] epoch={epoch} | "
            f"total={avg_loss:.4f} | trip_a={sum_trip_a/steps:.4f} | "
            f"trip_s={sum_trip_s/steps:.4f} | "
            f"adv_s={sum_adv_s/steps:.4f} | adv_a={sum_adv_a/steps:.4f}\n"
        )

    send_discord_message(
        f"✅ Training complete: {suffix}\n"
        f"Saved model: {model_path}\n"
        f"Total Loss: {avg_loss:.4f}\n"
        f"Triplet Action Loss: {sum_trip_a/steps:.4f}\n"
        f"Triplet Species Loss: {sum_trip_s/steps:.4f}\n"
        f"Adversarial Species Loss: {sum_adv_s/steps:.4f}\n"
        f"Adversarial Action Loss: {sum_adv_a/steps:.4f}"
    )


# ====== 実行 ======
def main():
    vmae_version = "base" # "base", "v2-base", "v2-large"
    all_modes = ['sliding'] # 'simple', 'adaptive3d', 'adaptive1d', 'sliding'
    all_loss_types = ['triplet', 'improved']
    datatype = 'wolf'

    for mode in all_modes:
        for loss_type in all_loss_types:
            for use_mlp in [True, False]:
                print(f"\n🚀 Training: mode={mode} | loss={loss_type} | {'mlp' if use_mlp else 'linear'} | GRL=ON | datatype={datatype}")
                train_one(loss_type, mode, use_grl=True, use_mlp=use_mlp, datatype=datatype, vmae_version=vmae_version)

                print(f"\n🚀 Training: mode={mode} | loss={loss_type} | {'mlp' if use_mlp else 'linear'} | GRL=OFF | datatype={datatype}")
                train_one(loss_type, mode, use_grl=False, use_mlp=use_mlp, datatype=datatype, vmae_version=vmae_version)


if __name__ == "__main__":
    main()
