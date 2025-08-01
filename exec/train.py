import os
import json
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import LabelEncoder

from model import (
    DisentangleNetMLP,
    DisentangleNetSimple,
    GatedFusion,
    load_data,
    create_dataloader,
)
from improved_triplet_loss import ImprovedTripletLoss

# ====== 共通定数 ======
LAMBDA_ACTION = 2.0
LAMBDA_SPECIES = 0.5
LAMBDA_ADV = 0.05
WEBHOOK_URL = "https://discord.com/api/webhooks/1392755117576556624/uhRwB1_5v90a-0A1JDspuCelrnIJxr93mOMyBt6S5kM2kGjXJsjEc5kOE3NVaCMxEQSI"


def send_discord_message(message: str):
    payload = {"content": message}
    response = requests.post(WEBHOOK_URL, json=payload)
    if response.status_code != 204:
        print("❌ Discord通知に失敗:", response.status_code, response.text)


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

# ====== Flow-only Dataset ======
class FlowNpyDataset(Dataset):
    def __init__(self, csv_path, flow_dir):
        self.df = pd.read_csv(csv_path)
        self.flow_dir = flow_dir
        valid = []
        for idx, row in self.df.iterrows():
            video_id = os.path.splitext(os.path.basename(row['video_path']))[0]
            npy = os.path.join(flow_dir, video_id, f"{video_id}.npy")
            if os.path.isfile(npy):
                valid.append(idx)
        self.df = self.df.loc[valid].reset_index(drop=True)
        self.le_act = LabelEncoder().fit(self.df['action'])
        self.le_sp = LabelEncoder().fit(self.df['species'])
        self.df['act_id'] = self.le_act.transform(self.df['action'])
        self.df['sp_id'] = self.le_sp.transform(self.df['species'])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = os.path.splitext(os.path.basename(row['video_path']))[0]
        flow_vec = np.load(os.path.join(self.flow_dir, video_id, f"{video_id}.npy")).squeeze(0)
        return torch.tensor(flow_vec, dtype=torch.float32), int(row['act_id']), int(row['sp_id'])


# ===== Datasetクラス =====
class X3DVideoMAEDataset(Dataset):
    """X3DとVideoMAEを同時にロードするDataset"""
    def __init__(self, csv_path, x3d_dir, vmae_json):
        print(f"=== ✅ 受け取った csv_path: {csv_path} ===")
        self.df = pd.read_csv(csv_path)
        with open(vmae_json, 'r') as f:
            self.vmae_dict = json.load(f)
        self.x3d_dir = x3d_dir

        # --- npyファイルとvmae両方存在するものだけに絞る ---
        valid_indices = []
        for idx, row in self.df.iterrows():
            video_path = row['video_path'].replace('\\', '/').strip()
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            x3d_path = os.path.join(self.x3d_dir, video_id, f"{video_id}.npy")
            # npyとvmae両方ある
            if os.path.isfile(x3d_path) and (video_path in self.vmae_dict):
                valid_indices.append(idx)
            # else:
            #     print(f"❌ スキップ: {video_path} (npy or vmae 不足)")
        self.df = self.df.loc[valid_indices].reset_index(drop=True)

        # --- ラベルエンコード ---
        self.le_act = LabelEncoder().fit(self.df['action'])
        self.le_sp = LabelEncoder().fit(self.df['species'])
        self.df['act_id'] = self.le_act.transform(self.df['action'])
        self.df['sp_id'] = self.le_sp.transform(self.df['species'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row['video_path'].replace('\\', '/').strip()
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # --- X3Dベクトル ---
        x3d_path = os.path.join(self.x3d_dir, video_id, f"{video_id}.npy")
        x3d_vec = np.load(x3d_path).squeeze(0)

        # --- VMAEベクトル ---
        vmae_vec = np.array(self.vmae_dict.get(video_path))
        if vmae_vec is None:
            raise ValueError(f"❌ VMAE vector not found for {video_path}")

        a = row['act_id']
        s = row['sp_id']

        return (
            torch.tensor(x3d_vec, dtype=torch.float32),
            torch.tensor(vmae_vec, dtype=torch.float32),
            int(a),
            int(s),
        )

# ====== train_one_flow ======
def train_one_flow(loss_type, use_grl, use_mlp, datatype):
    csv_file = f"./label/{datatype}/train/labels.csv"
    flow_dir = f"./x3d_output/{datatype}/train"
    dataset = FlowNpyDataset(csv_file, flow_dir)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    D = dataset[0][0].shape[0]
    A, S = len(dataset.le_act.classes_), len(dataset.le_sp.classes_)
    net = (DisentangleNetMLP if use_mlp else DisentangleNetSimple)(D, 256, A, S).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    triplet_loss = ImprovedTripletLoss() if loss_type == 'improved' else nn.TripletMarginLoss(0.1)
    ce_act, ce_sp = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    suffix = f"flow_{'mlp' if use_mlp else 'linear'}-{'grl' if use_grl else 'nogrl'}-adv{LAMBDA_ADV:.2f}"
    os.makedirs(f"./models/model_flow/{datatype}/{loss_type}", exist_ok=True)
    best_loss, patience = float('inf'), 50
    no_improve, log = 0, []
    for epoch in range(1000):
        losses = []
        for vec, a, s in loader:
            vec, a, s = vec.cuda(), a.cuda().long(), s.cuda().long()
            a_vec, s_vec, s_pred, a_pred = net(vec, grl_lambda=1.0 if use_grl else 0.0)
            a_vec = nn.functional.normalize(a_vec, dim=-1)
            s_vec = nn.functional.normalize(s_vec, dim=-1)
            anc_a, pos_a, neg_a = make_triplets_hard(a_vec, a)
            anc_s, pos_s, neg_s = make_triplets_hard(s_vec, s)
            if anc_a is None or anc_s is None: continue
            loss = LAMBDA_ACTION * triplet_loss(anc_a, pos_a, neg_a)
            loss += LAMBDA_SPECIES * triplet_loss(anc_s, pos_s, neg_s)
            if use_grl: loss += LAMBDA_ADV * (ce_sp(s_pred, s) + ce_act(a_pred, a))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        avg = np.mean(losses)
        print(f"[FLOW][{loss_type.upper()}][MLP:{use_mlp}][GRL:{use_grl}] Epoch {epoch:03d} | Loss: {avg:.4f}")
        if avg < best_loss: best_loss, no_improve = avg, 0; torch.save(net.state_dict(), f"./models/model_flow/{datatype}/{loss_type}/{suffix}.pth")
        else: no_improve += 1
        if no_improve >= patience: break


# ====== train_one_gated ======
def train_one_gated(loss_type, use_grl, use_mlp, datatype):
    csv_file = f"./label/{datatype}/train/labels.csv"
    x3d_dir = f"./x3d_output/{datatype}/train"
    vmae_json = f"./vector/{datatype}/train/vectors_sliding_base.json"
    dataset = X3DVideoMAEDataset(csv_file, x3d_dir, vmae_json)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    A, S = len(dataset.le_act.classes_), len(dataset.le_sp.classes_)
    fusion = GatedFusion(2048, 768, 512).cuda()
    net = (DisentangleNetMLP if use_mlp else DisentangleNetSimple)(512, 256, A, S).cuda()
    params = list(fusion.parameters()) + list(net.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    triplet_loss = ImprovedTripletLoss() if loss_type == 'improved' else nn.TripletMarginLoss(0.1)
    ce_act, ce_sp = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    suffix = f"gated_{'mlp' if use_mlp else 'linear'}-{'grl' if use_grl else 'nogrl'}-adv{LAMBDA_ADV:.2f}"
    model_dir = f"./models/model_gated/{datatype}/{loss_type}"
    os.makedirs(model_dir, exist_ok=True)
    alpha_plot_path = os.path.join(model_dir, f"{suffix}_alpha.png")

    best_loss, patience = float('inf'), 50
    alpha_means_all = []

    for epoch in range(1000):
        losses, alpha_means_epoch = [], []
        for x3d, vmae, a, s in loader:
            x3d, vmae, a, s = x3d.cuda(), vmae.cuda(), a.cuda().long(), s.cuda().long()
            fused, alpha = fusion(x3d, vmae)
            a_vec, s_vec, s_pred, a_pred = net(fused, grl_lambda=1.0 if use_grl else 0.0)
            a_vec, s_vec = nn.functional.normalize(a_vec, dim=-1), nn.functional.normalize(s_vec, dim=-1)
            anc_a, pos_a, neg_a = make_triplets_hard(a_vec, a)
            anc_s, pos_s, neg_s = make_triplets_hard(s_vec, s)
            if anc_a is None or anc_s is None: continue
            loss = LAMBDA_ACTION * triplet_loss(anc_a, pos_a, neg_a) + LAMBDA_SPECIES * triplet_loss(anc_s, pos_s, neg_s)
            if use_grl: loss += LAMBDA_ADV * (ce_sp(s_pred, s) + ce_act(a_pred, a))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            alpha_means_epoch.append(alpha.mean().item())

        mean_alpha = np.mean(alpha_means_epoch)
        alpha_means_all.append(mean_alpha)

        avg = np.mean(losses)
        print(f"[GATED][{loss_type.upper()}][MLP:{use_mlp}][GRL:{use_grl}] "
          f"Epoch {epoch:03d} | Loss: {avg:.4f} | α mean: {mean_alpha:.4f}")

        if avg < best_loss:
            best_loss, no_improve = avg, 0
            torch.save({'fusion': fusion.state_dict(), 'net': net.state_dict()},
                       os.path.join(model_dir, f"{suffix}.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    # === αの平均推移をグラフで保存 ===
    plt.figure()
    plt.plot(alpha_means_all, label='Gating α mean per epoch')
    plt.xlabel("Epoch")
    plt.ylabel("Mean α")
    plt.title(f"GatedFusion α: {suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(alpha_plot_path)
    print(f"✅ Saved α plot: {alpha_plot_path}")

# ====== train_one_mae ======
def train_one_mae(loss_type, use_grl, use_mlp, datatype):
    vmae_json = f"./vector/{datatype}/train/vectors_sliding_base.json"
    csv_file = f"./label/{datatype}/train/labels.csv"
    df, vecs, le_act, le_sp = load_data(csv_file, vmae_json)
    A, S = len(le_act.classes_), len(le_sp.classes_)
    loader = create_dataloader(df, vecs, batch_size=64, shuffle=True)
    net = (DisentangleNetMLP if use_mlp else DisentangleNetSimple)(768, 256, A, S).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    triplet_loss = ImprovedTripletLoss() if loss_type == 'improved' else nn.TripletMarginLoss(0.1)
    ce_act, ce_sp = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()
    suffix = f"mae_{'mlp' if use_mlp else 'linear'}-{'grl' if use_grl else 'nogrl'}-adv{LAMBDA_ADV:.2f}"
    os.makedirs(f"./models/model_mae/{datatype}/{loss_type}", exist_ok=True)
    best_loss, patience = float('inf'), 50
    for epoch in range(1000):
        losses = []
        for z, a, s in loader:
            z, a, s = z.cuda(), a.cuda().long(), s.cuda().long()
            a_vec, s_vec, s_pred, a_pred = net(z, grl_lambda=1.0 if use_grl else 0.0)
            a_vec, s_vec = nn.functional.normalize(a_vec, dim=-1), nn.functional.normalize(s_vec, dim=-1)
            anc_a, pos_a, neg_a = make_triplets_hard(a_vec, a)
            anc_s, pos_s, neg_s = make_triplets_hard(s_vec, s)
            if anc_a is None or anc_s is None: continue
            loss = LAMBDA_ACTION * triplet_loss(anc_a, pos_a, neg_a) + LAMBDA_SPECIES * triplet_loss(anc_s, pos_s, neg_s)
            if use_grl: loss += LAMBDA_ADV * (ce_sp(s_pred, s) + ce_act(a_pred, a))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        avg = np.mean(losses)
        print(f"[MAE][{loss_type.upper()}][MLP:{use_mlp}][GRL:{use_grl}] "
          f"Epoch {epoch:03d} | Loss: {avg:.4f}")
        if avg < best_loss: best_loss, no_improve = avg, 0; torch.save(net.state_dict(), f"./models/model_mae/{datatype}/{loss_type}/{suffix}.pth")
        else: no_improve += 1
        if no_improve >= patience: break

# ====== Main ======
def main():
    datatype = 'animalkingdom'
    for loss_type in ['improved',"triplet"]:
        for use_mlp in [True, False]:
            for use_grl in [True, False]:
                print(f"=== Training: {loss_type}, MLP: {use_mlp}, GRL: {use_grl}, Data: {datatype} ===")
                try:
                    train_one_flow(loss_type, use_grl, use_mlp, datatype)
                    train_one_gated(loss_type, use_grl, use_mlp, datatype)
                    train_one_mae(loss_type, use_grl, use_mlp, datatype)
                    send_discord_message(f"✅ Training completed: {loss_type}, MLP: {use_mlp}, GRL: {use_grl}, Data: {datatype}")
                except Exception as e:
                    traceback.print_exc()
                    send_discord_message(f"❌ Training failed: {loss_type}, MLP: {use_mlp}, GRL: {use_grl}, Data: {datatype}\nError: {str(e)}")
                print("\n")
if __name__ == "__main__":
    main()
