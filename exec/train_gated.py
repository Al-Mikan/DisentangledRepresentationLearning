import os
import json
import traceback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import requests
from model import DisentangleNetMLP, DisentangleNetSimple, GatedFusion
from improved_triplet_loss import ImprovedTripletLoss
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# ===== 定数 =====
LAMBDA_ACTION = 2.0
LAMBDA_SPECIES = 0.5
LAMBDA_ADV = 0.1
WEBHOOK_URL = "https://discord.com/api/webhooks/1390239435991552010/DScH5B8o6P5Akgk5X1l_1FIt7Jd2q6pezbrUJvaMfqTu4AO0eTC_bkNc6HUGmRFXKqhc"

def send_discord_message(message: str):
    payload = {"content": message}
    response = requests.post(WEBHOOK_URL, json=payload)
    if response.status_code != 204:
        print("❌ Discord通知に失敗:", response.status_code, response.text)

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

# ===== ハードトリプレット作成 =====
def make_triplets_hard(vectors, labels):
    """ハードなポジティブ・ネガティブを選んでトリプレットを作る"""
    anchors, positives, negatives = [], [], []
    with torch.no_grad():
        dists = torch.cdist(vectors, vectors, p=2)  # ペア距離行列
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

# ===== トレーニングループ =====
def train_one(loss_type, use_grl=True, use_mlp=False, datatype='animalkingdom'):
    try:
        send_discord_message(f"🚀 Start: loss={loss_type} | {'MLP' if use_mlp else 'Linear'} | GRL={'ON' if use_grl else 'OFF'}")

        n_epochs = 1000
        patience = 50

        # --- 入力パス ---
        csv_file = f"./label/{datatype}/train/labels.csv"
        x3d_dir = f"./x3d_output/animalkingdom/train"
        vmae_json = f"./vector/{datatype}/train/vectors_sliding_base.json"

        # --- データセット & DataLoader ---
        dataset = X3DVideoMAEDataset(csv_file, x3d_dir, vmae_json)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # --- クラス数 ---
        A = len(dataset.df['action'].unique())
        S = len(dataset.df['species'].unique())

        # --- モデル準備 ---
        fusion = GatedFusion(d_x3d=2048, d_vmae=768, d_hidden=512).cuda()
        net = (DisentangleNetMLP if use_mlp else DisentangleNetSimple)(D=512, H=256, A=A, S=S).cuda()

        params = list(fusion.parameters()) + list(net.parameters())
        opt = torch.optim.Adam(params, lr=1e-4)

        # --- ロス関数 ---
        triplet_loss_fn = nn.TripletMarginLoss(margin=0.1, p=2) if loss_type == 'triplet' else ImprovedTripletLoss(tau1=0.1, tau2=0.2, beta=0.5)
        ce_act = nn.CrossEntropyLoss()
        ce_sp = nn.CrossEntropyLoss()

        # --- 保存パス ---
        suffix = f"{'mlp' if use_mlp else 'linear'}-{'grl' if use_grl else 'nogrl'}-adv{LAMBDA_ADV:.2f}"
        
        dir_model = f"./models/model_gated/{datatype}/{loss_type}"
        dir_loss = f"./losses/loss_gated/{datatype}/{loss_type}"
        os.makedirs(dir_model, exist_ok=True)
        os.makedirs(dir_loss, exist_ok=True)

        model_path = os.path.join(dir_model, f"{suffix}.pth")
        loss_plot_path = os.path.join(dir_loss, f"{suffix}.png")
        loss_log_path = os.path.join(dir_loss, "final_losses_summary.txt")

        # --- ログ ---
        best_loss = float('inf')
        no_improve = 0
        log = {'triplet_action': [], 'triplet_species': [], 'adv_species': [], 'adv_action': [], 'total': []}

        log_alpha_mean = []
        # === エポックループ ===
        for epoch in range(n_epochs):
            sum_trip_a = sum_trip_s = sum_adv_s = sum_adv_a = sum_total = 0
            steps = 0

            log_alpha_epoch = []

            for x3d, vmae, a, s in loader:
                x3d, vmae = x3d.cuda(), vmae.cuda()
                a, s = a.cuda().long(), s.cuda().long()
                grl_lambda = 1.0 if use_grl else 0.0

                fused_vec, gating_weight = fusion(x3d, vmae)
                fused_vec = nn.functional.normalize(fused_vec, dim=-1)
                alpha = gating_weight.detach().cpu().numpy()

                # === α のバッチ平均を保存 ===
                log_alpha_epoch.append(alpha.mean())

                if epoch == 0 and steps == 0:
                    print("=== GatedFusion 動作チェック ===")
                    print("x3d.shape        :", x3d.shape)
                    print("vmae.shape       :", vmae.shape)
                    print("fused_vec.shape  :", fused_vec.shape)
                    print("gating_weight    :", gating_weight[:5].detach().cpu().numpy())  # 先頭5つ
                    print("===============================")

                a_vec, s_vec, s_pred_from_a, a_pred_from_s = net(fused_vec, grl_lambda=grl_lambda)
                a_vec = nn.functional.normalize(a_vec, dim=-1)
                s_vec = nn.functional.normalize(s_vec, dim=-1)

                anc_a, pos_a, neg_a = make_triplets_hard(a_vec, a)
                anc_s, pos_s, neg_s = make_triplets_hard(s_vec, s)
                if anc_a is None or anc_s is None:
                    continue

                # --- ロス計算 ---
                loss_trip_a = triplet_loss_fn(anc_a, pos_a, neg_a)
                loss_trip_s = triplet_loss_fn(anc_s, pos_s, neg_s)
                loss_adv_s = ce_sp(s_pred_from_a, s)
                loss_adv_a = ce_act(a_pred_from_s, a)

                loss = LAMBDA_ACTION * loss_trip_a + LAMBDA_SPECIES * loss_trip_s
                if use_grl:
                    loss += LAMBDA_ADV * (loss_adv_s + loss_adv_a)

                # --- 最適化 ---
                opt.zero_grad()
                loss.backward()
                opt.step()

                # --- ログ更新 ---
                sum_trip_a += loss_trip_a.item()
                sum_trip_s += loss_trip_s.item()
                sum_adv_s += loss_adv_s.item()
                sum_adv_a += loss_adv_a.item()
                sum_total += loss.item()
                steps += 1

            if steps == 0:
                print(f"⚠️ No valid steps at epoch {epoch}. Stopping early.")
                break

            # === エポックの最後でエポック内 α 平均をまとめる ===
            mean_alpha = np.mean(log_alpha_epoch)
            log_alpha_mean.append(mean_alpha)
            print(f"[Epoch {epoch}] mean α over epoch: {mean_alpha:.4f}")

            avg_loss = sum_total / steps
            log['triplet_action'].append(sum_trip_a / steps)
            log['triplet_species'].append(sum_trip_s / steps)
            log['adv_species'].append(sum_adv_s / steps)
            log['adv_action'].append(sum_adv_a / steps)
            log['total'].append(avg_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"total={avg_loss:.4f} | "
                f"triplet_action={sum_trip_a/steps:.4f} | "
                f"triplet_species={sum_trip_s/steps:.4f} | "
                f"adv_species={sum_adv_s/steps:.4f} | "
                f"adv_action={sum_adv_a/steps:.4f} | "
                f"steps={steps} | "
                f"best={best_loss:.4f} | "
                f"no_improve={no_improve}"
            )

            # --- モデル保存 & アーリーストップ ---
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                torch.save({'fusion': fusion.state_dict(), 'net': net.state_dict()}, model_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"🛑 Early stopping")
                    break

        # --- 損失グラフ保存 ---
        plt.figure()
        plt.plot(log['total'], label='Total')
        plt.plot(log['triplet_action'], label='Triplet Action')
        plt.plot(log['triplet_species'], label='Triplet Species')
        plt.plot(log['adv_species'], label='Adv Species')
        plt.plot(log['adv_action'], label='Adv Action')
        plt.title(suffix)
        plt.legend()
        plt.grid()
        plt.savefig(loss_plot_path)

        # === αの平均推移グラフ ===
        alpha_plot_path = os.path.join(dir_loss, f"{suffix}_alpha.png")
        plt.figure()
        plt.plot(log_alpha_mean, label='mean α (GatedFusion)', color='tab:orange')
        plt.xlabel('Epoch')
        plt.ylabel('Mean α')
        plt.title(f"Gating weight (α) mean per epoch")
        plt.grid()
        plt.legend()
        plt.savefig(alpha_plot_path)

        # --- ログファイルに最終結果追記 ---
        with open(loss_log_path, "a") as f:
            f.write(f"[{suffix}] last_loss={avg_loss:.4f}\n")

        send_discord_message(f"✅ Training complete: {suffix}\nSaved: {model_path}")

    except Exception as e:
        tb = traceback.format_exc()
        msg = (
            f"❌ Error during training!\n"
            f"loss_type: {loss_type}, use_grl: {use_grl}, use_mlp: {use_mlp}, datatype: {datatype}\n"
            f"Error: {str(e)}\n"
            f"Traceback:\n```{tb}```"
        )
        print(msg)
        send_discord_message(msg)
        raise  

def main():
    # ---- どの設定を試すか ----
    loss_types = ['improved']           # 'triplet', 'improved' など
    use_mlp_options = [True, False]     # True: MLP ヘッド, False: Linear ヘッド
    use_grl_options = [True, False]     # True: GRL 使用, False: GRL なし
    datatype = 'animalkingdom'          # データセット名

    # ---- 組み合わせで全探索 ----
    for loss_type in loss_types:
        for use_mlp in use_mlp_options:
            for use_grl in use_grl_options:
                print("\n🚀 === 実行 ===")
                print(f"loss_type : {loss_type}")
                print(f"use_mlp   : {use_mlp}")
                print(f"use_grl   : {use_grl}")
                print(f"dataset   : {datatype}")

                try:
                    train_one(
                        loss_type=loss_type,
                        use_grl=use_grl,
                        use_mlp=use_mlp,
                        datatype=datatype
                    )
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue


if __name__ == "__main__":
    main()