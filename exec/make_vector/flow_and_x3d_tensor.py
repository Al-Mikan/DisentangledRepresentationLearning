import os
from pathlib import Path
import time
import numpy as np
import torch
import cv2
from types import SimpleNamespace
import pandas as pd
from PIL import Image
from tqdm import tqdm
from decord import VideoReader
from torchvision import transforms
from torchvision.transforms.functional import normalize

THIS_DIR = Path(__file__).parent.resolve()
os.sys.path.insert(0, str(THIS_DIR))

from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image

from pytorchvideo.models.hub import x3d_m
from typing import List


# ============================================
# ① RAFT & X3D のロード
# ============================================
class AttrDict(dict):
    """dict と Namespace のハイブリッド（RAFT 対策）"""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


args = AttrDict(
    small=False,
    mixed_precision=True,
    dropout=0.0,
    alternate_corr=False,
)

raft_model = RAFT(args)

state_dict = torch.load("./exec/make_vector/RAFT/models/raft-sintel.pth")
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
raft_model.load_state_dict(new_state_dict)

raft_model = raft_model.eval().cuda()
device = torch.device("cuda")

x3d_model = x3d_m(pretrained=True).eval().to(device)


# ============================================
# ② X3D 抽出 + 保存（normal / centered 共通）
# ============================================
def process_one_flow_type(flow_list, out_video_root, clip_len, stride, clip_batch_size):
    sliding_dir = os.path.join(out_video_root, "sliding_list")
    avg_pool_path = os.path.join(out_video_root, "avg_pooling.npy")

    os.makedirs(sliding_dir, exist_ok=True)

    rgb_frames = np.stack([
        cv2.resize(flow_to_image(f), (224, 224))
        for f in flow_list
    ])

    # ---------- sliding clips ----------
    clips = []
    # 指摘⑤への対応: stride=1 は重いが、細かく見るならこれでOK
    for start in range(0, len(rgb_frames) - clip_len + 1, stride):
        clip = rgb_frames[start:start + clip_len]

        clip_tensor = torch.from_numpy(clip).float().div(255.0).permute(0, 3, 1, 2)

        # X3D入力用の正規化 (Kinetics-400/ImageNet準拠)
        for t in range(clip_tensor.shape[0]):
            clip_tensor[t] = normalize(
                clip_tensor[t],
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225]
            )

        clips.append(clip_tensor.permute(1, 0, 2, 3))  # (C, T, H, W)

    if not clips:
        # 短すぎてクリップが作れなかった場合など
        if len(rgb_frames) > 0:
             print(f"⚠️ 動画が短すぎます (len={len(rgb_frames)}): {out_video_root}")
        return

    # ---------- X3D Feature Extraction ----------
    all_features = []
    for i in range(0, len(clips), clip_batch_size):
        batch = torch.stack(clips[i:i+clip_batch_size]).to(device)

        with torch.no_grad():
            x = batch
            # 指摘④への対応: 分類ヘッドの手前で特徴を抜く正しい処理
            for j in range(5):
                x = x3d_model.blocks[j](x)

            pool = x3d_model.blocks[5].pool
            x = pool.pre_conv(x)
            x = pool.pre_norm(x)
            x = pool.pre_act(x)
            x = pool.pool(x)
            x = pool.post_conv(x)
            x = pool.post_act(x)

            feats = x.flatten(1).cpu()
            all_features.append(feats)

    all_features = torch.cat(all_features, dim=0)

    # ---------- 保存 ----------
    # sliding_list
    for idx, feat in enumerate(all_features):
        np.save(os.path.join(sliding_dir, f"{idx:03d}.npy"), feat.numpy())

    # avg_pooling (指摘⑥: 動画全体の特徴として正しい)
    avg_vec = all_features.mean(dim=0, keepdim=True)
    np.save(avg_pool_path, avg_vec.numpy())

    # print(f"💾 保存完了: {out_video_root}")


# ============================================
# ③ normal + centered の両方を作る
# ============================================
def extract_flow_to_x3d(
    video_path,
    out_dir_normal,
    out_dir_centered,
    clip_len=16,
    stride=1,
    batch_size=2,
    clip_batch_size=2,
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    out_norm = os.path.join(out_dir_normal, video_name)
    out_cent = os.path.join(out_dir_centered, video_name)

    os.makedirs(out_norm, exist_ok=True)
    os.makedirs(out_cent, exist_ok=True)

    if os.path.exists(os.path.join(out_norm, "avg_pooling.npy")) and \
       os.path.exists(os.path.join(out_cent, "avg_pooling.npy")):
        # print(f"⏩ Skip: {video_name}")
        return

    vr = VideoReader(video_path)
    if len(vr) < clip_len:
        print(f"⚠️ フレーム不足: {video_name}")
        return

    # ▼▼▼【修正点1】RAFT入力用の正規化を追加 (指摘①への対応) ▼▼▼
    # これにより [0, 1] -> [-1, 1] に変換され、RAFTの精度が安定します
    raft_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    pairs = []
    for i in range(len(vr) - 1):
        f1 = raft_transform(Image.fromarray(vr[i].asnumpy()))
        f2 = raft_transform(Image.fromarray(vr[i + 1].asnumpy()))
        pairs.append((f1, f2))

    flow_list = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        img1 = torch.stack([p[0] for p in batch]).cuda()
        img2 = torch.stack([p[1] for p in batch]).cuda()

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            
            # ▼▼▼【修正点2】iters=12 に設定 (指摘②への対応) ▼▼▼
            # 3は弱すぎ、5でも可だが、精度重視で12を採用
            _, flow_up = raft_model(image1, image2, iters=12, test_mode=True)

            for j in range(flow_up.shape[0]):
                f = padder.unpad(flow_up[j]).permute(1, 2, 0).cpu().numpy()
                flow_list.append(f)

    if not flow_list:
        print(f"⚠️ Flow エラー: {video_name}")
        return

    # ---------- normal / centered の2種類 ----------
    normal_flows = flow_list

    centered_flows = []
    for f in flow_list:
        # ▼▼▼【維持】Medianによるカメラワーク除去 (指摘⑦の発展形) ▼▼▼
        # 指摘ではMeanでもOKとありましたが、カメラワークにはMedianが最強です
        cam_flow = np.median(f, axis=(0, 1))
        centered_flows.append(f - cam_flow)

    # ---------- X3D抽出 ----------
    process_one_flow_type(normal_flows, out_norm, clip_len, stride, clip_batch_size)
    process_one_flow_type(centered_flows, out_cent, clip_len, stride, clip_batch_size)


# ============================================
# ④ CSV を読み取って全動画処理
# ============================================
if __name__ == "__main__":

    # CSVファイルのリスト
    csv_files = [
        "./label/train/animalkingdom_augmented.csv",
        # 必要に応じて追加
    ]

    for csv in csv_files:
        if not os.path.exists(csv):
            print(f"⚠️ CSVが見つかりません: {csv}")
            continue

        df = pd.read_csv(csv)
        # Windowsパス対策
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        # データタイプの自動判定（フォルダ構成などに依存する場合は適宜修正）
        # 例: ./label/test/elephant.csv -> elephant
        datatype = os.path.splitext(os.path.basename(csv))[0] 

        out_normal = f"./x3d_vector/{datatype}"
        out_centered = f"./x3d_vector_centered/{datatype}"

        os.makedirs(out_normal, exist_ok=True)
        os.makedirs(out_centered, exist_ok=True)

        print(f"🚀 Start processing: {datatype} (Total: {len(df)})")
        start_all = time.time()
        
        for rel_path in tqdm(df["video_path"], desc=f"Processing {datatype}"):
            full_path = os.path.join("./", rel_path)
            
            if not os.path.exists(full_path):
                # パスが見つからない場合のデバッグ用
                # print(f"Video not found: {full_path}")
                continue
                
            extract_flow_to_x3d(full_path, out_normal, out_centered)

        print(f"⏱️ {datatype} total time: {time.time() - start_all:.2f} sec")