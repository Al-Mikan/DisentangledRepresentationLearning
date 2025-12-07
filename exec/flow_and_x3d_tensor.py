import os
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

from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image

from pytorchvideo.models.hub import x3d_m
from typing import List


# ============================================
# ① RAFT & X3D のロード
# ============================================
args = SimpleNamespace(small=False, mixed_precision=True, dropout=0.0, alternate_corr=False)
raft_model = RAFT(args)

state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
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

    # ---------- flow → RGB ----------
    rgb_frames = np.stack([
        cv2.resize(flow_to_image(f), (224, 224))
        for f in flow_list
    ])

    # ---------- sliding clips ----------
    clips = []
    for start in range(0, len(rgb_frames) - clip_len + 1, stride):
        clip = rgb_frames[start:start + clip_len]

        clip_tensor = torch.from_numpy(clip).float().div(255.0).permute(0, 3, 1, 2)

        for t in range(clip_tensor.shape[0]):
            clip_tensor[t] = normalize(
                clip_tensor[t],
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225]
            )

        clips.append(clip_tensor.permute(1, 0, 2, 3))  # (C, T, H, W)

    if not clips:
        print(f"⚠️ 有効 clip なし: {out_video_root}")
        return

    # ---------- X3D ----------
    all_features = []
    for i in range(0, len(clips), clip_batch_size):
        batch = torch.stack(clips[i:i+clip_batch_size]).to(device)

        with torch.no_grad():
            x = batch

            # 前半5ブロック
            for j in range(5):
                x = x3d_model.blocks[j](x)

            # Global Average Pooling
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

    # avg_pooling
    avg_vec = all_features.mean(dim=0, keepdim=True)
    np.save(avg_pool_path, avg_vec.numpy())

    print(f"💾 保存完了: {out_video_root}")


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

    # ---------- 出力パス構築 ----------
    out_norm = os.path.join(out_dir_normal, video_name)
    out_cent = os.path.join(out_dir_centered, video_name)

    os.makedirs(out_norm, exist_ok=True)
    os.makedirs(out_cent, exist_ok=True)

    # どちらも完了していたらスキップ
    if os.path.exists(os.path.join(out_norm, "avg_pooling.npy")) and \
       os.path.exists(os.path.join(out_cent, "avg_pooling.npy")):
        print(f"⏩ Skip: {video_name}")
        return

    # ---------- 動画読み込み ----------
    vr = VideoReader(video_path)
    if len(vr) < clip_len:
        print(f"⚠️ フレーム不足: {video_name}")
        return

    transform = transforms.ToTensor()

    # ---------- RAFT optical flow ----------
    pairs = []
    for i in range(len(vr) - 1):
        f1 = transform(Image.fromarray(vr[i].asnumpy()))
        f2 = transform(Image.fromarray(vr[i + 1].asnumpy()))
        pairs.append((f1, f2))

    flow_list = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        img1 = torch.stack([p[0] for p in batch]).cuda()
        img2 = torch.stack([p[1] for p in batch]).cuda()

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            _, flow_up = raft_model(image1, image2, iters=3, test_mode=True)

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
        mean_flow = np.mean(f, axis=(0, 1))
        centered_flows.append(f - mean_flow)

    # ---------- X3D抽出 ----------
    process_one_flow_type(normal_flows, out_norm, clip_len, stride, clip_batch_size)
    process_one_flow_type(centered_flows, out_cent, clip_len, stride, clip_batch_size)


# ============================================
# ④ CSV を読み取って全動画処理
# ============================================
if __name__ == "__main__":

    csv_files = [
        "./label/animalkingdom/test/labels_test.csv",
    ]

    for csv in csv_files:
        df = pd.read_csv(csv)
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        parts = os.path.normpath(csv).split(os.sep)
        datatype = "polar"

        out_normal = f"./x3d_vector/{datatype}"
        out_centered = f"./x3d_vector_centered/{datatype}"

        os.makedirs(out_normal, exist_ok=True)
        os.makedirs(out_centered, exist_ok=True)

        start_all = time.time()
        for rel_path in tqdm(df["video_path"], desc=f"Processing {datatype}"):
            full_path = os.path.join("./", rel_path)
            extract_flow_to_x3d(full_path, out_normal, out_centered)

        print(f"⏱️ {datatype} total time: {time.time() - start_all:.2f} sec")
