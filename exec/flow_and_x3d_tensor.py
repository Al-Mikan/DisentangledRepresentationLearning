import os
import time
import numpy as np
import torch
import cv2
from types import SimpleNamespace
import pandas as pd
from PIL import Image
from decord import VideoReader
from torchvision import transforms
from torchvision.transforms.functional import normalize

from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image

from pytorchvideo.models.hub import x3d_m
from typing import List

# ============================================
# RAFT ロード
# ============================================
args = SimpleNamespace()
args.small = False
args.mixed_precision = True
args.dropout = 0.0
args.alternate_corr = False

raft_model = RAFT(args)
state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
raft_model.load_state_dict(new_state_dict)
raft_model = raft_model.eval().cuda()

# ============================================
# X3D ロード
# ============================================
device = torch.device("cuda")
x3d_model = x3d_m(pretrained=True).eval().to(device)

# ============================================
# optical flow → X3D → feature 保存
# ============================================
def extract_flow_to_x3d(video_path, out_dir, clip_len=16, stride=8, batch_size=2, clip_batch_size=2):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    feature_path = os.path.join(out_dir, f"{video_name}.npy")

    vr = VideoReader(video_path)
    total_frames = len(vr)
    if total_frames < 2:
        print(f"⚠️ フレーム不足: {video_path}")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    pairs = []
    for i in range(total_frames - 1):
        img1 = vr[i].asnumpy()
        img2 = vr[i+1].asnumpy()
        img1_t = transform(Image.fromarray(img1))
        img2_t = transform(Image.fromarray(img2))
        pairs.append((img1_t, img2_t))

    flow_list = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        img1_batch = torch.stack([p[0] for p in batch]).cuda()
        img2_batch = torch.stack([p[1] for p in batch]).cuda()

        with torch.no_grad():
            padder = InputPadder(img1_batch.shape)
            image1, image2 = padder.pad(img1_batch, img2_batch)
            _, flow_up = raft_model(image1, image2, iters=3, test_mode=True)

            for j in range(flow_up.shape[0]):
                f = padder.unpad(flow_up[j])
                f = f.permute(1, 2, 0).cpu().numpy()
                flow_list.append(f)

        del img1_batch, img2_batch, image1, image2, flow_up
        torch.cuda.empty_cache()

    flow_array = np.stack(flow_list)

    # === 可視化 & X3D 入力 ===
    total_T, H, W, _ = flow_array.shape
    rgb_frames: List[np.ndarray] = []

    for i in range(total_T):
        img = flow_to_image(flow_array[i])
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_frames.append(img)

    rgb_frames = np.stack(rgb_frames)

    clips = []
    for start in range(0, total_T - clip_len + 1, stride):
        clip = rgb_frames[start:start+clip_len]
        clip_tensor = torch.from_numpy(clip).float().div(255.0)
        clip_tensor = clip_tensor.permute(0, 3, 1, 2)
        for t in range(clip_tensor.shape[0]):
            clip_tensor[t] = normalize(clip_tensor[t], mean=[0.45]*3, std=[0.225]*3)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3)
        clips.append(clip_tensor)

    if not clips:
        print(f"⚠️ 有効クリップなし: {video_name}")
        return

    pooled_all = []

    for i in range(0, len(clips), clip_batch_size):
        clip_batch = clips[i:i+clip_batch_size]
        clip_batch = torch.stack(clip_batch).to(device)  # [B,3,clip_len,H,W]
        with torch.no_grad():
            x = clip_batch.to(device)

            # 0〜4 は backbone
            for j in range(5):
                x = x3d_model.blocks[j](x)

            # 5 は head (= ResNetBasicHead)
            pool = x3d_model.blocks[5].pool          # ProjectedPool

            x = pool.pre_conv(x)
            x = pool.pre_norm(x)
            x = pool.pre_act(x)
            x = pool.pool(x)                         # [B,C,1,1,1]
            x = pool.post_conv(x)
            x = pool.post_act(x)

            features = x.flatten(1)

        pooled_all.append(features.cpu())

        del clip_batch, x, features
        torch.cuda.empty_cache()

    # === 複数クリップを平均
    pooled_all = torch.cat(pooled_all, dim=0)               # [N,C]
    feature_vector = pooled_all.mean(dim=0, keepdim=True)   # [1,C]
    np.save(feature_path, feature_vector.numpy())

    del flow_array, pooled_all, feature_vector
    torch.cuda.empty_cache()

# ============================================
# 実行
# ============================================
if __name__ == "__main__":

    csv_files = [
        "./label/animalkingdom/train/labels.csv",
        "./label/animalkingdom/test/labels_test_24fps.csv",
        "./label/animalkingdom/test/labels_test.csv",
    ]

    valid_csv_files = [p for p in csv_files if os.path.isfile(p)]
    if not valid_csv_files:
        print("❌ 有効なCSVが見つかりません。")
        exit()

    for csv in valid_csv_files:
        
        df = pd.read_csv(csv)
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        parts = os.path.normpath(csv).split(os.sep)
        labels_idx = parts.index('label')
        datatype = parts[labels_idx + 1]
        is_test = 'test' in parts
        out_dir = f"./x3d_output/{datatype}/{'test' if is_test else 'train'}"
        os.makedirs(out_dir, exist_ok=True)

        start_all = time.time()

        for rel_path in df["video_path"]:
            full_path = os.path.join('./', rel_path)
            start = time.time()
            extract_flow_to_x3d(full_path, out_dir)
            elapsed = time.time() - start
            print(f"⏱️ 処理時間: {elapsed:.2f} 秒")

        print(f"⏱️ 総処理時間: {time.time()-start_all:.2f} 秒")
