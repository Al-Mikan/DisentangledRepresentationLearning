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
# モデルロード
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
# 特徴抽出関数
# ============================================
def extract_flow_to_x3d(video_path, out_dir, clip_len=16, stride=8, batch_size=2, clip_batch_size=2):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # ✨修正点 1: 保存先フォルダ名を "_centered" を付けて区別する
    out_dir_centered = os.path.join(out_dir, video_name)
    os.makedirs(out_dir_centered, exist_ok=True)

    feature_path = os.path.join(out_dir_centered, f"{video_name}.npy")
    if os.path.exists(feature_path):
        print(f"✅ 既に存在: {feature_path}")
        return

    vr = VideoReader(video_path)
    if len(vr) <  clip_len:
        print(f"⚠️ フレーム不足: {video_path}")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    pairs = [(transform(Image.fromarray(vr[i].asnumpy())), transform(Image.fromarray(vr[i+1].asnumpy()))) for i in range(len(vr) - 1)]

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
                f = padder.unpad(flow_up[j]).permute(1, 2, 0).cpu().numpy()
                flow_list.append(f)

    processed_flows = []
    for flow_vector in flow_list:
        mean_flow = np.mean(flow_vector, axis=(0, 1))
        centered_flow = flow_vector - mean_flow
        processed_flows.append(centered_flow)
    
    if not processed_flows:
        print(f"⚠️ フロー計算失敗: {video_name}")
        return
        
    flow_array = np.stack(processed_flows)

    # === X3Dへの入力準備 (以降は変更なし) ===
    total_T = flow_array.shape[0]
    rgb_frames = np.stack([cv2.resize(flow_to_image(flow_array[i]), (224, 224), interpolation=cv2.INTER_AREA) for i in range(total_T)])
    
    clips = []
    for start in range(0, total_T - clip_len + 1, stride):
        clip = rgb_frames[start:start+clip_len]
        clip_tensor = torch.from_numpy(clip).float().div(255.0).permute(0, 3, 1, 2)
        for t in range(clip_tensor.shape[0]):
            clip_tensor[t] = normalize(clip_tensor[t], mean=[0.45]*3, std=[0.225]*3)
        clips.append(clip_tensor.permute(1, 0, 2, 3))

    if not clips:
        print(f"⚠️ 有効クリップなし: {video_name}")
        return

    pooled_all = []
    for i in range(0, len(clips), clip_batch_size):
        clip_batch = torch.stack(clips[i:i+clip_batch_size]).to(device)
        with torch.no_grad():
            x = clip_batch
            for j in range(5):
                x = x3d_model.blocks[j](x)
            
            pool = x3d_model.blocks[5].pool
            x = pool.pre_conv(x); x = pool.pre_norm(x); x = pool.pre_act(x)
            x = pool.pool(x)
            x = pool.post_conv(x); x = pool.post_act(x)
            features = x.flatten(1)
            pooled_all.append(features.cpu())

    if not pooled_all:
        print(f"⚠️ 特徴抽出失敗: {video_name}")
        return

    pooled_all = torch.cat(pooled_all, dim=0)
    feature_vector = pooled_all.mean(dim=0, keepdim=True)
    np.save(feature_path, feature_vector.numpy())


# ============================================
# 実行 (変更なし)
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
        datatype = parts[parts.index('label') + 1]
        is_test = 'test' in parts
        # ✨修正点 3: 保存先の親フォルダ名を "x3d_output_centered" に変更
        out_dir = f"./x3d_output_centered/{datatype}/{'test' if is_test else 'train'}"
        os.makedirs(out_dir, exist_ok=True)

        start_all = time.time()
        for rel_path in tqdm(df["video_path"], desc=f"Processing {os.path.basename(csv)}"):
            full_path = os.path.join('./', rel_path)
            start = time.time()
            extract_flow_to_x3d(full_path, out_dir)
            elapsed = time.time() - start
            # print(f"⏱️ 処理時間: {elapsed:.2f} 秒") # ログが多くなりすぎるのでコメントアウト

        print(f"⏱️ {os.path.basename(csv)} の総処理時間: {time.time()-start_all:.2f} 秒")