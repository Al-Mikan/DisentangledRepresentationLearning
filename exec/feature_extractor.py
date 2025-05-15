# feature_extractor.py

import os
import csv
import json
import numpy as np
import torch
import decord
import pandas as pd
from collections import Counter
from transformers import VideoMAEImageProcessor, VideoMAEModel
import torch.nn.functional as F
# ----------------------
# ① CSV読み込み
# ----------------------
df = pd.read_csv("labels.csv")
df["species"] = df["species"].str.strip()
df["parent_class"] = df["parent_class"].str.strip().str.lower()
df["action"] = df["action"].str.strip()

# ----------------------
# ② 初期設定とモデル読み込み
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

# ----------------------
# ③ 動画 → ベクトル変換関数
# ----------------------
def video_to_vec(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        print(f"フレーム数: {total_frames}")
        if total_frames < n_frames:
            print(f"⚠️ フレーム数が少ない: {path}（{total_frames}フレーム）")
            return None

        idx = np.linspace(0, total_frames - 1, n_frames).astype(np.int64)
        frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0

        inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        z = outputs.last_hidden_state[:, 0]  # [CLS] token
        return z.squeeze(0).cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ 読み込みエラー: {path} → {e}")
        return None


def video_to_vec_adaptive(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        print(f"フレーム数: {total_frames}")
        if total_frames == 0:
            print(f"⚠️ フレーム数ゼロ: {path}")
            return None

        # 全フレーム取得
        frames = vr.get_batch(range(total_frames))  # shape: [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

        # T次元に adaptive average pooling を適用してフレーム数を調整
        frames = frames.unsqueeze(0)  # → [1, T, C, H, W]
        frames = frames.permute(0, 2, 3, 4, 1)  # → [1, C, H, W, T]
        pooled = F.adaptive_avg_pool3d(frames, (frames.shape[2], frames.shape[3], n_frames))
        frames = pooled.permute(0, 4, 1, 2, 3).squeeze(0)  # → [n_frames, C, H, W]

        inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        z = outputs.last_hidden_state[:, 0]  # [CLS]
        return z.squeeze(0).cpu().numpy().tolist()
    except Exception as e:
        print(f"❌ 読み込みエラー: {path} → {e}")
        return None

# ----------------------
# ④ ベクトル抽出と保存
# ----------------------
vectors = {}
total_count = 0
skipped_count = 0

with open('labels.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_count += 1
        rel_path = row['video_path'].replace('\\', '/')
        full_path = os.path.join('./', rel_path)

        vec = video_to_vec(full_path)
        # vec = video_to_vec_adaptive(full_path)

        if vec is not None:
            vectors[rel_path] = vec
        else:
            skipped_count += 1

with open('./exec/video_vectors.json', 'w', encoding='utf-8') as f:
    json.dump(vectors, f)

# ----------------------
# ⑤ 結果のレポート出力
# ----------------------
print(f"\n📊 データ数レポート")
print(f"🗂️ filtered_labels.csv の件数        : {total_count}")
print(f"✅ ベクトル抽出に成功した件数       : {len(vectors)}")
print(f"⚠️ スキップされた件数（失敗/短すぎ）: {skipped_count}")

