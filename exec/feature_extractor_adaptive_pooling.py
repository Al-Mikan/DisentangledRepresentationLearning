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
# ③ 動画 → ベクトル変換関数（16フレームずつ → 平均）
# ----------------------
def video_to_vec_chunked(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        print(f"📹 {path} - フレーム数: {total_frames}")

        if total_frames < n_frames:
            print(f"⚠️ フレーム数が少ないためスキップ: {total_frames}フレーム")
            return None

        cls_vectors = []

        for start in range(0, total_frames - n_frames + 1, n_frames):
            idx = list(range(start, start + n_frames))
            frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0  # [T, C, H, W]

            inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_vec = outputs.last_hidden_state[:, 0]  # [1, D]
                cls_vectors.append(cls_vec)

        if not cls_vectors:
            return None

        all_vecs = torch.cat(cls_vectors, dim=0).unsqueeze(0)  # [1, N, D]
        all_vecs = all_vecs.permute(0, 2, 1)  # [1, D, N]
        pooled = F.adaptive_avg_pool1d(all_vecs, 1).squeeze()  # [D]
        return pooled.cpu().numpy().tolist()

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

        vec = video_to_vec_chunked(full_path)
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
print(f"🗂️ labels.csv の件数        : {total_count}")
print(f"✅ ベクトル抽出に成功した件数       : {len(vectors)}")
print(f"⚠️ スキップされた件数（失敗/短すぎ）: {skipped_count}")
