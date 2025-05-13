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

# ----------------------
# ① CSV読み込みとmammal種フィルタ（100件以上）
# ----------------------
df = pd.read_csv("labels.csv")
df["species"] = df["species"].str.strip()
df["parent_class"] = df["parent_class"].str.strip().str.lower()
df["action"] = df["action"].str.strip()

mammals = df[df["parent_class"] == "mammal"]
species_counts = Counter(mammals["species"])

action_counts = Counter(mammals["action"])
valid_actions = {a for a, c in action_counts.items() if c >= 100}

print(f"✅ 100件以上ある行動: {len(valid_actions)} 種類")
print("🎯 対象行動一覧:")
for a in sorted(valid_actions):
    print(f"  - {a}")

filtered_df = mammals[mammals["action"].isin(valid_actions)]

filtered_df.to_csv("filtered_labels.csv", index=False)
print(f"✅ filtered_labels.csv を {len(filtered_df)} 件で保存しました。")

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
        if total_frames < n_frames:
            print(f"⚠️ フレーム数が少ない: {path}（{total_frames}フレーム）")
            return None

        idx = np.linspace(0, total_frames - 1, n_frames).astype(np.int64)
        frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0  # (n_frames, C, H, W)

        inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        z = outputs.last_hidden_state[:, 0]  # [CLS] token
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

with open('filtered_labels.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_count += 1
        rel_path = row['video_path'].replace('\\', '/')
        full_path = os.path.join('./', rel_path)

        vec = video_to_vec(full_path)
        if vec is not None:
            vectors[rel_path] = vec
        else:
            skipped_count += 1

with open('video_vectors.json', 'w', encoding='utf-8') as f:
    json.dump(vectors, f)

# ----------------------
# ⑤ 結果のレポート出力
# ----------------------
print(f"\n📊 データ数レポート")
print(f"🗂️ filtered_labels.csv の件数        : {total_count}")
print(f"✅ ベクトル抽出に成功した件数       : {len(vectors)}")
print(f"⚠️ スキップされた件数（失敗/短すぎ）: {skipped_count}")

