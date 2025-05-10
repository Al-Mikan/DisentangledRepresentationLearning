# feature_extractor.py

import os
import csv
import json
import numpy as np
import torch
import decord
from transformers import VideoMAEImageProcessor, VideoMAEModel

# ----------------------
# 初期設定
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

# モデルとプロセッサの読み込み（Hugging Face）
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

# ----------------------
# フィルター条件の設定
# ----------------------
target_actions = {"Eating", "Running", "Playing"}
mammal_species = set()

with open('Animal.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Parent Class'].strip().lower() == 'mammal':
            mammal_species.add(row['Animal'].strip())

print(f"🦣 対象のmammal種: {len(mammal_species)} 種")
print(f"🎯 対象行動: {target_actions}")

# ----------------------
# 動画をベクトルに変換する関数
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

        inputs = processor(list(frames), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        z = outputs.last_hidden_state[:, 0]  # [CLS] token (1, 768)

        return z.squeeze(0).cpu().numpy().tolist()

    except Exception as e:
        print(f"❌ 読み込みエラー: {path} → {e}")
        return None

# ----------------------
# ベクトル抽出と保存
# ----------------------
vectors = {}

with open('labels.csv', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rel_path = row['video_path'].replace('\\', '/')
        full_path = os.path.join('./', rel_path)

        vec = video_to_vec(full_path)
        if vec is not None:
            vectors[rel_path] = vec

with open('video_vectors.json', 'w', encoding='utf-8') as f:
    json.dump(vectors, f)

print(f"✅ 完了：{len(vectors)} 件のベクトルを保存しました。")
