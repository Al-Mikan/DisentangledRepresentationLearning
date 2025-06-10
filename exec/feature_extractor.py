import os
import json
import numpy as np
import torch
import decord
import pandas as pd
import argparse
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel

# --- 引数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['simple', '3d', '1d'], default='simple')
parser.add_argument('--csv', type=str, default='labels.csv')
parser.add_argument('--test', action='store_true', help='テスト用出力に切り替える')
args = parser.parse_args()

# --- 出力ファイルパスを mode に応じて決定 ---
output_path = {
    ('simple', False): './exec/vectors_simple.json',
    ('3d', False): './exec/vectors_adaptive3d.json',
    ('1d', False): './exec/vectors_adaptive1d.json',
    ('simple', True): './exec/vectors_simple_test.json',
    ('3d', True): './exec/vectors_adaptive3d_test.json',
    ('1d', True): './exec/vectors_adaptive1d_test.json'
}[(args.mode, args.test)]

# ----------------------
# 初期設定とモデル読み込み
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

# ----------------------
# 動画 → ベクトル変換関数（simple）
# ----------------------
def video_to_vec(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames < n_frames:
            return None
        idx = np.linspace(0, total_frames - 1, n_frames).astype(np.int64)
        frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0
        inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0].squeeze(0).cpu().numpy().tolist()
    except:
        return None

# ----------------------
# 動画 → ベクトル変換関数（adaptive）
# ----------------------
def video_to_vec_adaptive(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames == 0:
            return None
        frames = vr.get_batch(range(total_frames)).permute(0, 3, 1, 2).float() / 255.0
        frames = frames.unsqueeze(0).permute(0, 2, 3, 4, 1)
        pooled = F.adaptive_avg_pool3d(frames, (frames.shape[2], frames.shape[3], n_frames))
        frames = pooled.permute(0, 4, 1, 2, 3).squeeze(0)
        inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0].squeeze(0).cpu().numpy().tolist()
    except:
        return None

# ----------------------
# 動画 → ベクトル変換関数（chunked）
# ----------------------
def video_to_vec_chunked(path, n_frames=16):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames < n_frames:
            return None
        cls_vectors = []
        for start in range(0, total_frames - n_frames + 1, n_frames):
            idx = list(range(start, start + n_frames))
            frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0
            inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_vectors.append(outputs.last_hidden_state[:, 0])
        if not cls_vectors:
            return None
        all_vecs = torch.cat(cls_vectors, dim=0).unsqueeze(0).permute(0, 2, 1)
        pooled = F.adaptive_avg_pool1d(all_vecs, 1).squeeze()
        return pooled.cpu().numpy().tolist()
    except:
        return None

# ----------------------
# ベクトル抽出
# ----------------------
df = pd.read_csv(args.csv)
df["species"] = df["species"].str.strip()
df["parent_class"] = df["parent_class"].str.strip().str.lower()
df["action"] = df["action"].str.strip()

vectors = {}
total_count = 0
skipped_count = 0

for _, row in df.iterrows():
    total_count += 1
    rel_path = row['video_path'].replace('\\', '/')
    full_path = os.path.join('./', rel_path)

    # プログレス表示
    print(f"[{total_count}/{len(df)}] 処理中: {rel_path}", flush=True)

    if args.mode == 'simple':
        vec = video_to_vec(full_path)
    elif args.mode == 'adaptive':
        vec = video_to_vec_adaptive(full_path)
    else:
        vec = video_to_vec_chunked(full_path)

    if vec is not None:
        vectors[rel_path] = vec
    else:
        skipped_count += 1

# ----------------------
# 保存
# ----------------------
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(vectors, f)

# ----------------------
# レポート出力
# ----------------------
print(f"\n📊 データ数レポート")
print(f"🗂️ CSV件数        : {total_count}")
print(f"✅ 成功            : {len(vectors)}")
print(f"⚠️ 失敗・スキップ : {skipped_count}")
