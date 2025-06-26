import os
import json
import numpy as np
import torch
import decord
import pandas as pd
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel
import time  # ← これを追加


# ------------------------
# モデルとProcessorはグローバルで初期化
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()

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

def video_to_vec_sliding(path, n_frames=16, stride=1):
    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames < n_frames:
            return None
        cls_vectors = []
        for start in range(0, total_frames - n_frames + 1, stride):
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


def main(mode, csv_file, test=False, is24fps=True, stride=1):
    start_time = time.time()
    suffix = '_test' if test else ''
    fps_suffix = '_24fps' if is24fps else ''
    base_name = {
        'simple': 'vectors_simple',
        '3d': 'vectors_adaptive3d',
        '1d': 'vectors_adaptive1d',
        'sliding': 'vectors_sliding',
    }[mode]

    
    csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
    output_path = f'./exec/{base_name}_{csv_stem}{suffix}{fps_suffix}.json'

    print(f"✅ 出力パス: {output_path}")

    df = pd.read_csv(csv_file)
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

        elapsed = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"[{total_count}/{len(df)}] 処理中: {rel_path} | 経過: {elapsed_str}", flush=True)

        if mode == 'simple':
            vec = video_to_vec(full_path)
        elif mode == '3d':
            vec = video_to_vec_adaptive(full_path)
        elif mode == '1d':
            vec = video_to_vec_chunked(full_path)
        elif mode == 'sliding':
            vec = video_to_vec_sliding(full_path, n_frames=16, stride=stride)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if vec is not None:
            vectors[rel_path] = vec
        else:
            skipped_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vectors, f)

    print(f"\n📊 データ数レポート")
    print(f"🗂️ CSV件数        : {total_count}")
    print(f"✅ 成功            : {len(vectors)}")
    print(f"⚠️ 失敗・スキップ : {skipped_count}")


if __name__ == "__main__":
    modes = ['sliding']
    stride = 1
    csv = 'ucf_labels.csv'  # 固定で使用
    test = False            # テスト用ではない
    is24fps = False         # 24fpsでもない（必要に応じてTrueに）

    for mode in modes:
        print(f"\n🚀 実行中: mode={mode}, test={test}, csv={csv}")
        main(mode, csv, test=test, is24fps=is24fps, stride=stride)
