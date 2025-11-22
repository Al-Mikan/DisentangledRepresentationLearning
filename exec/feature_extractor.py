import os
import json
import numpy as np
import torch
import decord
import pandas as pd
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel
import time


# ------------------------
# ✅ VideoMAE バージョン選択
# ------------------------
VMAE_VERSION = "base"   # "base", "v2-base", "v2-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

if VMAE_VERSION == "base":
    print("✅ Using: MCG-NJU/videomae-base")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
    MODEL_SUFFIX = "_base"

else:
    raise ValueError("❌ VMAE_VERSION must be one of: 'base', 'v2-base', 'v2-large'")


# --------------------------------------------------
# 🔵 1動画→1ベクトル（従来）
# --------------------------------------------------
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
                cls_vectors.append(outputs.last_hidden_state[:, 0])  # [1, hidden_dim]

        if not cls_vectors:
            return None

        # 平均化（1動画→1ベクトル）
        all_vecs = torch.cat(cls_vectors, dim=0).unsqueeze(0).permute(0, 2, 1)
        pooled = F.adaptive_avg_pool1d(all_vecs, 1).squeeze()
        return pooled.cpu().numpy().tolist()

    except:
        return None


# --------------------------------------------------
# 1動画→複数ベクトル（平均しない）
# --------------------------------------------------
def video_to_vec_sliding_list(path, n_frames=16, stride=1):
    """Sliding window の CLS ベクトル列をそのまま返す。
    戻り値:
        list[list[float]] ← [num_windows, hidden_dim]
    """
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
                cls_vectors.append(outputs.last_hidden_state[:, 0])  # [1, hidden_dim]

        if not cls_vectors:
            return None

        # [num_windows, hidden_dim]
        all_vecs = torch.cat(cls_vectors, dim=0)
        return all_vecs.cpu().numpy().tolist()

    except:
        return None


# --------------------------------------------------
# 🔵 main
# --------------------------------------------------
def main(csv_file, test=False, is24fps=True, stride=1, datatype='animalkingdom', mode='sliding'):
    start_time = time.time()

    exec_subdir = "test" if test else "train"
    os.makedirs(f"./vector/{datatype}/{exec_subdir}", exist_ok=True)

    fps_suffix = '_24fps' if is24fps else ''

    base_name = {
        'sliding': 'vectors_sliding',          # 1動画→1ベクトル
        'sliding_list': 'vectors_sliding_list',  # 1動画→複数ベクトル
    }

    output_path = f'./vector/{datatype}/{exec_subdir}/{base_name[mode]}{fps_suffix}{MODEL_SUFFIX}.json'
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

        # --- モード別 ---
        if mode == 'sliding':
            vec = video_to_vec_sliding(full_path, n_frames=16, stride=stride)

        elif mode == 'sliding_list':
            vec = video_to_vec_sliding_list(full_path, n_frames=16, stride=stride)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if vec is not None:
            vectors[rel_path] = vec
        else:
            skipped_count += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vectors, f)

    print("\n📊 データ数レポート")
    print(f"🗂️ CSV件数        : {total_count}")
    print(f"✅ 成功            : {len(vectors)}")
    print(f"⚠️ 失敗・スキップ : {skipped_count}")


# --------------------------------------------------
# 🔵 Entry
# --------------------------------------------------
if __name__ == "__main__":
    stride = 1
    mode = 'sliding_list'   # ← ★ ここだけ変更すればOK（既存に干渉しない）

    csv_list = [
        "./label/animalkingdom_split/train/labels.csv",
        "./label/animalkingdom_split/test/labels_test.csv",
    ]

    for csv in csv_list:
        parts = os.path.normpath(csv).split(os.sep)
        csv_filename = os.path.basename(csv)

        if 'label' in parts:
            labels_idx = parts.index('label')
            datatype = parts[labels_idx + 1]
            test = 'test' in parts
            is24fps = "_24fps" in csv_filename

            print(f"\n📂 datatype: {datatype}")
            print(f"🔍 is test: {test}")
            print(f"🎞 is 24fps: {is24fps}")
        else:
            raise ValueError("❌ CSV パスに 'labels' が含まれていません。")

        print(f"\n🚀 実行中: mode={mode}, test={test}, csv={csv}")
        main(csv, test=test, is24fps=is24fps, stride=stride, datatype=datatype, mode=mode)
