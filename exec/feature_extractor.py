import os
import json
import numpy as np
import torch
import decord
import pandas as pd
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel
import time
import gc


# ------------------------
# VideoMAE バージョン選択
# ------------------------
VMAE_VERSION = "base"

device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

if VMAE_VERSION == "base":
    print("✅ Using: MCG-NJU/videomae-base")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
    MODEL_SUFFIX = "_base"
else:
    raise ValueError("❌ Unsupported model version")


# --------------------------------------------------
# 1動画→複数ベクトル（ウィンドウごとに保存）
# --------------------------------------------------
def video_to_vec_sliding_list_save(path, out_dir, n_frames=16, stride=1):
    """
    CLS ベクトルを溜め込まずに、1ウィンドウごとに .npy 保存。
    メモリ使用量は常に O(hidden_dim) のみ。
    """

    try:
        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames < n_frames:
            return 0

        window_idx = 0
        saved_count = 0

        for start in range(0, total_frames - n_frames + 1, stride):

            # すでに保存済みならスキップ（再開対応）
            save_path = os.path.join(out_dir, f"{window_idx:03d}.npy")
            if os.path.exists(save_path):
                window_idx += 1
                continue

            # フレーム読み込み
            idx = list(range(start, start + n_frames))
            frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0

            # 変換
            inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls = outputs.last_hidden_state[:, 0].cpu().numpy().squeeze()

            # 保存
            np.save(save_path, cls)
            saved_count += 1
            window_idx += 1

            # 明示的にメモリ開放
            del frames, inputs, outputs, cls
            torch.cuda.empty_cache()
            gc.collect()

        return saved_count

    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return 0


# --------------------------------------------------
# 1動画→1ベクトル の pooling モード
# --------------------------------------------------
def video_to_vec_pooling_save(path, save_path, n_frames=16, stride=1):
    """従来の sliding → 平均化 → 1ベクトル保存"""
    try:
        if os.path.exists(save_path):
            return True

        vr = decord.VideoReader(path)
        total_frames = len(vr)
        if total_frames < n_frames:
            return False

        cls_vectors = []

        for start in range(0, total_frames - n_frames + 1, stride):
            idx = list(range(start, start + n_frames))
            frames = vr.get_batch(idx).permute(0, 3, 1, 2).float() / 255.0
            inputs = processor(list(frames), return_tensors="pt", do_rescale=False).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                cls_vectors.append(outputs.last_hidden_state[:, 0].cpu())

            del frames, inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()

        if not cls_vectors:
            return False

        all_vecs = torch.cat(cls_vectors, dim=0).unsqueeze(0).permute(0, 2, 1)
        pooled = F.adaptive_avg_pool1d(all_vecs, 1).squeeze().numpy()

        np.save(save_path, pooled)
        return True

    except:
        return False


# --------------------------------------------------
# main
# --------------------------------------------------
def main(csv_file, test=False, datatype='animalkingdom', stride=1, modes=None):

    df = pd.read_csv(csv_file)
    df["species"] = df["species"].str.strip()
    df["parent_class"] = df["parent_class"].str.strip().str.lower()
    df["action"] = df["action"].str.strip()

    for _, row in df.iterrows():
        rel_path = row["video_path"].replace("\\", "/")
        video_path = os.path.join("./", rel_path)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_base = f"./vector/{datatype}/{video_name}"
        os.makedirs(out_base, exist_ok=True)

        print(f"\n🎞 Processing {video_name}")

        # ----------------------------
        # モードごとに保存
        # ----------------------------
        if "sliding_list" in modes:
            out_dir = os.path.join(out_base, "sliding_list")
            os.makedirs(out_dir, exist_ok=True)
            count = video_to_vec_sliding_list_save(video_path, out_dir, stride=stride)
            print(f"  ➜ sliding_list 保存: {count} files")

        if "sliding" in modes:
            save_path = os.path.join(out_base, f"{video_name}_pooling.npy")
            ok = video_to_vec_pooling_save(video_path, save_path, stride=stride)
            print(f"  ➜ pooling 保存: {ok}")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":

    csv_list = [
        "./label/animalkingdom/train/labels.csv",
    ]

    modes = ["sliding_list", "sliding"]  # ← ★ 配列で複数選択

    for csv in csv_list:
        parts = os.path.normpath(csv).split(os.sep)
        datatype = "animalkingdom"
        test = "test" in parts

        print(f"\n🚀 CSV: {csv}")
        main(csv, test=test, datatype=datatype, stride=1, modes=modes)
