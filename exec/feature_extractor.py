import os
import numpy as np
import torch
import decord
import pandas as pd
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel
import time

# ------------------------
# VideoMAE version
# ------------------------
VMAE_VERSION = "base"
device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

if VMAE_VERSION == "base":
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()
else:
    raise ValueError("Unsupported version")


# --------------------------------------------------
# ① 1動画 → 1ベクトル
# --------------------------------------------------
def video_to_vec_pooling(path, n_frames=16, stride=1):
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
        return pooled.cpu().numpy()

    except Exception as e:
        print(f"Error pooling {path}: {e}")
        return None


# --------------------------------------------------
# ② 1動画 → sliding window
# --------------------------------------------------
def video_to_vec_sliding_list(path, n_frames=16, stride=1):
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

        return torch.cat(cls_vectors, dim=0).cpu().numpy()

    except Exception as e:
        print(f"Error sliding {path}: {e}")
        return None


# --------------------------------------------------
# main（★mode 配列対応＋再開処理あり）
# --------------------------------------------------
def main(csv_file, test=False, datatype="animalkingdom", modes=["sliding_list"], stride=1):

    df = pd.read_csv(csv_file)
    df["video_path"] = df["video_path"].str.replace("\\", "/")

    for mode in modes:
        print(f"\n==============================")
        print(f"🔥 MODE: {mode}")
        print(f"==============================\n")

        for _, row in df.iterrows():
            rel_path = row["video_path"]
            full_path = os.path.join("./", rel_path)

            video_stem = os.path.splitext(os.path.basename(rel_path))[0]
            out_dir = f"./vector/{datatype}/{video_stem}"
            os.makedirs(out_dir, exist_ok=True)

            print(f"🎞 Processing: {rel_path}")

            # ----------------------
            # sliding_list (複数ベクトル)
            # ----------------------
            if mode == "sliding_list":
                # 既に出ている npy を数える → 再開処理
                existing = sorted([f for f in os.listdir(out_dir) if f.startswith(video_stem + "_") and f.endswith(".npy")])
                if len(existing) > 0:
                    print(f"   ⏩ Skip (already exists: {len(existing)} vectors)")
                    continue

                vecs = video_to_vec_sliding_list(full_path, stride=stride)
                if vecs is None:
                    print("   ⚠️ Failed")
                    continue

                for i, v in enumerate(vecs):
                    out_path = f"{out_dir}/{video_stem}_{i:03d}.npy"
                    np.save(out_path, v)
                print(f"   ✅ Saved {len(vecs)} chunks")

            # ----------------------
            # sliding → pooling（1ベクトル）
            # ----------------------
            elif mode == "sliding":
                out_path = f"{out_dir}/{video_stem}_pooling.npy"
                if os.path.exists(out_path):
                    print("   ⏩ Skip (already exists)")
                    continue

                v = video_to_vec_pooling(full_path, stride=stride)
                if v is None:
                    print("   ⚠️ Failed")
                    continue

                np.save(out_path, v)
                print("   ✅ Saved pooling vector")

            else:
                raise ValueError(f"Unknown mode: {mode}")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    stride = 1
    modes = ["sliding", "sliding_list"]   # ← ★複数モードに対応

    csv_list = [
        "./label/animalkingdom/train/labels.csv",
    ]

    for csv in csv_list:
        print(f"\n=== Running on {csv} ===")
        main(csv, datatype="animalkingdom", modes=modes, stride=stride)
