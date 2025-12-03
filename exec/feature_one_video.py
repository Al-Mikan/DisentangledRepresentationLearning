import os
import sys
import json
import numpy as np
import torch
import decord
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor, VideoMAEModel
import gc


# ========= 受け取る引数 =========
video_path = os.path.abspath(sys.argv[1])
out_base   = os.path.abspath(sys.argv[2])
stride     = int(sys.argv[3])

print(f"\n🎞 Worker processing {os.path.basename(video_path)}\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device).eval()


# ========= Utility =========
def load_progress(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_progress(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ========================================
# sliding_list（最も安全な Windows 用）
# ========================================
def process_sliding_list(video_path, out_dir, n_frames=16, stride=1):
    os.makedirs(out_dir, exist_ok=True)

    progress_path = os.path.join(out_dir, "progress.json")
    progress = load_progress(progress_path)

    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        total_windows = (total_frames - n_frames) // stride + 1

        progress.setdefault("windows_total", total_windows)
        progress.setdefault("video", os.path.basename(video_path))
        save_progress(progress_path, progress)

        last_success = progress.get("last_success", -1)
        start_window = last_success + 1

        # 🔥前回途中で壊れてる可能性のある npy を削除
        broken_file = os.path.join(out_dir, f"{start_window:03d}.npy")
        if os.path.exists(broken_file):
            print(f"⚠ Delete broken file: {broken_file}")
            os.remove(broken_file)

        # ========= メインループ =========
        for win in range(start_window, total_windows):
            save_path = os.path.join(out_dir, f"{win:03d}.npy")

            # Frame Load
            try:
                idx = list(range(win * stride, win * stride + n_frames))
                frames = vr.get_batch(idx).permute(0,3,1,2).float() / 255.0
            except Exception as e:
                print(f"⚠ Frame read error at window {win}: {e}")
                continue

            # (T, C, H, W) → (T, H, W, C)
            frames_np = frames.permute(0,2,3,1).cpu().numpy()
            frames_list = [frames_np[i] for i in range(frames_np.shape[0])]

            try:
                inputs = processor(frames_list, return_tensors="pt", do_rescale=False).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                    cls = out.last_hidden_state[:,0].cpu().numpy().squeeze()
            except Exception as e:
                print(f"⚠ VideoMAE forward failed at window {win}: {e}")
                continue

            # Save atomic
            try:
                np.save(save_path, cls)
            except Exception as e:
                print(f"⚠ Save error at window {win}: {e}")
                continue

            # Update progress
            progress["last_success"] = win
            save_progress(progress_path, progress)

            del frames, out, inputs, cls
            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error sliding_list: {e}")


# ========================================
# pooling（壊れやすいので毎回作り直し）
# ========================================
def process_pooling(video_path, save_path, n_frames=16, stride=1):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"⚠ Delete previous pooling file: {save_path}")
        os.remove(save_path)

    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        total_windows = (total_frames - n_frames) // stride + 1

        cls_vectors = []

        for win in range(total_windows):
            try:
                idx = list(range(win * stride, win * stride + n_frames))
                frames = vr.get_batch(idx).permute(0,3,1,2).float() / 255.0
            except:
                print(f"⚠ Frame read error at window {win}")
                continue

            try:
                frames_np = frames.permute(0,2,3,1).cpu().numpy()
                frames_list = [frames_np[i] for i in range(frames_np.shape[0])]
                inputs = processor(frames_list, return_tensors="pt", do_rescale=False).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                    cls_vectors.append(out.last_hidden_state[:,0].cpu())
            except:
                print(f"⚠ VideoMAE forward error at window {win}")
                continue

            del frames, out, inputs
            gc.collect()
            torch.cuda.empty_cache()

        if not cls_vectors:
            print("⚠ No vectors collected for pooling")
            return

        all_vecs = torch.cat(cls_vectors, dim=0).unsqueeze(0).permute(0,2,1)
        pooled = F.adaptive_avg_pool1d(all_vecs, 1).squeeze().numpy()

        np.save(save_path, pooled)

    except Exception as e:
        print(f"❌ Error pooling: {e}")


# ========= Run =========
sliding_dir = os.path.join(out_base, "sliding_list")
process_sliding_list(video_path, sliding_dir, stride=stride)

pool_path = os.path.join(out_base, "avg_pooling.npy")
process_pooling(video_path, pool_path, stride=stride)

print("✅ Worker finished\n")
