import os
import sys
import json
import numpy as np
import torch
import decord
from transformers import VideoMAEImageProcessor, VideoMAEModel
import gc

# ========= 受け取る引数 =========
video_path = os.path.abspath(sys.argv[1])
out_base   = os.path.abspath(sys.argv[2])
stride     = int(sys.argv[3])

print(f"\n🎞 Worker processing {os.path.basename(video_path)}\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
decord.bridge.set_bridge("torch")

# モデル読み込み
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
# 統合処理関数 (Sliding Save + Pooling)
# ========================================
def process_video_features(video_path, out_base, n_frames=16, stride=1):
    # 出力先設定
    sliding_dir = os.path.join(out_base, "sliding_list")
    os.makedirs(sliding_dir, exist_ok=True)
    
    pool_path = os.path.join(out_base, "avg_pooling.npy")
    progress_path = os.path.join(sliding_dir, "progress.json")
    
    progress = load_progress(progress_path)

    try:
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        # ウィンドウ数の計算
        if total_frames < n_frames:
            print(f"⚠ Video too short ({total_frames} frames). Skipping.")
            return

        total_windows = (total_frames - n_frames) // stride + 1

        progress.setdefault("windows_total", total_windows)
        progress.setdefault("video", os.path.basename(video_path))
        save_progress(progress_path, progress)

        last_success = progress.get("last_success", -1)
        start_window = last_success + 1

        # Pooling用の累積変数 (途中再開の場合は考慮が必要だが、今回は簡易的に)
        # ※ 厳密に途中再開でPoolingを計算するには、既存のnpyを読み込む必要がありますが
        #    ここでは「Sliding保存」を優先し、Poolingは最後に全npyから計算する方式にします。
        
        # ========= 1. Sliding Loop (個別保存) =========
        for win in range(start_window, total_windows):
            save_path = os.path.join(sliding_dir, f"{win:03d}.npy")
            
            # 既にファイルがあるならスキップ（念のため）
            if os.path.exists(save_path):
                continue

            try:
                # Frame Load
                idx = list(range(win * stride, win * stride + n_frames))
                frames = vr.get_batch(idx).permute(0,3,1,2).float() / 255.0 # [T, C, H, W]
                
                # Processor expects list of numpy [H, W, C]
                frames_np = frames.permute(0,2,3,1).cpu().numpy()
                frames_list = [frames_np[i] for i in range(frames_np.shape[0])]

                # Preprocess
                inputs = processor(frames_list, return_tensors="pt", do_rescale=False).to(device)
                
                # Inference
                with torch.no_grad():
                    out = model(**inputs)
                    # [Batch, Hidden] -> [Hidden]
                    cls_vector = out.last_hidden_state[:, 0].cpu().numpy().squeeze()

                # Save
                np.save(save_path, cls_vector)
                
                # Progress Update
                progress["last_success"] = win
                if win % 10 == 0: # 毎回保存すると遅いので間引く
                    save_progress(progress_path, progress)

            except Exception as e:
                print(f"⚠ Error at window {win}: {e}")
                continue
            
            # Memory Cleanup
            del frames, inputs, out, cls_vector
            # gc.collect() # 毎回呼ぶと遅いので、VRAMがカツカツでなければ外すか頻度を下げる

        # 最後の進捗保存
        save_progress(progress_path, progress)

        # ========= 2. Pooling Calculation (保存済みファイルから計算) =========
        # メモリに全ベクトルを持たずに計算します（省メモリ）
        
        if os.path.exists(pool_path):
            print("ℹ️ Pooling file already exists. Skipping.")
            return

        print("🔄 Calculating pooling from saved files...")
        sum_vec = None
        count = 0
        
        # 保存された全npyファイルを走査
        npy_files = sorted([f for f in os.listdir(sliding_dir) if f.endswith(".npy")])
        
        for f in npy_files:
            try:
                vec = np.load(os.path.join(sliding_dir, f)) # [768]
                if sum_vec is None:
                    sum_vec = vec.astype(np.float64) # 精度維持のためfloat64
                else:
                    sum_vec += vec
                count += 1
            except:
                pass
        
        if count > 0 and sum_vec is not None:
            avg_vec = (sum_vec / count).astype(np.float32)
            np.save(pool_path, avg_vec)
            print(f"✅ Pooling saved (averaged {count} vectors)")
        else:
            print("⚠ No vectors found for pooling.")

    except Exception as e:
        print(f"❌ Critical Error: {e}")
        import traceback
        traceback.print_exc()

# ========= Run =========
process_video_features(video_path, out_base, stride=stride)
print("✅ Worker finished\n")