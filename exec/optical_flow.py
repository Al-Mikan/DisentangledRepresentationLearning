import os
import time  # 追加
from types import SimpleNamespace
import pandas as pd
import torch
import numpy as np
import decord
from PIL import Image
from RAFT.core.utils.utils import InputPadder
from torchvision import transforms
from RAFT.core.raft import RAFT
import shutil

total, used, free = shutil.disk_usage(".")
print("Total: %.2f GB" % (total / (2**30)))
print("Used: %.2f GB" % (used / (2**30)))
print("Free: %.2f GB" % (free / (2**30)))

exit()

args = SimpleNamespace()
args.small = False  # or True
args.mixed_precision = True
args.dropout = 0.0
args.alternate_corr = False

# RAFTのモデルロード
raft_model = RAFT(args)
state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
raft_model.load_state_dict(new_state_dict)
raft_model = raft_model.eval().cuda()

def extract_flow(video_path, out_dir, batch_size=3):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_out_dir = os.path.join(out_dir, video_name)
    out_path = os.path.join(video_out_dir, f"{video_name}.npy")

    if os.path.isfile(out_path):
        print(f"⏩ 既に存在するためスキップ: {out_path}")
        return out_path

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames < 2:
        print(f"⚠️ スキップ: {video_path}（フレーム不足）")
        return None

    os.makedirs(video_out_dir, exist_ok=True)

    flow_list = []
    transform = transforms.Compose([transforms.ToTensor()])

    pairs = []
    for i in range(total_frames - 1):
        img1 = vr[i].asnumpy()
        img2 = vr[i+1].asnumpy()
        img1_t = transform(Image.fromarray(img1))
        img2_t = transform(Image.fromarray(img2))
        pairs.append( (img1_t, img2_t) )

    # === バッチで回す ===
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        img1_batch = torch.stack([p[0] for p in batch]).cuda()
        img2_batch = torch.stack([p[1] for p in batch]).cuda()

        with torch.no_grad():
            padder = InputPadder(img1_batch.shape)
            image1, image2 = padder.pad(img1_batch, img2_batch)
            _, flow_up = raft_model(image1, image2, iters=3, test_mode=True)
            flow_np = flow_up.permute(0, 2, 3, 1).cpu().numpy()
            flow_np = np.stack([padder.unpad(flow_np[j]) for j in range(flow_np.shape[0])])
            flow_list.extend(flow_np)

    flow_array = np.stack(flow_list)
    out_path = os.path.join(video_out_dir, f"{video_name}.npy")
    np.save(out_path, flow_array)
    print(f"✅ Flow saved: {out_path}")

    return out_path

if __name__ == "__main__":
    csv_files = [
        "./label/animalkingdom/train/labels.csv",
        "./label/animalkingdom/test/labels_test_24fps.csv",
        "./label/animalkingdom/test/labels_test.csv",
    ]

    # === 存在しないファイルを弾く ===
    valid_csv_files = []
    for path in csv_files:
        if os.path.isfile(path):
            valid_csv_files.append(path)
        else:
            print(f"⚠️ CSVファイルが存在しません: {path}")

    found_one = False
    for csv in valid_csv_files:
        print(f"🔍 処理中のCSV: {csv}")
        parts = os.path.normpath(csv).split(os.sep)
        csv_filename = os.path.basename(csv)

        if 'label' in parts:
            labels_idx = parts.index('label')
            datatype = parts[labels_idx + 1]
            is_test = 'test' in parts
            is24fps = "_24fps" in csv_filename

            print(f"📂 datatype: {datatype}")
            print(f"🔍 is test: {is_test}")
            print(f"🎞 is 24fps: {is24fps}")

            output_dir = f"./flows/{datatype}/{'test' if is_test else 'train'}"
            os.makedirs(output_dir, exist_ok=True)

            df = pd.read_csv(csv)
            df["species"] = df["species"].str.strip()
            df["parent_class"] = df["parent_class"].str.strip().str.lower()
            df["action"] = df["action"].str.strip()

            # === CSVを「配列化」して一括ループ ===
            video_paths = df["video_path"].str.replace("\\", "/").tolist()
            start_time_label = time.time()

            for rel_path in video_paths:
                full_path = os.path.join('./', rel_path)
                
                # ⏱️ 開始
                start_time = time.time()
                
                out_npy = extract_flow(full_path, output_dir)
                
                elapsed = time.time() - start_time
                elapsed_label = time.time() - start_time_label
                print(f"⏱️ 処理時間: {elapsed:.2f} 秒")
                print(f"⏱️ ラベル経過時間: {elapsed_label:.2f} 秒")

