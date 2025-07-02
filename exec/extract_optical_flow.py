import os
from turtle import pd
from types import SimpleNamespace
import pandas as pd
import torch
import numpy as np
import decord
from PIL import Image
from RAFT.core.utils.utils import InputPadder
from torchvision import transforms
from RAFT.core.raft import RAFT
from torchvision import transforms

args = SimpleNamespace()
args.small = False  # or True
args.mixed_precision = False
args.dropout = 0.0
args.alternate_corr = False

# RAFTのモデルロード
raft_model = RAFT(args)
state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
raft_model.load_state_dict(new_state_dict)
raft_model = raft_model.eval().cuda()


def extract_flow(video_path, out_dir):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames < 2:
        print(f"⚠️ スキップ: {video_path}（フレーム不足）")
        return

    os.makedirs(out_dir, exist_ok=True)
    flow_list = []

    transform = transforms.Compose([
    transforms.ToTensor(),
    ])

    for i in range(total_frames - 1):
        img1 = vr[i].asnumpy()
        img2 = vr[i+1].asnumpy()
        img1 = transform(Image.fromarray(img1)).unsqueeze(0).cuda()
        img2 = transform(Image.fromarray(img2)).unsqueeze(0).cuda()

        with torch.no_grad():
            padder = InputPadder(img1.shape)
            image1, image2 = padder.pad(img1, img2)
            _, flow_up = raft_model(image1, image2, iters=20, test_mode=True)
            flow_np = flow_up[0].permute(1, 2, 0).cpu().numpy()  # [H,W,2]
            flow_list.append(flow_np)

    flow_array = np.stack(flow_list)
    out_path = os.path.join(out_dir, os.path.basename(video_path).replace('.mp4', '.npy'))
    np.save(out_path, flow_array)
    print(f"✅ Flow saved: {out_path}")

if __name__ == "__main__":

    csv= "./label/animalkingdom/test/labels_test.csv"
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
    for _, row in df.iterrows():
        rel_path = row['video_path'].replace('\\', '/')
        full_path = os.path.join('./', rel_path)
        extract_flow(full_path, output_dir)
