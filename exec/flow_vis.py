import os
import numpy as np
import torch
import cv2
from PIL import Image
from decord import VideoReader
from torchvision import transforms

from RAFT.core.utils.utils import InputPadder
from RAFT.core.raft import RAFT
from RAFT.core.utils.flow_viz import flow_to_image

def extract_flow_and_save_vis(
    video_path, out_dir,
    batch_size=2, n_save_png=5
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    vr = VideoReader(video_path)
    total_frames = len(vr)
    if total_frames < 2:
        print(f"⚠️ フレーム不足: {video_path}")
        return

    transform = transforms.Compose([transforms.ToTensor()])

    pairs = []
    for i in range(total_frames - 1):
        img1 = vr[i].asnumpy()
        img2 = vr[i+1].asnumpy()
        img1_t = transform(Image.fromarray(img1))
        img2_t = transform(Image.fromarray(img2))
        pairs.append((img1_t, img2_t))

    flow_list = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        img1_batch = torch.stack([p[0] for p in batch]).cuda()
        img2_batch = torch.stack([p[1] for p in batch]).cuda()

        with torch.no_grad():
            padder = InputPadder(img1_batch.shape)
            image1, image2 = padder.pad(img1_batch, img2_batch)
            _, flow_up = raft_model(image1, image2, iters=3, test_mode=True)

            for j in range(flow_up.shape[0]):
                f = padder.unpad(flow_up[j])
                f = f.permute(1, 2, 0).cpu().numpy()
                flow_list.append(f)

        del img1_batch, img2_batch, image1, image2, flow_up
        torch.cuda.empty_cache()

    flow_array = np.stack(flow_list)

    # === 可視化PNG保存 ===
    rgb_frames = []
    for i in range(flow_array.shape[0]):
        img = flow_to_image(flow_array[i])                       # [H,W,3] uint8
        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA) 
        rgb_frames.append(img)

    save_vis_dir = os.path.join(out_dir, "flow_vis")
    os.makedirs(save_vis_dir, exist_ok=True)
    for idx in range(min(n_save_png, len(rgb_frames))):
        img = rgb_frames[idx]
        save_path = os.path.join(save_vis_dir, f"flow_vis_{idx:03d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ {video_name}: フロー可視化PNG {min(n_save_png, len(rgb_frames))}枚を保存しました")

# --- 使い方例 ---
if __name__ == "__main__":
    # RAFTロードも忘れずに
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.small = False
    args.mixed_precision = True
    args.dropout = 0.0
    args.alternate_corr = False
    global raft_model
    raft_model = RAFT(args)
    state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    raft_model.load_state_dict(new_state_dict)
    raft_model = raft_model.eval().cuda()

    # 動画パスと出力先を指定
    extract_flow_and_save_vis("./video/animalkingdom/ABNMGXGD.mp4", "sample_img", n_save_png=5)
