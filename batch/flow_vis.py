import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from decord import VideoReader
from torchvision import transforms
from types import SimpleNamespace

import os
import sys

# batch/a.py の場所
script_dir = os.path.dirname(os.path.abspath(__file__))

# project_root
project_root = os.path.dirname(script_dir)

# project_root を Python パスに追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# RAFT
from exec.RAFT.core.utils.utils import InputPadder
from exec.RAFT.core.raft import RAFT
from exec.RAFT.core.utils.flow_viz import flow_to_image


def _select_indices(total, n_save, frames_every=1):
    """
    サンプリングするフレームインデックスを返す。
    frames_every=30 → 30フレームごと
    frames_every=1  → 先頭から連続で
    n_save → 上限枚数
    """
    idxs = list(range(0, total, frames_every))
    return idxs[:n_save]


def _save_raw_frames(vr, indices, out_dir, resize=None, prefix="raw"):
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i in indices:
        if i < 0 or i >= len(vr):
            continue
        img = vr[i].asnumpy()  # RGB
        if resize is not None:
            w, h = resize
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_{i:06d}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved += 1
    return saved


def extract_and_save(
    video_path: str,
    out_dir: str,
    n_save_png: int = 10,         # 保存する最大枚数
    frames_every: int = 1,        # 何フレームごとにサンプリングするか
    outputs: tuple[str, ...] = ("raw",),  # "raw", "raft", "raft_center" から選ぶ
    frames_resize: tuple[int, int] | None = None,
    batch_size: int = 2,
    raft_iters: int = 3,
):
    """
    動画から raw / raft / raft_center を選択保存。
    - サンプリングは frames_every のみ
    - 保存上限は n_save_png
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    want_raw = "raw" in outputs
    want_raft = "raft" in outputs
    want_center = "raft_center" in outputs
    if not (want_raw or want_raft or want_center):
        print("⚠️ outputs が空です。raw/raft/raft_center から選んでください。")
        return

    # 出力先
    raft_dir = os.path.join(out_dir, "raft", video_name)
    raftc_dir = os.path.join(out_dir, "raft_center", video_name)
    raw_dir = os.path.join(out_dir, "raw", video_name)
    if want_raft: os.makedirs(raft_dir, exist_ok=True)
    if want_center: os.makedirs(raftc_dir, exist_ok=True)
    if want_raw: os.makedirs(raw_dir, exist_ok=True)

    # 動画読み込み
    vr = VideoReader(video_path)
    total_frames = len(vr)
    if total_frames < 2:
        print(f"⚠️ フレーム不足: {video_path}")
        return

    # 共通サンプリング
    raw_indices = _select_indices(total_frames, n_save_png, frames_every)
    # flowは (i, i+1) のペアなので最大は total_frames-2
    flow_indices = [i for i in raw_indices if i < total_frames - 1]

    # (A) raw 保存
    if want_raw:
        saved = _save_raw_frames(vr, raw_indices, raw_dir, resize=frames_resize, prefix="raw")
        print(f"🖼️ raw 保存: {saved} 枚 -> '{raw_dir}'")

    # (B) RAFT 推論（必要なときだけ）
    if want_raft or want_center:
        transform = transforms.Compose([transforms.ToTensor()])
        pairs = [
            (transform(Image.fromarray(vr[i].asnumpy())),
             transform(Image.fromarray(vr[i + 1].asnumpy())))
            for i in range(total_frames - 1)
        ]
        flow_list = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            img1_batch = torch.stack([p[0] for p in batch]).cuda()
            img2_batch = torch.stack([p[1] for p in batch]).cuda()
            with torch.no_grad():
                padder = InputPadder(img1_batch.shape)
                image1, image2 = padder.pad(img1_batch, img2_batch)
                _, flow_up = raft_model(image1, image2, iters=raft_iters, test_mode=True)
                for j in range(flow_up.shape[0]):
                    f = padder.unpad(flow_up[j]).permute(1, 2, 0).cpu().numpy()
                    flow_list.append(f)

        if not flow_list:
            print(f"⚠️ フロー計算失敗: {video_name}")
            return

        # raft 保存
        if want_raft:
            cnt = 0
            for rank, i in enumerate(flow_indices):
                rgb = flow_to_image(flow_list[i])
                save_path = os.path.join(raft_dir, f"flow_vis_{rank:03d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cnt += 1
            print(f"✅ raft/{video_name}: {cnt} 枚保存")

        # raft_center 保存
        if want_center:
            cnt = 0
            for rank, i in enumerate(flow_indices):
                flow = flow_list[i]
                mean_flow = np.mean(flow, axis=(0, 1))
                centered = flow - mean_flow
                rgb = flow_to_image(centered)
                save_path = os.path.join(raftc_dir, f"flow_vis_{rank:03d}.png")
                cv2.imwrite(save_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                cnt += 1
            print(f"✅ raft_center/{video_name}: {cnt} 枚保存")


# --- 実行例 ---
if __name__ == "__main__":
    args = SimpleNamespace(small=False, mixed_precision=True, dropout=0.0, alternate_corr=False)
    raft_model = RAFT(args)
    state_dict = torch.load("./exec/RAFT/models/raft-sintel.pth")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    raft_model.load_state_dict(new_state_dict)
    raft_model = raft_model.eval().cuda()

    video_path = "./video/elephant/clip_000.mp4"
    out_dir = "sample_img/elephant"

    # 例1: 30フレームごとにサンプリングし、先頭20枚まで raw と flow を保存
    # extract_and_save(
    #     video_path, out_dir,
    #     n_save_png=20,
    #     frames_every=30,
    #     outputs=("raw", "raft", "raft_center"),
    # )

    # print("-" * 50)
    # 例2: 先頭から10フレーム分だけ raw 保存（800x450 にリサイズ）
    extract_and_save(
        video_path, out_dir,
        n_save_png=20,
        frames_every=10,
        outputs=("raw", "raft", "raft_center"),
    )
