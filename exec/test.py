import os

import cv2
import os

# ========== 設定 ==========
video_path = "./video/animalkingdom/clip_000.mp4"   # 読み込みたいMP4ファイル
output_dir = "./sample_img/clip_000"          # フレームの保存先ディレクトリ
num_frames_to_save = 10          # 最初から保存する枚数

os.makedirs(output_dir, exist_ok=True)

# ========== 動画を開く ==========
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ 動画を開けませんでした: {video_path}")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ 総フレーム数: {frame_count}")

saved_idx = 0

while saved_idx < num_frames_to_save:
    ret, frame = cap.read()
    if not ret:
        break

    output_path = os.path.join(output_dir, f"frame_{saved_idx:05d}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"💾 Saved: {output_path}")
    saved_idx += 1

cap.release()
print(f"✅ 完了！ 保存枚数: {saved_idx}")
