import cv2
import albumentations as A
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

# =========================
# 設定
# =========================
CSV_PATH = "./label/train/animalkingdom.csv"
OUTPUT_VIDEO_DIR = Path("animalkingdom_augmented")
OUTPUT_CSV_PATH = "./label/train/animalkingdom_augmented.csv"

NUM_AUG_PER_VIDEO = 3   # 1本の動画から何本作るか
FPS = None              # None → 元動画のFPSを使う

OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Albumentations (video-safe / zoom-in only)
# =========================
transform = A.Compose([
    # 左右反転（確率的）
    A.HorizontalFlip(p=0.5),

    # ズームイン + ランダム位置 crop
    A.RandomResizedCrop(
        height=224,
        width=224,
        scale=(0.75, 1.0),   # ズームインのみ
        ratio=(0.9, 1.1),
        p=1.0
    ),
])

# =========================
# Video I/O
# =========================
def load_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        return None, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    if len(frames) == 0:
        return None, 0

    return np.stack(frames), fps


def save_video(frames, path, fps):
    if len(frames) == 0:
        return

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()

# =========================
# Main
# =========================
df = pd.read_csv(CSV_PATH)
rows = []

print(f"🎬 Generating {NUM_AUG_PER_VIDEO} augmented videos per input...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    video_path = Path(row["video_path"])

    frames, fps = load_video(video_path)
    if frames is None:
        print(f"⚠️ Skipping broken video: {video_path}")
        continue

    save_fps = FPS if FPS is not None else fps
    if save_fps <= 0:
        save_fps = 30.0

    for i in range(NUM_AUG_PER_VIDEO):
        try:
            # 全フレーム一括変換（時間的一貫性あり）
            augmented = transform(images=list(frames))
            aug_frames = np.stack(augmented["images"])

            out_name = f"{video_path.stem}_aug{i}.mp4"
            out_path = OUTPUT_VIDEO_DIR / out_name

            save_video(aug_frames, out_path, save_fps)

            # CSV 行を追加
            new_row = row.to_dict()
            new_row["video_path"] = str(out_path)
            rows.append(new_row)

        except Exception as e:
            print(f"❌ Error augmenting {video_path.name}: {e}")

# =========================
# Save CSV
# =========================
pd.DataFrame(rows).to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✅ Saved: {OUTPUT_CSV_PATH} (Total {len(rows)} augmented videos)")
