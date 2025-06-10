import os
import cv2
import pandas as pd

def get_video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0

video_dir = "./test_video"
csv_path = "labels_test.csv"
duration_threshold = 5  # ← この秒数以下の動画を削除

# 読み込む
df = pd.read_csv(csv_path)
df["video_path"] = df["video_path"].str.replace("\\", "/")  # パスの整形

deleted_files = []

# 動画ファイルごとに確認
for file in sorted(os.listdir(video_dir)):
    if file.endswith(".mp4"):
        full_path = os.path.join(video_dir, file)
        duration = get_video_duration(full_path)

        if duration <= duration_threshold:
            print(f"🗑️ {file}: {duration:.2f} 秒 → 削除対象")
            os.remove(full_path)
            deleted_files.append(os.path.join(video_dir, file).replace("\\", "/"))

# CSVから該当ファイルの行を削除
original_len = len(df)
df = df[~df["video_path"].isin(deleted_files)]
df.to_csv(csv_path, index=False)

print(f"\n✅ {len(deleted_files)} 本の動画とラベルを削除しました")
print(f"📝 CSVの行数: {original_len} → {len(df)}")

# action の種類と数を表示
print("\n🎬 アクションの種類と件数:")
print(df["action"].value_counts())