import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 動画フォルダ
video_dir = "./video"

# 秒数一覧（小数も含める）
duration_list = []

# 動画を走査して長さ（秒）を取得
for root, _, files in os.walk(video_dir):
    for file in files:
        if file.endswith(".mp4"):
            path = os.path.join(root, file)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()

            if fps > 0 and frame_count > 0:
                duration = frame_count / fps
                duration_list.append(duration)

# 統計表示（小数2桁で表示）
print(f"✅ 動画数: {len(duration_list)}")
print(f"⏱️ 平均秒数: {sum(duration_list)/len(duration_list):.2f} 秒")
print(f"⏳ 最短: {min(duration_list):.2f} 秒")
print(f"⏱️ 最長: {max(duration_list):.2f} 秒")

# ヒストグラムを0.5秒刻みで作成
plt.figure(figsize=(8, 5))
plt.hist(duration_list, bins=np.arange(0, max(duration_list) + 0.5, 0.5),
         color='salmon', edgecolor='black')
plt.xlabel("動画の長さ（秒）")
plt.ylabel("動画数")
plt.title("🎬 /video ディレクトリ内の動画長さの分布")
plt.grid(True)
plt.tight_layout()
os.makedirs("./analyze", exist_ok=True)
plt.savefig("./analyze/video_duration_histogram.png")
# plt.show()
