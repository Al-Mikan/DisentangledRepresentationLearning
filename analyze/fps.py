import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# 動画ディレクトリのパス
video_dir = "./video"

# fps の一覧
fps_list = []

# すべての mp4 ファイルを探索
for root, _, files in os.walk(video_dir):
    for file in files:
        if file.endswith(".mp4"):
            path = os.path.join(root, file)
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                fps_list.append(round(fps))  # 小数点を丸める（例: 29.97 → 30）

# FPS分布のカウント
fps_count = Counter(fps_list)
print("📊 FPS分布:", fps_count)

# 可視化
plt.figure(figsize=(8, 5))
plt.hist(fps_list, bins=range(int(min(fps_list)), int(max(fps_list)) + 2), edgecolor='black', align='left', color='skyblue')
plt.xlabel("FPS（フレームレート）")
plt.ylabel("動画数")
plt.title("🎥 /video ディレクトリ内の動画FPS分布")
plt.grid(True)
plt.tight_layout()
plt.savefig("./analyze/fps_histogram.png")
plt.show()
