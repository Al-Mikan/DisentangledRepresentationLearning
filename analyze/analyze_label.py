import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, bridge
import os
import numpy as np

# decordのバックエンド設定
bridge.set_bridge('native')  # 'torch' でも可

# CSV読み込み
df = pd.read_csv("./label/animalkingdom/train/labels.csv")

# 🎯 ラベル統計情報の表示
print("📁 labels.csv の基本情報")
print(f"総データ数: {len(df)}")
print(f"行動 (action) の種類数: {df['action'].nunique()}")
print(df['action'].value_counts())
print()
print(f"種 (species) の種類数: {df['species'].nunique()}")
print(df['species'].value_counts())

# 🎞 フレーム数の取得
frame_counts = []
missing_files = []

for path in df["video_path"]:
    if not os.path.exists(path):
        missing_files.append(path)
        continue

    try:
        vr = VideoReader(path)
        frame_counts.append(len(vr))
    except Exception as e:
        print(f"⚠️ 読み込み失敗: {path}")
        continue

# 📊 フレーム数のヒストグラム
# plt.figure(figsize=(8, 5))
# plt.hist(frame_counts, bins=30, color='skyblue', edgecolor='black')
# plt.xlabel("フレーム数")
# plt.ylabel("動画数")
# plt.title("🎬 labels.csv に含まれる動画のフレーム数分布")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("analyze/frame_count_histogram.png")
# plt.show()

# 📈 統計情報の出力
# print("\n📊 フレーム数の統計:")
# print(f"件数: {len(frame_counts)}")
# print(f"平均: {np.mean(frame_counts):.1f}")
# print(f"中央値: {np.median(frame_counts):.1f}")
# print(f"最小: {np.min(frame_counts)}")
# print(f"最大: {np.max(frame_counts)}")

# 🚨 欠損ファイル表示
if missing_files:
    print("\n⚠️ 存在しないファイル:")
    for p in missing_files:
        print(" ", p)
