import pandas as pd
import matplotlib.pyplot as plt
from decord import VideoReader, bridge
import os
import numpy as np


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