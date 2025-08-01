import pandas as pd
from decord import VideoReader

# --- ① 入力 CSV 読み込み ---
df = pd.read_csv("./label/animalkingdom/train/labels.csv")

# --- ② フレーム数条件でフィルタ ---
keep_rows = []
for idx, row in df.iterrows():
    path = row["video_path"]
    try:
        vr = VideoReader(path)
        total_frames = len(vr)
        if total_frames >= 16:
            keep_rows.append(row)
        else:
            print(f"除外: {path} (フレーム数={total_frames})")
    except Exception as e:
        print(f"⚠️ 読み込み失敗: {path} -> {e}")

# --- ③ フィルタ後に保存 ---
df_filtered = pd.DataFrame(keep_rows)
df_filtered.to_csv("./label/animalkingdom/train/labels_filtered.csv", index=False)
print(f"✅ {len(df_filtered)} 件を保存しました！")
