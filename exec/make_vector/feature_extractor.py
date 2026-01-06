import os
import sys
import subprocess
import pandas as pd

CSV_PATH = os.path.abspath("./label/train/animalkingdom_augmented.csv")
VIDEO_ROOT = os.path.abspath("./")
VECTOR_ROOT = os.path.abspath("./vector")
STRIDE = "1"


def launch_worker(video_path, out_base):
    worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "feature_one_video.py"))

    cmd = [
        sys.executable,  # 同じ Python で実行
        worker_path,
        video_path,
        out_base,
        STRIDE
    ]

    print(f"\n🚀 Launch worker for {video_path}\n")
    subprocess.run(cmd)  # GPU を完全解放して次へ


def main():
    df = pd.read_csv(CSV_PATH)
    df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

    for _, row in df.iterrows():

        video_rel = row["video_path"]
        video_path = os.path.abspath(os.path.join(VIDEO_ROOT, video_rel))

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_base = os.path.abspath(os.path.join(VECTOR_ROOT, "animalkingdom_augmented", video_name))

        os.makedirs(out_base, exist_ok=True)

        launch_worker(video_path, out_base)


if __name__ == "__main__":
    main()
