import os
import csv
import subprocess

def hms_to_seconds(hms):
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(hms.split(":"))))

input_path = "./video/elephant/31_2022_08_10_12.mp4"
start_time_str = "00:41:40"
end_time_str = "00:42:17"
clip_length = 30  # 秒

label = "walking"
species = "Elephant"
parent_class = "mammal"
csv_path = "labels_test.csv"
output_dir = "./video/elephant"
os.makedirs(output_dir, exist_ok=True)

start_time_sec = hms_to_seconds(start_time_str)
end_time_sec = hms_to_seconds(end_time_str)

with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["video_path", "action", "species", "parent_class"])
    # writer.writeheader()

    # すでにある clip_XXX.mp4 の最大番号を調べて続きから始める
    existing_clips = [f for f in os.listdir(output_dir) if f.startswith("clip_") and f.endswith(".mp4")]
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_clips if f.split("_")[1].split(".")[0].isdigit()]
    i = max(existing_indices) + 1 if existing_indices else 0
    count = 0
    for t in range(start_time_sec, end_time_sec, clip_length):
        clip_start = t
        clip_end = min(t + clip_length, end_time_sec)
        duration = clip_end - clip_start

        clip_name = f"clip_{i:03d}.mp4"
        output_path = os.path.join(output_dir, clip_name)

        cmd = [
            "ffmpeg",
            "-ss", str(clip_start),
            "-i", input_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",  # 上書き確認なし
            output_path
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        writer.writerow({
            "video_path": output_path.replace("\\", "/"),
            "action": label,
            "species": species,
            "parent_class": parent_class
        })

        i += 1
        count += 1

print(f"✅ {count}本の動画クリップとCSVを保存しました → {csv_path}")
