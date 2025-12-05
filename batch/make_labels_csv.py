import pandas as pd
import ast
import csv
import os
from collections import Counter
import random
from decord import VideoReader

ACTION_MIN_COUNT = 50
MIN_FRAMES = 16
csv_input_path = "./label/AR_metadata.csv"
video_dir = "./video/animalkingdom"

species_target = ""
use_split = True
test_ratio = 0.1

# 再現性
random.seed(42)

# === 出力パス ===
if species_target:
    csv_test = f"./label/{species_target.lower()}/test/labels_test.csv"
    csv_train = f"./label/{species_target.lower()}/train/labels.csv"
elif use_split:
    csv_test = "./label/animalkingdom_split/test/labels_test.csv"
    csv_train = "./label/animalkingdom_split/train/labels.csv"
else:
    csv_test = None
    csv_train = "./label/animalkingdom_50/train/labels.csv"

# === ディレクトリ作成 ===
if csv_test is not None:
    os.makedirs(os.path.dirname(csv_test), exist_ok=True)
os.makedirs(os.path.dirname(csv_train), exist_ok=True)


# === 読み込み ===
df = pd.read_csv(csv_input_path)

# --- 一時リスト ---
temp_rows = []

for _, row in df.iterrows():
    video_id = row["video_id"]
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    # 動画チェック
    if not os.path.isfile(video_path):
        print(f"⚠ Missing video: {video_id}")
        continue

    try:
        vr = VideoReader(video_path)
        if len(vr) < MIN_FRAMES:
            print(f"⏩ Skip short (<{MIN_FRAMES}) : {video_id}")
            continue
    except:
        print(f"⚠ Video open failed: {video_id}")
        continue

    # ラベル解析
    try:
        actions = ast.literal_eval(row["list_animal_action"])
        parent_classes = ast.literal_eval(row["list_animal_parent_class"])
    except:
        continue

    if len(actions) == 1 and len(parent_classes) == 1 and parent_classes[0].lower() == "mammal":
        species, action = actions[0]
        temp_rows.append([video_path, action, species, parent_classes[0]])


# --- action 出現数フィルタ ---
action_counter = Counter([row[1] for row in temp_rows])
valid_actions = {a for a, c in action_counter.items() if c >= ACTION_MIN_COUNT}

final_rows = [r for r in temp_rows if r[1] in valid_actions]

print(f"📊 Valid videos: {len(final_rows)}")


# --- ① species_target split ---
if species_target:
    species_rows = [row for row in final_rows if row[2] == species_target]
    other_rows = [row for row in final_rows if row[2] != species_target]

    with open(csv_test, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "action", "species", "parent_class"])
        writer.writerows(species_rows)

    with open(csv_train, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "action", "species", "parent_class"])
        writer.writerows(other_rows)

    print(f"TEST={len(species_rows)}, TRAIN={len(other_rows)}")
    exit()


# --- ② ランダム split ---
if use_split:
    random.shuffle(final_rows)
    split_idx = int(len(final_rows) * test_ratio)

    test_rows = final_rows[:split_idx]
    train_rows = final_rows[split_idx:]

    with open(csv_test, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "action", "species", "parent_class"])
        writer.writerows(test_rows)

    with open(csv_train, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "action", "species", "parent_class"])
        writer.writerows(train_rows)

    print(f"🔀 Random split done: TEST={len(test_rows)}, TRAIN={len(train_rows)}")
    exit()


# --- ③ split なし ---
with open(csv_train, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(final_rows)

print(f"TRAIN={len(final_rows)} (all)")
