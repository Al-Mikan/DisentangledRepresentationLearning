import pandas as pd
import ast
import csv
import os
import random
from collections import Counter

# === パラメータ ===
ACTION_MIN_COUNT = 100  # アクション出現数の閾値
TEST_RATIO = 0.2       # テストデータ比率
csv_input_path = "./label/AR_metadata.csv"
video_dir = "./video/animalkingdom"

# === 出力パス ===
csv_train = "./label/animalkingdom_split/train/labels.csv"
csv_test = "./label/animalkingdom_split/test/labels_test.csv"

# === 出力先ディレクトリ作成 ===
os.makedirs(os.path.dirname(csv_train), exist_ok=True)
os.makedirs(os.path.dirname(csv_test), exist_ok=True)

# === 入力読み込み ===
df = pd.read_csv(csv_input_path)

# === 一時リストに格納 ===
temp_rows = []

for _, row in df.iterrows():
    video_id = row["video_id"]
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    try:
        actions = ast.literal_eval(row["list_animal_action"])
        parent_classes = ast.literal_eval(row["list_animal_parent_class"])
    except Exception as e:
        print(f"⚠️ パースエラー: {video_id}")
        continue

    if len(actions) == 1 and len(parent_classes) == 1 and parent_classes[0].lower() == "mammal":
        species, action = actions[0]
        parent_class = parent_classes[0]
        temp_rows.append([video_path, action, species, parent_class])

# === action 出現数でフィルタ ===
action_counter = Counter([row[1] for row in temp_rows])
valid_actions = {action for action, count in action_counter.items() if count >= ACTION_MIN_COUNT}

filtered_rows = [row for row in temp_rows if row[1] in valid_actions]

# === シャッフルして train/test に分割 ===
random.shuffle(filtered_rows)
split_idx = int(len(filtered_rows) * (1 - TEST_RATIO))
train_rows = filtered_rows[:split_idx]
test_rows = filtered_rows[split_idx:]

# === 書き出し ===
with open(csv_train, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(train_rows)

with open(csv_test, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(test_rows)

print(f"✅ Train CSV: {len(train_rows)} 件 → {csv_train}")
print(f"✅ Test CSV:  {len(test_rows)} 件 → {csv_test}")

# === 使用された action ラベル一覧 ===
used_actions = sorted({row[1] for row in filtered_rows})
print("\n🔍 使用された Action ラベル一覧:")
for action in used_actions:
    print(f"・{action}")
