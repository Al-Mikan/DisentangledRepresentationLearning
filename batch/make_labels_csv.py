import pandas as pd
import ast
import csv
import os
from collections import Counter

# === 設定 ===
csv_input_path = "AR_metadata.csv"
video_dir = "video"  # 動画のディレクトリ
species_target = "Wolf"  # ← ここで抽出したい種を指定

csv_output_species = f"labels_{species_target}.csv"
csv_output_others = f"labels_not_{species_target}.csv"

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
valid_actions = {action for action, count in action_counter.items() if count >= 100}

final_rows = [row for row in temp_rows if row[1] in valid_actions]

# === 対象種とそれ以外で分割 ===
species_rows = [row for row in final_rows if row[2] == species_target]
other_rows = [row for row in final_rows if row[2] != species_target]

# === 書き出し ===
with open(csv_output_species, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(species_rows)

with open(csv_output_others, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(other_rows)

print(f"✅ {species_target} 用 CSV: {len(species_rows)} 件 → {csv_output_species}")
print(f"✅ その他種 CSV: {len(other_rows)} 件 → {csv_output_others}")

# === 使用された action ラベル一覧 ===
used_actions = sorted({row[1] for row in final_rows})
print("\n🔍 使用された Action ラベル一覧:")
for action in used_actions:
    print(f"・{action}")
