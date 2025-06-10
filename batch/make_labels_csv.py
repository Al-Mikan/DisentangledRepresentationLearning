import pandas as pd
import ast
import csv
import os
from collections import Counter

# パス設定
csv_input_path = "AR_metadata.csv"
csv_output_path = "labels.csv"
video_dir = "video"  # 動画が平置きされているディレクトリ

df = pd.read_csv(csv_input_path)

# 一時的にすべての条件を満たす行を収集
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

# 行動（action）の出現回数をカウント
action_counter = Counter([row[1] for row in temp_rows])
valid_actions = {action for action, count in action_counter.items() if count >= 100}

# 条件に合うものだけ抽出
final_rows = [row for row in temp_rows if row[1] in valid_actions]

# 書き出し
with open(csv_output_path, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["video_path", "action", "species", "parent_class"])
    writer.writerows(final_rows)

print(f"✅ labels.csv を {len(final_rows)} 件で作成しました（mammal かつ 1ラベル かつ action出現数100以上）。")
