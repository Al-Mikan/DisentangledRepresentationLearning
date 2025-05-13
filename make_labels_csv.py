import os
import csv

data_root = "AnimalKingdomDataset"  # 動画フォルダのルート
csv_path = "labels.csv"  # 出力CSVファイル
species_info_path = "Animal.csv"  # 種とParent Classの対応表

# ------------------------------
# 対応表CSVの読み込み (species → parent class)
# ------------------------------
species_to_parent = {}

with open(species_info_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        species = row["Animal"].strip().lower().replace("'", "")  # ← ' を除去
        parent_class = row["Parent Class"].strip()
        species_to_parent[species] = parent_class

# ------------------------------
# データ収集
# ------------------------------
rows = []
unknown_species = set()

for action in os.listdir(data_root):
    action_path = os.path.join(data_root, action)
    if not os.path.isdir(action_path): continue

    for species in os.listdir(action_path):
        species_path = os.path.join(action_path, species)
        if not os.path.isdir(species_path): continue

        for fname in os.listdir(species_path):
            if fname.endswith(".mp4"):
                relative_path = os.path.join(data_root, action, species, fname)
                parent_class = species_to_parent.get(species.lower(), "Unknown")
                if parent_class == "Unknown":
                    unknown_species.add(species)
                rows.append([relative_path, action, species, parent_class])

# ------------------------------
# CSV 書き出し
# ------------------------------
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'action', 'species', 'parent_class'])
    writer.writerows(rows)

print(f"✅ labels.csv を {len(rows)} 件で作成しました。")

# ------------------------------
# 未知の種を警告
# ------------------------------
if unknown_species:
    print("\n⚠️ 以下の種は species_info.csv に見つかりませんでした：")
    for s in sorted(unknown_species):
        print(f"  - {s}")
