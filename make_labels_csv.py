# make_labels_csv.py
import os
import csv

data_root = "./"  # ルートディレクトリ（あなたの動画が入っている場所）
csv_path = "labels.csv"  # 出力ファイル名

rows = []

for action in os.listdir(data_root):
    action_path = os.path.join(data_root, action)
    if not os.path.isdir(action_path): continue

    for species in os.listdir(action_path):
        species_path = os.path.join(action_path, species)
        if not os.path.isdir(species_path): continue

        for fname in os.listdir(species_path):
            if fname.endswith(".mp4"):
                relative_path = os.path.join(action, species, fname)
                rows.append([relative_path, action, species])

# CSV 書き出し
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video_path', 'action', 'species'])
    writer.writerows(rows)

print(f"✅ labels.csv を {len(rows)} 件で作成しました")
