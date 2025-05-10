import os
import shutil
import ast
import pandas as pd

csv_path = "./AR_metadata.csv"
video_src_dir = "C:/Users/nekom/research/Animal_Kingdom-20250414T050205Z-002/Animal_Kingdom/action_recognition/video-001/video"  # 動画の元フォルダ
dst_root = "./"

df = pd.read_csv(csv_path)

for idx, row in df.iterrows():
    video_id = row['video_id']
    animal_actions = ast.literal_eval(row['list_animal_action'])  # [('Common Crane', 'Attending'), ...]
    print(animal_actions)

    for i, (species, action) in enumerate(animal_actions):
        action_dir = os.path.join(dst_root, action)
        species_dir = os.path.join(action_dir, species)

        os.makedirs(species_dir, exist_ok=True)

        src_path = os.path.join(video_src_dir, f"{video_id}.mp4")
        dst_path = os.path.join(species_dir, f"{video_id}{i}.mp4")  # 複数同じ動画が存在するため indexを付ける

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"⚠️ 動画が見つかりません: {src_path}")