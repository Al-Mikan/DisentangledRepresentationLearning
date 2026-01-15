import pandas as pd
from decord import VideoReader
from tqdm import tqdm
import os

def count_valid_videos(
    csv_path: str,
    video_root: str = "./",
    min_frames: int = 17,
):
    df = pd.read_csv(csv_path)
    df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

    total = len(df)
    valid = 0
    invalid = 0
    errors = 0

    invalid_list = []

    for p in tqdm(df["video_path"], desc="Checking videos"):
        full_path = os.path.join(video_root, p)

        if not os.path.exists(full_path):
            errors += 1
            invalid_list.append((p, "not_found"))
            continue

        try:
            vr = VideoReader(full_path)
            n_frames = len(vr)

            if n_frames >= min_frames:
                valid += 1
            else:
                invalid += 1
                invalid_list.append((p, f"short({n_frames})"))

        except Exception as e:
            errors += 1
            invalid_list.append((p, "load_error"))

    print("\n===== Summary =====")
    print(f"CSV total       : {total}")
    print(f"Valid (>= {min_frames} frames): {valid}")
    print(f"Invalid (short) : {invalid}")
    print(f"Errors          : {errors}")
    print(f"Valid ratio     : {valid / total:.3f}")

    if "action" in df.columns:
        actions = df["action"].unique()
        print(f"\n===== Actions ({len(actions)}) =====")
        print(df["action"].value_counts().to_string())

    if "species" in df.columns:
        species = df["species"].unique()
        print(f"\n===== Species ({len(species)}) =====")
        print(df["species"].value_counts().to_string())

    return invalid_list

if __name__ == "__main__":
    invalid = count_valid_videos(
    csv_path="./label/test/elephant.csv",
    video_root="./",
    min_frames=17,
)
