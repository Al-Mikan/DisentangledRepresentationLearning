import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path


# =============================================
# vec_root 判定
# =============================================
def detect_vec_root(video_path: str) -> str:
    p = video_path.lower()

    if "animalkingdom_augmented" in p:
        return "animalkingdom_augmented"
    if "animalkingdom" in p:
        return "animalkingdom"

    if "polar" in p:
        return "polar"
    if "elephant" in p:
        return "elephant"
    return "animalkingdom"



# =====================================================
# Base Dataset（df を encode し、最低限の処理だけ）
# =====================================================
class BaseDataset(Dataset):
    def __init__(self, df, le_act, le_sp):
        df = df.copy()
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        df["action"] = le_act.transform(df["action"])
        df["species"] = le_sp.transform(df["species"])

        self.df = df
        self.le_act = le_act
        self.le_sp = le_sp

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _load_sliding(dir_path: Path):
        files = sorted((dir_path).glob("*.npy"))
        mats = [np.load(f).squeeze() for f in files]  
        return np.stack(mats)


# =====================================================
# MAE Dataset
# =====================================================
class MAE_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, pooling=True, frame_stride=1):
        super().__init__(df, le_act, le_sp)
        self.pooling = pooling
        self.frame_stride = frame_stride

        # samplesリストを構築（pooling=True/False共通で使用）
        # 形式: (npy_path, action, species, vid)
        self.samples = []
        
        valid_rows = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])
            base = Path(f"./vector/{root}/{vid}")

            if pooling:
                npy_path = base / "avg_pooling.npy"
                if npy_path.exists():
                    self.samples.append((npy_path, row["action"], row["species"], vid))
                    valid_rows.append(row)
            else:
                npy_files = sorted((base / "sliding_list").glob("*.npy"))
                # frame_strideごとにサンプリング（重複フレーム回避）
                npy_files = npy_files[::frame_stride]
                if npy_files:
                    valid_rows.append(row)
                    for npy_path in npy_files:
                        self.samples.append((npy_path, row["action"], row["species"], vid))

        # dfも保持（species prior計算などで使用）
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, action, species, vid = self.samples[idx]
        x = np.load(npy_path).squeeze()

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(species, dtype=torch.long),
        )


# =====================================================
# X3D Dataset
# =====================================================
class X3D_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, centered=False, pooling=True, frame_stride=1):
        super().__init__(df, le_act, le_sp)
        self.centered = centered
        self.pooling  = pooling
        self.frame_stride = frame_stride

        folder = "x3d_vector_centered" if centered else "x3d_vector"

        # samplesリストを構築（pooling=True/False共通で使用）
        # 形式: (npy_path, action, species, vid)
        self.samples = []
        
        valid_rows = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])
            base = Path(f"./{folder}/{root}/{vid}")

            if pooling:
                npy_path = base / "avg_pooling.npy"
                if npy_path.exists():
                    self.samples.append((npy_path, row["action"], row["species"], vid))
                    valid_rows.append(row)
            else:
                npy_files = sorted((base / "sliding_list").glob("*.npy"))
                # frame_strideごとにサンプリング（重複フレーム回避）
                npy_files = npy_files[::frame_stride]
                if npy_files:
                    valid_rows.append(row)
                    for npy_path in npy_files:
                        self.samples.append((npy_path, row["action"], row["species"], vid))

        # dfも保持（species prior計算などで使用）
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, action, species, vid = self.samples[idx]
        x = np.load(npy_path).squeeze()

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(species, dtype=torch.long),
        )


# =====================================================
# GatedFusion Dataset
# =====================================================
class X3D_MAE_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, centered=False, pooling=True, frame_stride=1):
        super().__init__(df, le_act, le_sp)
        self.centered = centered
        self.pooling  = pooling
        self.frame_stride = frame_stride

        folder = "x3d_vector_centered" if centered else "x3d_vector"

        # samplesリストを構築（pooling=True/False共通で使用）
        # 形式: (x3d_npy_path, mae_npy_path, action, species, vid)
        self.samples = []
        
        valid_rows = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])

            x3d_base = Path(f"./{folder}/{root}/{vid}")
            mae_base = Path(f"./vector/{root}/{vid}")

            if pooling:
                x3d_path = x3d_base / "avg_pooling.npy"
                mae_path = mae_base / "avg_pooling.npy"
                if x3d_path.exists() and mae_path.exists():
                    self.samples.append((x3d_path, mae_path, row["action"], row["species"], vid))
                    valid_rows.append(row)
            else:
                x3d_files = sorted((x3d_base / "sliding_list").glob("*.npy"))
                mae_files = sorted((mae_base / "sliding_list").glob("*.npy"))
                # frame_strideごとにサンプリング（重複フレーム回避）
                x3d_files = x3d_files[::frame_stride]
                mae_files = mae_files[::frame_stride]

                if x3d_files and mae_files:
                    valid_rows.append(row)
                    # フレーム数が少ない方に合わせる
                    T = min(len(x3d_files), len(mae_files))
                    for i in range(T):
                        self.samples.append((x3d_files[i], mae_files[i], row["action"], row["species"], vid))

        # dfも保持（species prior計算などで使用）
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x3d_path, mae_path, action, species, vid = self.samples[idx]
        x3d = np.load(x3d_path).squeeze()
        mae = np.load(mae_path).squeeze()

        return (
            torch.tensor(x3d, dtype=torch.float32),
            torch.tensor(mae, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(species, dtype=torch.long),
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
