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
    if "polar" in p:
        return "polar"
    if "animalkingdom" in p:
        return "animalkingdom"
    return "animalkingdom"


# =====================================================
# Base Dataset（df を encode し、最低限の処理だけ）
# =====================================================
class BaseDataset(Dataset):
    def __init__(self, df, le_act, le_sp):
        df = df.copy()
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        # 🔥 action/species をここで数値化
        df["action"] = le_act.transform(df["action"])
        df["species"] = le_sp.transform(df["species"])

        self.df = df
        self.le_act = le_act
        self.le_sp = le_sp

    def __len__(self):
        return len(self.df)

    # 便利ヘルパー
    @staticmethod
    def _load_sliding(dir_path: Path):
        files = sorted((dir_path).glob("*.npy"))
        mats = [np.load(f) for f in files]
        return np.stack(mats)


# =====================================================
# MAE Dataset
# =====================================================
class MAE_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, pooling=True):
        super().__init__(df, le_act, le_sp)
        self.pooling = pooling

        valid = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])
            base = Path(f"./vector/{root}/{vid}")

            if pooling:
                if (base / "avg_pooling.npy").exists():
                    valid.append(row)
            else:
                if list((base/"sliding_list").glob("*.npy")):
                    valid.append(row)

        self.df = pd.DataFrame(valid).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid  = Path(row["video_path"]).stem
        root = detect_vec_root(row["video_path"])
        base = Path(f"./vector/{root}/{vid}")

        if self.pooling:
            x = np.load(base / "avg_pooling.npy")
        else:
            x = self._load_sliding(base / "sliding_list")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(row["action"], dtype=torch.long),
            torch.tensor(row["species"], dtype=torch.long),
        )


# =====================================================
# X3D Dataset
# =====================================================
class X3D_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, centered=False, pooling=True):
        super().__init__(df, le_act, le_sp)
        self.centered = centered
        self.pooling  = pooling

        folder = "x3d_vector_centered" if centered else "x3d_vector"

        valid = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])
            base = Path(f"./{folder}/{root}/{vid}")

            if pooling:
                if (base / "avg_pooling.npy").exists():
                    valid.append(row)
            else:
                if list((base/"sliding_list").glob("*.npy")):
                    valid.append(row)

        self.df = pd.DataFrame(valid).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid  = Path(row["video_path"]).stem
        root = detect_vec_root(row["video_path"])
        folder = "x3d_vector_centered" if self.centered else "x3d_vector"
        base = Path(f"./{folder}/{root}/{vid}")

        if self.pooling:
            x = np.load(base / "avg_pooling.npy")
        else:
            x = self._load_sliding(base / "sliding_list")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(row["action"], dtype=torch.long),
            torch.tensor(row["species"], dtype=torch.long),
        )


# =====================================================
# GatedFusion Dataset
# =====================================================
class X3D_MAE_Dataset(BaseDataset):
    def __init__(self, df, le_act, le_sp, centered=False, pooling=True):
        super().__init__(df, le_act, le_sp)
        self.centered = centered
        self.pooling  = pooling

        valid = []
        for _, row in self.df.iterrows():
            vid = Path(row["video_path"]).stem
            root = detect_vec_root(row["video_path"])

            # x3d
            folder = "x3d_vector_centered" if centered else "x3d_vector"
            xb = Path(f"./{folder}/{root}/{vid}")

            if pooling:
                if not (xb / "avg_pooling.npy").exists():
                    continue
            else:
                if not list((xb/"sliding_list").glob("*.npy")):
                    continue

            # mae
            mb = Path(f"./vector/{root}/{vid}")
            if pooling:
                if not (mb / "avg_pooling.npy").exists():
                    continue
            else:
                if not list((mb/"sliding_list").glob("*.npy")):
                    continue

            valid.append(row)

        self.df = pd.DataFrame(valid).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid  = Path(row["video_path"]).stem
        root = detect_vec_root(row["video_path"])

        # X3D
        folder = "x3d_vector_centered" if self.centered else "x3d_vector"
        xb = Path(f"./{folder}/{root}/{vid}")

        if self.pooling:
            x3d = np.load(xb / "avg_pooling.npy")
        else:
            x3d = self._load_sliding(xb / "sliding_list")

        # MAE
        mb = Path(f"./vector/{root}/{vid}")
        if self.pooling:
            mae = np.load(mb / "avg_pooling.npy")
            return (
                torch.tensor(x3d, dtype=torch.float32),
                torch.tensor(mae, dtype=torch.float32),
                torch.tensor(row["action"], dtype=torch.long),
                torch.tensor(row["species"], dtype=torch.long),
            )

        mae_mat = self._load_sliding(mb / "sliding_list")

        # フレーム整形
        T = min(x3d.shape[0], mae_mat.shape[0])
        return (
            torch.tensor(x3d[:T], dtype=torch.float32),
            torch.tensor(mae_mat[:T], dtype=torch.float32),
            torch.tensor(row["action"], dtype=torch.long),
            torch.tensor(row["species"], dtype=torch.long),
        )
