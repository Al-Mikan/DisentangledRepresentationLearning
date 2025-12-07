import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================================
# Base Dataset (共通処理)
# =============================================
class BaseDataset(Dataset):
    def __init__(self, dataframe, le_act, le_sp):
        df = dataframe.copy()
        df["video_path"] = df["video_path"].str.replace("\\", "/").str.strip()

        # ラベルを ID へ変換
        df["action"] = le_act.transform(df["action"])
        df["species"] = le_sp.transform(df["species"])

        self.le_act = le_act
        self.le_sp = le_sp
        self.df = df

    def __len__(self):
        return len(self.df)


# =============================================
# MAE Dataset（VideoMAE特徴量）
# ---------------------------------------------
# vector/<datatype>/<video_name>/
#     avg_pooling.npy
#     sliding_list/000.npy
# =============================================
class MAE_Dataset(BaseDataset):
    def __init__(self, dataframe, le_act, le_sp, vector_root, pooling=True):
        super().__init__(dataframe, le_act, le_sp)

        self.vector_root = vector_root
        self.pooling = pooling

        valid_rows = []

        for _, row in self.df.iterrows():
            vid = os.path.splitext(os.path.basename(row["video_path"]))[0]
            base_dir = os.path.join(vector_root, vid)

            if pooling:
                f = os.path.join(base_dir, "avg_pooling.npy")
                if os.path.exists(f):
                    valid_rows.append(row)
            else:
                sliding_dir = os.path.join(base_dir, "sliding_list")
                files = glob.glob(os.path.join(sliding_dir, "*.npy"))
                if len(files) > 0:
                    valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = os.path.splitext(os.path.basename(row["video_path"]))[0]
        base_dir = os.path.join(self.vector_root, vid)

        # =======================
        # pooling モード
        # =======================
        if self.pooling:
            npy_path = os.path.join(base_dir, "avg_pooling.npy")
            vec = np.load(npy_path)   # (D,)
            return (
                torch.tensor(vec, dtype=torch.float32),
                row["action"],
                row["species"],
            )

        # =======================
        # sliding_list モード
        # =======================
        sliding_dir = os.path.join(base_dir, "sliding_list")
        files = sorted(glob.glob(os.path.join(sliding_dir, "*.npy")))

        vecs = [np.load(f) for f in files]
        mat = np.stack(vecs)   # (T, D)

        return (
            torch.tensor(mat, dtype=torch.float32),
            row["action"],
            row["species"],
        )


# =============================================
# X3D Dataset（Flow / X3D Motion 特徴量）
# ---------------------------------------------
# x3d_dir/<video_name>/<video_name>.npy
# =============================================
class X3D_Dataset(BaseDataset):
    def __init__(self, dataframe, le_act, le_sp, x3d_dir, pooling=True):
        super().__init__(dataframe, le_act, le_sp)
        self.x3d_dir = x3d_dir
        self.pooling = pooling

        valid_rows = []
        for _, row in self.df.iterrows():
            vid = os.path.splitext(os.path.basename(row["video_path"]))[0]
            base = os.path.join(x3d_dir, vid)
            if pooling:
                p = os.path.join(base, "avg_pooling.npy")
                if os.path.exists(p):
                    valid_rows.append(row)
            else:
                sliding_dir = os.path.join(base, "sliding_list")
                if glob.glob(os.path.join(sliding_dir, "*.npy")):
                    valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = os.path.splitext(os.path.basename(row["video_path"]))[0]
        base = os.path.join(self.x3d_dir, vid)

        if self.pooling:
            npy_path = os.path.join(base, "avg_pooling.npy")
            x3d_vec = np.load(npy_path)  # (D,)
            return (
                torch.tensor(x3d_vec, dtype=torch.float32),
                row["action"],
                row["species"],
            )

        sliding_dir = os.path.join(base, "sliding_list")
        files = sorted(glob.glob(os.path.join(sliding_dir, "*.npy")))
        x3d_mat = np.stack([np.load(f) for f in files])  # (T, D)
        return (
            torch.tensor(x3d_mat, dtype=torch.float32),
            row["action"],
            row["species"],
        )


# =============================================
# X3D + MAE Dataset（GatedFusion 用）
# ---------------------------------------------
# Motion: x3d_dir/<video>/<video>.npy
# MAE:
#   pooling:      vector_root/<video>/avg_pooling.npy
#   sliding_list: vector_root/<video>/sliding_list/*.npy  (T, D)
# =============================================
class X3D_MAE_Dataset(BaseDataset):
    def __init__(self, dataframe, le_act, le_sp, x3d_dir, vector_root, pooling=True):
        super().__init__(dataframe, le_act, le_sp)

        self.x3d_dir = x3d_dir
        self.vector_root = vector_root
        self.pooling = pooling

        valid_rows = []
        for _, row in self.df.iterrows():
            vid = os.path.splitext(os.path.basename(row["video_path"]))[0]

            # Motion
            x3d_base = os.path.join(x3d_dir, vid)
            if pooling:
                x3d_path = os.path.join(x3d_base, "avg_pooling.npy")
                if not os.path.exists(x3d_path):
                    continue
            else:
                x3d_sliding = os.path.join(x3d_base, "sliding_list")
                if len(glob.glob(os.path.join(x3d_sliding, "*.npy"))) == 0:
                    continue

            # Appearance
            base_dir = os.path.join(vector_root, vid)

            if pooling:
                p2 = os.path.join(base_dir, "avg_pooling.npy")
                if not os.path.exists(p2):
                    continue
            else:
                sliding_dir = os.path.join(base_dir, "sliding_list")
                if len(glob.glob(os.path.join(sliding_dir, "*.npy"))) == 0:
                    continue

            valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid = os.path.splitext(os.path.basename(row["video_path"]))[0]

        # Motion (X3D)
        x3d_base = os.path.join(self.x3d_dir, vid)
        if self.pooling:
            x3d_vec = np.load(os.path.join(x3d_base, "avg_pooling.npy"))  # (D,)
        else:
            x3d_files = sorted(glob.glob(os.path.join(x3d_base, "sliding_list", "*.npy")))
            x3d_vec = np.stack([np.load(f) for f in x3d_files])  # (T, D)

        # Appearance (VideoMAE)
        base_dir = os.path.join(self.vector_root, vid)

        if self.pooling:
            mae_vec = np.load(os.path.join(base_dir, "avg_pooling.npy"))    # (D,)
            return (
                torch.tensor(x3d_vec, dtype=torch.float32),
                torch.tensor(mae_vec, dtype=torch.float32),
                row["action"],
                row["species"],
            )

        mae_files = sorted(glob.glob(os.path.join(base_dir, "sliding_list", "*.npy")))
        mae_mat = np.stack([np.load(f) for f in mae_files])      # (T, D)

        # スライディング長が一致しない場合は短い方に合わせる
        T = min(x3d_vec.shape[0], mae_mat.shape[0]) if x3d_vec.ndim == 2 else mae_mat.shape[0]
        if x3d_vec.ndim == 1:
            x3d_mat = np.tile(x3d_vec, (T, 1))
        else:
            x3d_mat = x3d_vec[:T]
            mae_mat = mae_mat[:T]

        return (
            torch.tensor(x3d_mat, dtype=torch.float32),
            torch.tensor(mae_mat, dtype=torch.float32),
            row["action"],
            row["species"],
        )


def set_seed(seed: int) -> None:
    """乱数シードを固定する"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)