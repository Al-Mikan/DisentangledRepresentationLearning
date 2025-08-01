import os
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    """データセットの共通処理を担う基底クラス"""
    def __init__(self, dataframe, le_act, le_sp):
        df = dataframe.copy()
        df['video_path'] = df['video_path'].str.replace('\\', '/').str.strip()
        
        self.le_act = le_act
        self.le_sp = le_sp
        df['action'] = self.le_act.transform(df['action'])
        df['species'] = self.le_sp.transform(df['species'])
        self.df = df

    def __len__(self):
        return len(self.df)

class MAEDataset(BaseDataset):
    def __init__(self, csv_path, vmae_json, le_act, le_sp):
        super().__init__(csv_path, le_act, le_sp)
        with open(vmae_json, 'r') as f:
            self.vmae_dict = json.load(f)
        
        # 存在しないVMAE特徴量を持つ行を削除
        self.df = self.df[self.df['video_path'].isin(self.vmae_dict)].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vmae_vec = np.array(self.vmae_dict[row['video_path']])
        return (
            torch.tensor(vmae_vec, dtype=torch.float32),
            row['action'],
            row['species']
        )

class FlowNpyDataset(BaseDataset):
    def __init__(self, csv_path, flow_dir, le_act, le_sp):
        super().__init__(csv_path, le_act, le_sp)
        self.flow_dir = flow_dir
        
        # 存在しないNPYファイルを持つ行を削除
        self.df['npy_path'] = self.df['video_path'].apply(
            lambda p: os.path.join(flow_dir, os.path.splitext(os.path.basename(p))[0], f"{os.path.splitext(os.path.basename(p))[0]}.npy")
        )
        self.df = self.df[self.df['npy_path'].apply(os.path.isfile)].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vec = np.load(row['npy_path']).squeeze(0)
        return (
            torch.tensor(vec, dtype=torch.float32),
            row['action'],
            row['species']
        )

class X3DVideoMAEDataset(BaseDataset):
    def __init__(self, csv_path, x3d_dir, vmae_json, le_act, le_sp):
        super().__init__(csv_path, le_act, le_sp)
        self.x3d_dir = x3d_dir
        with open(vmae_json, 'r') as f:
            self.vmae_dict = json.load(f)

        # 存在しない特徴量を持つ行を削除
        self.df = self.df[self.df['video_path'].isin(self.vmae_dict)].reset_index(drop=True)
        self.df['x3d_path'] = self.df['video_path'].apply(
            lambda p: os.path.join(x3d_dir, os.path.splitext(os.path.basename(p))[0], f"{os.path.splitext(os.path.basename(p))[0]}.npy")
        )
        self.df = self.df[self.df['x3d_path'].apply(os.path.isfile)].reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x3d_vec = np.load(row['x3d_path']).squeeze(0)
        vmae_vec = np.array(self.vmae_dict[row['video_path']])
        return (
            torch.tensor(x3d_vec, dtype=torch.float32),
            torch.tensor(vmae_vec, dtype=torch.float32),
            row['action'],
            row['species']
        )