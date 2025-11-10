import os
import json
from typing import Optional
from urllib import request as urlrequest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

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
    


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility where practical."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
# ---------------------------------
# Discord notification (optional)
# ---------------------------------
def discord_notify(
    content: str,
    *,
    username: str = "DisentangleBot",
    webhook_url: Optional[str] = None,
) -> None:
    """Send a message to Discord via webhook.

    - Skips silently if DISCORD_WEBHOOK_URL is not set.
    - Uses only stdlib (no external dependencies).
    - Adds User-Agent to avoid 403 Forbidden.
    - Automatically splits messages over 2000 chars.
    """
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        print("[discord_notify] Skipped: no webhook URL found.")
        return

    try:
        max_len = 1900  # Discord limit ≈ 2000
        chunks = [content[i:i + max_len] for i in range(0, len(content), max_len)] or [content]

        for i, chunk in enumerate(chunks):
            payload = {"content": chunk, "username": username}
            data = json.dumps(payload).encode("utf-8")

            req = urlrequest.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; DisentangleBot/1.0)"
                },
                method="POST"
            )

            with urlrequest.urlopen(req, timeout=10) as resp:
                if resp.status == 204:
                    print(f"[discord_notify] ✅ Sent message chunk {i+1}/{len(chunks)}")
                else:
                    print(f"[discord_notify] ⚠️ Unexpected status {resp.status}")

    except Exception as e:
        print(f"[discord_notify] ❌ Failed to send message: {e}")


def cleanup_memory() -> None:
    """CUDAキャッシュとガーベジコレクションを呼び出してメモリを解放する。"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def _compute_embeddings(models, loader: DataLoader, config) -> Tuple[np.ndarray, np.ndarray]:
    """ローダ全体をエンコードし (特徴行列, ラベルベクトル) を返す。"""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for m in models.values():
        m.eval()
    with torch.no_grad():
        for batch in loader:
            a_vec, a, *_ = _encode_batch(models, batch, config)
            xs.append(a_vec.detach().cpu().numpy())
            ys.append(a.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def _compute_clustering_metrics(models, loader, config):
    """埋め込みにKMeansクラスタリングを行い ARI/NMI/平均 を返す。"""
    X, y = _compute_embeddings(models, loader, config)
    n_clusters = len(np.unique(y))
    pred = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)
    return ari, nmi, (ari + nmi) / 2