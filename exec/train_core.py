import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gc
from triplet_losses import grl
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import set_seed
from utils import MAE_Dataset,X3D_Dataset, X3D_MAE_Dataset
from model import (
    SimpleLinearNet, SimpleMLPNet, ActionLinearNet, ActionMLPNet,
    SpeciesDiscriminator, GatedFusion
)
from pytorch_metric_learning import losses, miners, distances
import optuna
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOP_PATIENCE = 50



def cleanup_memory() -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


# ---------------- Loss and miner -----------------

def get_loss_fn_and_miner(
    loss_type: str,
    temperature: float = 0.07,
    triplet_margin: float = 0.1,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    miner: Optional[nn.Module] = None
    if loss_type == "supcon":
        loss_fn = losses.SupConLoss(temperature=temperature)
    elif loss_type == "cosine":
        distance = distances.CosineSimilarity()
        loss_fn = losses.TripletMarginLoss(distance=distance, margin=triplet_margin)
        miner = miners.TripletMarginMiner(margin=triplet_margin, distance=distance, type_of_triplets="hard")
    else:  # default triplet
        loss_fn = losses.TripletMarginLoss(margin=triplet_margin)
        miner = miners.TripletMarginMiner(margin=triplet_margin, type_of_triplets="hard")
    return loss_fn, miner


# --------------- Data and loaders ----------------

def build_datasets_and_loaders(
    config: Dict[str, Any],
    train_df,
    val_df,
    le_act: LabelEncoder,
    le_sp: LabelEncoder,
) -> Tuple[DataLoader, DataLoader, Optional[nn.Module]]:

    pooling  = bool(config.get("pooling", True))
    centered = (config.get("flow_preprocessing", "normal") == "centered")
    train_mode = config.get("train_mode", "gated")

    fusion_model: Optional[nn.Module] = None

    if train_mode == "mae":
        train_dataset = MAE_Dataset(train_df, le_act, le_sp, pooling=pooling)
        val_dataset   = MAE_Dataset(val_df,   le_act, le_sp, pooling=pooling)

    elif train_mode == "flow":
        train_dataset = X3D_Dataset(train_df, le_act, le_sp,
                                    centered=centered, pooling=pooling)
        val_dataset   = X3D_Dataset(val_df,   le_act, le_sp,
                                    centered=centered, pooling=pooling)
    elif train_mode == "gated":
        train_dataset = X3D_MAE_Dataset(train_df, le_act, le_sp,
                                        centered=centered, pooling=pooling)
        val_dataset   = X3D_MAE_Dataset(val_df,   le_act, le_sp,
                                        centered=centered, pooling=pooling)

        # X3D_dim=2048, MAE_dim=768 → config["fused_dim"]
        fusion_model = GatedFusion(2048, 768, int(config["fused_dim"]))
        fusion_model = fusion_model.to(DEVICE)

    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    # ----------------------------
    # Loader
    # ----------------------------
    workers     = int(config.get("num_workers", 0))
    pin_memory  = bool(config.get("pin_memory", torch.cuda.is_available()))
    batch_size  = int(config.get("batch_size", 32))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, fusion_model



# --------------- Forward helpers ----------------

def _encode_batch(models: nn.ModuleDict, batch, config: Dict[str, Any]):
    if config["train_mode"] == "gated":
        x3d, vmae, a, s = batch
        x3d = x3d.to(DEVICE)
        vmae = vmae.to(DEVICE)
        a = a.to(DEVICE, dtype=torch.long)
        s = s.to(DEVICE, dtype=torch.long)
        
        # Flatten if not pooled (Batch, Frames, Dim) -> (Batch*Frames, Dim)
        if x3d.dim() == 3:
            b, t, d = x3d.shape
            x3d = x3d.reshape(b * t, d)
            vmae = vmae.reshape(b * t, -1)
            # Expand labels: (Batch,) -> (Batch, Frames) -> (Batch*Frames,)
            a = a.unsqueeze(1).expand(b, t).reshape(-1)
            s = s.unsqueeze(1).expand(b, t).reshape(-1)

        fused, alpha = models["fusion"](x3d, vmae)
        x = fused
    else:
        x, a, s = batch
        x = x.to(DEVICE)
        a = a.to(DEVICE, dtype=torch.long)
        s = s.to(DEVICE, dtype=torch.long)
        
        # Flatten if not pooled
        if x.dim() == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            a = a.unsqueeze(1).expand(b, t).reshape(-1)
            s = s.unsqueeze(1).expand(b, t).reshape(-1)

        alpha = None
    encoder = models["action_encoder"] if "action_encoder" in models else models["net"]
    a_vec = encoder(x)
    return a_vec, a, s, alpha


def train_step(
    models: nn.ModuleDict,
    batch,
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    config: Dict[str, Any],
    le_sp: LabelEncoder,
) -> Tuple[float, Optional[np.ndarray]]:
    for m in models.values():
        m.train()

    a_vec, a, s, alpha = _encode_batch(models, batch, config)
    adv_mode = config.get("adversarial", "off")
    adv_enabled = adv_mode != "off"

    if adv_enabled:
        discriminator = models["discriminator"]
        opt_disc = optimizers["discriminator"]
        for _ in range(int(config.get("disc_steps_per_batch", 1))):
            opt_disc.zero_grad()
            logits_disc = discriminator(a_vec.detach())
            ce = nn.CrossEntropyLoss()(logits_disc, s)
            ce.backward()
            opt_disc.step()

    main_opt = optimizers.get("encoder") or optimizers.get("main")
    if main_opt is None:
        raise RuntimeError("Missing main optimizer")
    main_opt.zero_grad()

    if config["loss_type"] == "supcon":
        main_loss = loss_fn(a_vec, labels=a)
    elif miner is not None:
        hard = miner(a_vec, a)
        main_loss = loss_fn(a_vec, a, hard)
    else:
        main_loss = loss_fn(a_vec, a)

    total = main_loss
    if adv_enabled:
        logits = models["discriminator"](a_vec)
        if adv_mode == "kl":
            logp = nn.functional.log_softmax(logits, dim=1)
            uniform = torch.full_like(logits, 1.0 / len(le_sp.classes_))
            adv_loss = nn.KLDivLoss(reduction="batchmean")(logp, uniform)
            total = total + float(config["lambda_adv"]) * adv_loss
        elif adv_mode == "gan":
            ce_enc = nn.CrossEntropyLoss()(logits, s)
            total = total - float(config["lambda_adv"]) * ce_enc
        elif adv_mode == "dann":
            reversed_vec = grl(a_vec, float(config["lambda_adv"]))
            logits_dann = models["discriminator"](reversed_vec)

            adv_loss = nn.CrossEntropyLoss()(logits_dann, s)
            total = total + adv_loss

    total.backward()
    enc = models["action_encoder"] if "action_encoder" in models else models["net"]
    nn.utils.clip_grad_norm_(enc.parameters(), max_norm=5.0)
    if "fusion" in models:
        nn.utils.clip_grad_norm_(models["fusion"].parameters(), max_norm=5.0)
    main_opt.step()

    alpha_np = alpha.detach().cpu().numpy() if alpha is not None else None
    return float(total.item()), alpha_np


def evaluate_model(models: nn.ModuleDict, loader: DataLoader, config: Dict[str, Any], loss_fn: nn.Module, miner: Optional[nn.Module]) -> float:
    for m in models.values():
        m.eval()
    losses_acc: List[float] = []
    with torch.no_grad():
        for batch in loader:
            a_vec, a, _s, _alpha = _encode_batch(models, batch, config)
            if config["loss_type"] == "supcon":
                loss = loss_fn(a_vec, labels=a)
            elif miner is not None:
                hard = miner(a_vec, a)
                loss = loss_fn(a_vec, a, hard)
            else:
                loss = loss_fn(a_vec, a)
            losses_acc.append(float(loss.item()))
    return float(np.mean(losses_acc)) if losses_acc else float("inf")


def save_alpha_epoch(alpha_parts: List[np.ndarray], epoch: int, config: Dict[str, Any], trial: optuna.trial.Trial, results_root: Optional[Path], is_ablation: bool) -> None:
    """gated モード時に収集した α を一時保存する。アブレーション時はスキップ。"""
    if config.get("train_mode") != "gated" or not alpha_parts or is_ablation:
        return
    try:
        alpha_epoch = np.concatenate(alpha_parts, axis=0)
    except Exception:
        print("⚠️ Failed to concatenate alpha parts; skip saving.")
        return
    if results_root is not None:
        alpha_tmp_dir = Path(results_root) / 'alpha_logs_tmp' / f"trial_{trial.number:03d}"
    else:
        date_dir = datetime.now().strftime("%Y-%m-%d")
        alpha_tmp_dir = Path('./train_result') / date_dir / 'alpha_logs_tmp' / f"trial_{trial.number:03d}"
    alpha_tmp_dir.mkdir(parents=True, exist_ok=True)
    alpha_tmp_path = alpha_tmp_dir / f"alpha_trial{trial.number:03d}_epoch{epoch:03d}.npy"
    try:
        np.save(str(alpha_tmp_path), alpha_epoch)
    except Exception as e:
        print(f"⚠️ Failed to save temp alpha for trial #{trial.number}, epoch {epoch}: {e}")


def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    le_sp: LabelEncoder,
    trial,
    study_name: str,
    fusion: Optional[nn.Module] = None,
    results_root: Optional[Path] = None,
    run_name_override: Optional[str] = None,
    is_ablation: bool = False,
    ablation_subdir: Optional[str] = None,
) -> float:
    """モデル学習ループ（Optuna・アブレーション共通）"""

    sample = next(iter(train_loader))[0]
    train_mode = config.get("train_mode")
    if train_mode == "gated":
        D = int(config["fused_dim"])
    elif train_mode == "flow":
        D = 2048
    elif train_mode == "mae":
        D = 768
    else:
        raise ValueError("Unknown train_mode")

    models = nn.ModuleDict()
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    wd = float(config.get("weight_decay", 1e-5))
    adv_mode = config.get("adversarial", "off")
    adv_enabled = adv_mode != "off"

    # -------------------------------
    # モデル・オプティマイザ構築
    # -------------------------------
    if adv_enabled:
        models["action_encoder"] = (
            ActionMLPNet(D, 256, 256) 
        ).to(DEVICE)
        models["discriminator"] = SpeciesDiscriminator(256, len(le_sp.classes_)).to(DEVICE)
        params_enc = list(models["action_encoder"].parameters())
        if fusion is not None:
            models["fusion"] = fusion.to(DEVICE)
            params_enc.extend(models["fusion"].parameters())
        enc_lr = float(config.get("lr_enc", 1e-4))
        disc_lr = float(config.get("lr_disc", enc_lr))
        optimizers["encoder"] = torch.optim.Adam(params_enc, lr=enc_lr, weight_decay=wd)
        optimizers["discriminator"] = torch.optim.Adam(
            models["discriminator"].parameters(), lr=disc_lr, weight_decay=wd
        )
    else:
        models["net"] = (   
            SimpleMLPNet(D, 256, 256)
        ).to(DEVICE)
        params = list(models["net"].parameters())
        if fusion is not None:
            models["fusion"] = fusion.to(DEVICE)
            params.extend(models["fusion"].parameters())
        enc_lr = float(config.get("lr_enc", 1e-4))
        optimizers["main"] = torch.optim.Adam(params, lr=enc_lr, weight_decay=wd)

    # -------------------------------
    # 損失関数とminer
    # -------------------------------
    loss_fn, miner = get_loss_fn_and_miner(
        str(config["loss_type"]),
        temperature=float(config.get("temperature", 0.07)),
        triplet_margin=float(config.get("triplet_margin", 0.1)),
    )

    # -------------------------------
    # 保存ディレクトリ設定
    # -------------------------------
    if results_root is None:
        results_root = Path("./train_result")
    results_root.mkdir(parents=True, exist_ok=True)

    model_dir = results_root / "checkpoints" / study_name
    if is_ablation and ablation_subdir:
        model_dir = model_dir / ablation_subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # モデル名生成
    # -------------------------------
    if run_name_override:
        run_name = run_name_override
    else:
        run_name = (
            f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}_adv{config['adversarial']}_pool{config.get('pooling', True)}"
            + (f"_{config['flow_preprocessing']}" if config.get("train_mode") in ["flow", "gated"] else "")
        )

    save_path = model_dir / f"{run_name}_best.pth"

    # -------------------------------
    # メイントレーニングループ
    # -------------------------------
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    max_epochs = int(config.get("epochs", 500))

    for epoch in range(max_epochs):
        batch_losses: List[float] = []
        alpha_parts: List[np.ndarray] = []  # ← 追加: alphaを格納するリスト

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}"):
            loss_val, alpha_np = train_step(models, batch, loss_fn, miner, optimizers, config, le_sp)
            batch_losses.append(loss_val)
            if alpha_np is not None:
                alpha_parts.append(alpha_np)

        # GatedFusion α 保存
        save_alpha_epoch(alpha_parts, epoch, config, trial, results_root, is_ablation)

        # 検証
        val_loss = evaluate_model(models, val_loader, config, loss_fn, miner)

        # ベストモデル保存とearly stopping
        if val_loss < best_val:
            best_val, best_epoch, no_improve = val_loss, epoch + 1, 0
            torch.save(models.state_dict(), save_path)
            trial.set_user_attr("model_save_path", str(save_path))
            trial.set_user_attr("best_epoch", int(best_epoch))
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    trial.set_user_attr("epochs_run", int(best_epoch))
    return float(best_val)


def build_basename_from_config(cfg: Dict[str, Any]) -> str:
    """
    config_search.yml の設定内容から自動でファイル名を構築する。

    例:
    loss_type: triplet
    train_mode: gated
    adversarial: gan
    flow_preprocessing: centered

    → "triplet_gated_adv_gan_centered"
    """
    parts: List[str] = []

    # === 損失関数 ===
    parts.append(str(cfg.get("loss_type", "unknown")))

    # === 学習モード ===
    parts.append(str(cfg.get("train_mode", "unknown")))

    # === 敵対的学習モード ===
    adv_mode = cfg.get("adversarial") or "off"
    parts.append(f"adv_{adv_mode}")

    # === Flow前処理 ===
    if cfg.get("train_mode") in ["flow", "gated"]:
        parts.append(str(cfg.get("flow_preprocessing", "normal")))

    # === その他パラメータ（任意）===
    if "triplet_margin" in cfg:
        parts.append(f"margin{cfg['triplet_margin']}")
    if "lambda_adv" in cfg:
        parts.append(f"lam{cfg['lambda_adv']}")

    return "_".join(map(str, parts))

class DummyTrial:
    def __init__(self, number: int = -1):
        self.number = number
        self.user_attrs: Dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value

    def report(self, value: float, step: int) -> None:
        # No-op for compatibility; optionally store minimal trace
        try:
            logs = self.user_attrs.setdefault("_reports", [])
            logs.append((int(step), float(value)))
        except Exception:
            pass

    def should_prune(self) -> bool:
        return False


# ---------- Inference and clustering metrics ----------

def _build_inference_models(config: Dict[str, Any], D: int, fusion: Optional[nn.Module] = None) -> nn.ModuleDict:
    models = nn.ModuleDict()
    if config.get("adversarial", "off") != "off":
        models["action_encoder"] = ActionMLPNet(D, 256, 256).to(DEVICE)
    else:
        models["net"] = SimpleMLPNet(D, 256, 256).to(DEVICE)
    if fusion is not None:
        models["fusion"] = fusion.to(DEVICE)
    return models


def _compute_embeddings(models: nn.ModuleDict, loader: DataLoader, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []; ys: List[np.ndarray] = []
    for m in models.values(): m.eval()
    with torch.no_grad():
        for batch in loader:
            a_vec, a, _s, _alpha = _encode_batch(models, batch, config)
            xs.append(a_vec.detach().cpu().numpy()); ys.append(a.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _compute_clustering_metrics(models: nn.ModuleDict, loader: DataLoader, config: Dict[str, Any]):
    X, y = _compute_embeddings(models, loader, config)
    n_clusters = len(np.unique(y))
    if n_clusters <= 1:
        return 0.0, 0.0, 0.0
    pred = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
    ari = adjusted_rand_score(y, pred)
    nmi = normalized_mutual_info_score(y, pred)
    return float(ari), float(nmi), float((ari + nmi) / 2.0)
