from collections import defaultdict
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gc
from triplet_losses import ImprovedTripletLoss, grl
import torch.nn.functional as F
import numpy as np
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
    ActionClassifier, ActionMLPNet,
    SpeciesDiscriminator, GatedFusion
)
from sklearn.cluster import AgglomerativeClustering
from pytorch_metric_learning import losses, miners, distances
import optuna
import wandb
from sklearn.metrics import matthews_corrcoef
import itertools
from sklearn.mixture import GaussianMixture

from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.miners import MultiSimilarityMiner

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
    triplet_margin: float = 0.1,
):
    miner = None

    if loss_type == "cosine":
        # --- Triplet (そのまま) ---
        dist = distances.CosineSimilarity()
        loss_fn = losses.TripletMarginLoss(
            distance=dist,
            margin=triplet_margin
        )
        miner = miners.TripletMarginMiner(
            margin=triplet_margin,
            distance=dist,
            type_of_triplets="semihard"
        )

    else:
        # --- Multi-Similarity (公式構成) ---
        loss_fn = MultiSimilarityLoss(
            alpha=8.0,
            beta=40.0,
            base=0.3,
            distance=CosineSimilarity(),
        )
        miner = None

    return loss_fn, miner



def compute_distance_stats_epoch(
    embeddings: torch.Tensor,  # [N, D]
    labels: torch.Tensor,      # [N]
):
    """
    valid embedding から
    intra / inter 距離の mean / var を計算
    """
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()

    intra_dists = []
    inter_dists = []

    unique_labels = labels.unique()

    # ---- intra-class ----
    for lbl in unique_labels:
        vecs = embeddings[labels == lbl]
        if vecs.size(0) < 2:
            continue

        # if vecs.size(0) > max_per_class:
        #     vecs = vecs[torch.randperm(vecs.size(0))[:max_per_class]]

        sim = torch.mm(vecs, vecs.t())
        dist = 1.0 - sim
        mask = ~torch.eye(dist.size(0), dtype=torch.bool)
        intra_dists.append(dist[mask])

    # ---- inter-class ----
    for la, lb in itertools.combinations(unique_labels, 2):
        va = embeddings[labels == la]
        vb = embeddings[labels == lb]
        if va.size(0) == 0 or vb.size(0) == 0:
            continue

        # if va.size(0) > max_per_class:
        #     va = va[torch.randperm(va.size(0))[:max_per_class]]
        # if vb.size(0) > max_per_class:
        #     vb = vb[torch.randperm(vb.size(0))[:max_per_class]]

        sim = torch.mm(va, vb.t())
        dist = 1.0 - sim
        inter_dists.append(dist.flatten())

    if not intra_dists or not inter_dists:
        return None

    intra_all = torch.cat(intra_dists)
    inter_all = torch.cat(inter_dists)

    return {
        "intra_mean": intra_all.mean().item(),
        "intra_var": intra_all.var(unbiased=False).item(),
        "inter_mean": inter_all.mean().item(),
        "inter_var": inter_all.var(unbiased=False).item(),
    }



def compute_species_prior(df, le_sp, device):
    """
    train_df に基づく species prior を作る
    - le_sp.classes_ の順序・次元に必ず一致
    - train_df に存在しない種は確率 
    - KL(q || p) 用に安全
    """
    sp_ids = torch.tensor(df["species"].values, device=device, dtype=torch.long)
    counts = torch.bincount(sp_ids, minlength=len(le_sp.classes_)).float()
    prior = counts / counts.sum().clamp_min(1.0)

    return prior

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
        fusion_model = GatedFusion(2048, 768, ).to(DEVICE)

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
        
        x3d = x3d.to(DEVICE).squeeze(1)
        vmae = vmae.to(DEVICE)
        a = a.to(DEVICE, dtype=torch.long)
        s = s.to(DEVICE, dtype=torch.long)

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
    # a_vec = torch.nn.functional.normalize(a_vec, p=2, dim=1)
    return a_vec, a, s, alpha


def train_step(
    models: nn.ModuleDict,
    batch,
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    config: Dict[str, Any],
    le_sp: LabelEncoder,
    lambda_p: float,
    species_prior: Optional[torch.Tensor] = None,
    current_step: int = 0,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:

    metrics = {}
    for m in models.values():
        m.train()
    # ===== DEBUG: input similarity check (only once) =====
    if current_step == 0:
        if config["train_mode"] == "gated":
            x3d_dbg, vmae_dbg, a_dbg, s_dbg = batch

            # X3D
            x3d_dbg = x3d_dbg.to(DEVICE).squeeze(1).float()
            x3d_norm = F.normalize(x3d_dbg, dim=1)
            sim_x3d = x3d_norm @ x3d_norm.T
            print(
                f"⚠️ INPUT X3D  : mean={sim_x3d.mean():.4f}, "
                f"var={sim_x3d.var(unbiased=False):.6f}"
            )

            # VMAE
            vmae_dbg = vmae_dbg.to(DEVICE).float()
            vmae_norm = F.normalize(vmae_dbg, dim=1)
            sim_vmae = vmae_norm @ vmae_norm.T
            print(
                f"⚠️ INPUT VMAE : mean={sim_vmae.mean():.4f}, "
                f"var={sim_vmae.var(unbiased=False):.6f}"
            )

        else:
            x_dbg, a_dbg, s_dbg = batch
            x_dbg = x_dbg.to(DEVICE).float()
            x_norm = F.normalize(x_dbg, dim=1)
            sim = x_norm @ x_norm.T
            print(
                f"⚠️ INPUT FEAT : mean={sim.mean():.4f}, "
                f"var={sim.var(unbiased=False):.6f}"
            )
    # ====================================================

    # 特徴抽出 (Forward)
    a_vec, a, s, alpha = _encode_batch(models, batch, config)
    adv_mode = config.get("adversarial", "off")
    adv_enabled = adv_mode != "off"
    lam = float(config.get("lambda_adv", 0.1))

    # ==========================================
    # (1) Discriminator の更新 (種を当てたい)
    # ==========================================
    if adv_enabled:
        opt_disc = optimizers["discriminator"]
        opt_disc.zero_grad()

        # a_vec.detach() で Encoder への勾配逆流を遮断
        logits_disc = models["discriminator"](a_vec.detach())
        disc_loss = nn.CrossEntropyLoss()(logits_disc, s)

        # 指標の計算 (Acc & MCC)
        pred_disc = logits_disc.argmax(dim=1)
        metrics["adv_disc/loss"] = disc_loss.item()
        metrics["adv_disc/acc"] = (pred_disc == s).float().mean().item()
        metrics["adv_disc/mcc"] = matthews_corrcoef(
            s.detach().cpu().numpy(),
            pred_disc.detach().cpu().numpy()
        )

        disc_loss.backward()
        opt_disc.step()

    # ==========================================
    # (2) Encoder の更新 (行動分離 + 種の隠蔽)
    # ==========================================
    main_opt = optimizers.get("encoder") or optimizers.get("main")
    main_opt.zero_grad()

    # 1. Triplet Loss 
    if miner is not None:
        # Triplet Loss系: マイナーが作ったペア(hard)を渡す
        hard = miner(a_vec, a)
        total_loss = loss_fn(a_vec, a, hard)
    else:
        # Multi Similarity Loss系: 
        # Loss関数内部で全ペアを計算するため、第3引数(None)は不要です
        total_loss = loss_fn(a_vec, a)
    
    metrics["triplet/loss"] = total_loss.item()

    # 2. 行動分類 CE (もし lambda_cls が設定されていれば)
    lambda_cls = float(config.get("lambda_cls", 0.0))
    if lambda_cls > 0:
        logits_act = models["action_classifier"](a_vec)
        ce_act = nn.CrossEntropyLoss()(logits_act, a)
        metrics["action_ce/loss"] = ce_act.item()
        metrics["action_ce/acc"] = (logits_act.argmax(dim=1) == a).float().mean().item()
        metrics["action_ce/mcc"] = matthews_corrcoef(
            a.detach().cpu().numpy(),
            logits_act.argmax(dim=1).detach().cpu().numpy()
        )
        total_loss = total_loss + lambda_cls * ce_act

    # 3. Adversarial (GRLを使用)
    if adv_enabled:
        if adv_mode == "dann":
            rev = grl(a_vec, lam)
            logits_enc = models["discriminator"](rev)
            adv_loss = nn.CrossEntropyLoss()(logits_enc, s)
            metrics["adv_enc/loss"] = adv_loss.item()
            total_loss = total_loss + adv_loss
        elif adv_mode == "kl":
            logits_enc = models["discriminator"](a_vec)
            logp = nn.functional.log_softmax(logits_enc, dim=1)
            prior = species_prior.unsqueeze(0).expand_as(logp)

            adv_loss = nn.KLDivLoss(reduction="batchmean")(logp, prior)
        
            metrics["adv_enc/loss"] = adv_loss.item()
            total_loss = total_loss + lam * adv_loss

    # 4. Encoder 側の Backward & Step
    total_loss.backward()

    enc = models["action_encoder"] if "action_encoder" in models else models["net"]
    nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
    main_opt.step()

    alpha_np = alpha.detach().cpu().numpy() if alpha is not None else None
    return metrics, alpha_np


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


def evaluate_model(
    models: nn.ModuleDict,
    loader: DataLoader,
    config: Dict[str, Any],
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
) -> float:
    """
    valid でも train と同じく
    metric loss + lambda_cls * CE を使って val_loss を計算
    """
    for m in models.values():
        m.eval()
    losses_acc: List[float] = []
    lambda_cls = float(config.get("lambda_cls", 0.0))

    with torch.no_grad():
        for batch in loader:
            a_vec, a, _s, _alpha = _encode_batch(models, batch, config)


            if isinstance(loss_fn, MultiSimilarityLoss):
                 loss = loss_fn(a_vec, a)
            else:
                loss = loss_fn(a_vec, a)
            # 行動 CE も加える（lambda_cls = 0 の時は実質無効）
            if lambda_cls > 0:
                logits_act = models["action_classifier"](a_vec)
                ce_act = nn.CrossEntropyLoss()(logits_act, a)
                loss = loss + lambda_cls * ce_act

            losses_acc.append(float(loss.item()))

    return float(np.mean(losses_acc)) if losses_acc else float("inf")

def get_dann_lambda(p: float, gamma: float = 5.0) -> float:
    """
    論文 (Ganin et al., 2016) に基づく λ のスケジューリング
    p: 学習の進捗 (0.0 -> 1.0)
    gamma: スケジュールのカーブ形状 (論文では10)
    return: 0.0 -> 1.0 に変化する係数
    """
    return 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0

def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    le_sp: LabelEncoder,
    le_act: LabelEncoder,
    trial,
    study_name: str,
    fusion: Optional[nn.Module] = None,
    results_root: Optional[Path] = None,
    run_name_override: Optional[str] = None,
    is_ablation: bool = False,
    ablation_subdir: Optional[str] = None,
    fold_idx: int = 0,
    log_fold: int = 1,
) -> float:
    """モデル学習ループ（Optuna・アブレーション共通）"""

    species_prior = compute_species_prior(train_loader.dataset.df, le_sp, DEVICE)

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

    num_species = len(le_sp.classes_)
    num_actions = len(le_act.classes_)

    # -------------------------------
    # モデル・オプティマイザ構築
    # -------------------------------
    if adv_enabled:
        models["action_encoder"] = ActionMLPNet(input_dim=D, feature_dim=128, hidden_dim=512).to(DEVICE)
        models["discriminator"] = SpeciesDiscriminator(128, num_species).to(DEVICE)
        models["action_classifier"] = ActionClassifier(128, num_actions).to(DEVICE)

        params_enc = list(models["action_encoder"].parameters()) + \
                     list(models["action_classifier"].parameters())
        if fusion is not None:
            models["fusion"] = fusion.to(DEVICE)
            params_enc.extend(models["fusion"].parameters())

        enc_lr = float(config.get("lr_enc", 1e-4))
        disc_lr = float(config.get("lr_disc", enc_lr))
        optimizers["encoder"] = torch.optim.Adam(
            params_enc,
            lr=enc_lr,
            weight_decay=wd,
        )
        optimizers["discriminator"] = torch.optim.Adam(
            models["discriminator"].parameters(),
            lr=disc_lr,
            weight_decay=wd,
        )
    else:
        models["net"] = ActionMLPNet(input_dim=D, feature_dim=128, hidden_dim=512).to(DEVICE)
        models["action_classifier"] = ActionClassifier(128, num_actions).to(DEVICE)

        params = list(models["net"].parameters()) + \
                 list(models["action_classifier"].parameters())
        if fusion is not None:
            models["fusion"] = fusion.to(DEVICE)
            params.extend(models["fusion"].parameters())

        enc_lr = float(config.get("lr_enc", 1e-4))
        optimizers["main"] = torch.optim.Adam(
            params,
            lr=enc_lr,
            weight_decay=wd,
        )
    
    target_optimizer = optimizers["main"] if "main" in optimizers else optimizers["encoder"]

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     target_optimizer,
    #     milestones=[50, 80],  # ← epoch基準（※あとで調整）
    #     gamma=0.1
    # )

    scheduler=None


    # -------------------------------
    # 損失関数と miner
    # -------------------------------
    loss_fn, miner = get_loss_fn_and_miner(
        str(config["loss_type"]),
        triplet_margin=float(config.get("triplet_margin")),
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
            f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}"
            f"_adv{config['adversarial']}_pool{config.get('pooling', True)}"
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

    total_steps = max_epochs * len(train_loader)
    current_step = 0


    for epoch in range(max_epochs):
        epoch_metrics = defaultdict(list)
        alpha_parts = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}"):

            p = current_step / total_steps
            lambda_p = get_dann_lambda(p)

            metrics, alpha_np = train_step(
                models, batch, loss_fn, miner,
                optimizers, config, le_sp, lambda_p, species_prior, current_step
            )

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            if alpha_np is not None:
                alpha_parts.append(alpha_np)

            current_step += 1

        # ---- validation ----
        val_loss = evaluate_model(models, val_loader, config, loss_fn, miner)

        # ---- collect valid embeddings (epoch-wise) ----
        val_embeds = []
        val_labels = []

        for m in models.values():
            m.eval()

        with torch.no_grad():
            for batch in val_loader:
                a_vec, a, _s, _alpha = _encode_batch(models, batch, config)
                val_embeds.append(a_vec.detach().cpu())
                val_labels.append(a.detach().cpu())

        val_embeds = torch.cat(val_embeds, dim=0)   # [N, D]
        val_labels = torch.cat(val_labels, dim=0)   # [N]

        dist_stats = compute_distance_stats_epoch(
            embeddings=val_embeds,
            labels=val_labels,
        )

        log_dict = {
            "valid/loss": val_loss,
        }

        if dist_stats is not None:
            log_dict.update({
                "valid/intra_mean": dist_stats["intra_mean"],
                "valid/intra_var": dist_stats["intra_var"],
                "valid/inter_mean": dist_stats["inter_mean"],
                "valid/inter_var": dist_stats["inter_var"],
                "valid/separation_ratio": (
                    dist_stats["inter_mean"] / (dist_stats["intra_mean"] + 1e-8)
                ),
            })

        for k, v in epoch_metrics.items():
            log_dict[f"train/{k}"] = float(np.mean(v))

        # ---- alpha statistics (gated only) ----
        if alpha_parts:
            alpha_epoch = np.concatenate(alpha_parts, axis=0)  # [N, D]

            wandb.log({
                "alpha/mean": float(alpha_epoch.mean()),
                "alpha/std":  float(alpha_epoch.std()),
                "alpha/min":  float(alpha_epoch.min()),
                "alpha/max":  float(alpha_epoch.max()),
            }, step=epoch)


        if fold_idx == log_fold:
            wandb.log(log_dict, step=epoch)

        # ---- early stopping ----
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(models.state_dict(), save_path)
            trial.set_user_attr("model_save_path", str(save_path))
            # ===============================
            # class × class table（trial 1回）
            # ===============================
            df_mean, df_var = compute_classwise_distance_tables(
                val_embeds,
                val_labels,
                le_act,
                max_per_class=50,
            )

            log_classwise_tables_once_per_trial(
                df_mean,
                df_var,
                epoch=epoch + 1,
                trial_number=trial.number,
                prefix="valid",
            )
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

        if scheduler is not None:
            scheduler.step()

    return best_val


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
        models["action_encoder"] = ActionMLPNet(input_dim=D, feature_dim=128, hidden_dim=512).to(DEVICE)
    else:
        models["net"] = ActionMLPNet(input_dim=D, feature_dim=128, hidden_dim=512).to(DEVICE)
    if fusion is not None:
        models["fusion"] = fusion.to(DEVICE)
    return models


def _compute_embeddings(models: nn.ModuleDict, loader: DataLoader, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for m in models.values():
        m.eval()
    with torch.no_grad():
        for batch in loader:
            a_vec, a, _s, _alpha = _encode_batch(models, batch, config)
            xs.append(a_vec.detach().cpu().numpy())
            ys.append(a.detach().cpu().numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _compute_clustering_metrics(
    models: nn.ModuleDict,
    loader: DataLoader,
    config: Dict[str, Any],
):
    X, y = _compute_embeddings(models, loader, config)
    y = np.asarray(y)

    n_clusters = len(np.unique(y))
    if n_clusters <= 1:
        return {
            "agg_ari": 0.0, "agg_nmi": 0.0,
            "km_ari": 0.0,  "km_nmi": 0.0,
            "gmm_ari": 0.0, "gmm_nmi": 0.0,
        }

    results = {}

    # =========================
    # Agglomerative (MAIN)
    # =========================
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    pred_agg = agg.fit_predict(X)
    results["agg_ari"] = float(adjusted_rand_score(y, pred_agg))
    results["agg_nmi"] = float(normalized_mutual_info_score(y, pred_agg))

    # # =========================
    # # KMeans (reference)
    # # =========================
    # km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # pred_km = km.fit_predict(X)
    # results["km_ari"] = float(adjusted_rand_score(y, pred_km))
    # results["km_nmi"] = float(normalized_mutual_info_score(y, pred_km))

    # # =========================
    # # GMM (reference)
    # # =========================
    # gmm = GaussianMixture(
    #     n_components=n_clusters,
    #     covariance_type="full",
    #     random_state=42,
    # )
    # pred_gmm = gmm.fit_predict(X)
    # results["gmm_ari"] = float(adjusted_rand_score(y, pred_gmm))
    # results["gmm_nmi"] = float(normalized_mutual_info_score(y, pred_gmm))

    return results



import torch
import pandas as pd

def compute_classwise_distance_tables(
    embeddings: torch.Tensor,   # [N, D]
    labels: torch.Tensor,       # [N]
    label_encoder,
    max_per_class: int = 50,
):
    """
    クラス×クラスの
    - mean cosine distance
    - variance of cosine distance
    を DataFrame で返す
    """

    # ---- normalize (Cosine必須) ----
    # embeddings = torch.nn.functional.normalize(
    #     embeddings.detach().cpu(), p=2, dim=1
    # )
    labels = labels.detach().cpu()

    unique_labels = sorted(labels.unique().tolist())
    
    # 安全策: ラベルIDが範囲外でないかチェック
    class_names = []
    for i in unique_labels:
        if 0 <= i < len(label_encoder.classes_):
            class_names.append(label_encoder.classes_[i])
        else:
            class_names.append(f"class_{i}")

    # ---- collect vectors per class ----
    class_vecs = {}
    for lbl in unique_labels:
        vecs = embeddings[labels == lbl]
        if vecs.size(0) == 0:
            continue
        if vecs.size(0) > max_per_class:
            vecs = vecs[torch.randperm(vecs.size(0))[:max_per_class]]
        class_vecs[lbl] = vecs

    n = len(unique_labels)
    mean_mat = torch.zeros(n, n)
    var_mat  = torch.zeros(n, n)

    for ii, i in enumerate(unique_labels):
        vi = class_vecs.get(i)
        for jj, j in enumerate(unique_labels):
            vj = class_vecs.get(j)
            
            if vi is None or vj is None:
                continue

            # sim: (Ni, Nj)
            sim_matrix = torch.mm(vi, vj.t())
            
            # distance = 1.0 - sim
            dist_vals = 1.0 - sim_matrix
            dist_vals = torch.clamp(dist_vals, 0.0, 2.0)

            # intra: self-distance (対角成分=0) を除外して計算
            if i == j:
                mask = ~torch.eye(dist_vals.size(0), dtype=torch.bool)
                vals = dist_vals[mask]
                # データが1つしかない場合は分散計算できないので0埋め
                if vals.size(0) == 0:
                    mean_mat[ii, jj] = 0.0
                    var_mat[ii, jj]  = 0.0
                    continue
            else:
                vals = dist_vals.flatten()

            mean_mat[ii, jj] = vals.mean()
            var_mat[ii, jj]  = vals.var(unbiased=False)

    df_mean = pd.DataFrame(
        mean_mat.numpy(),
        index=class_names,
        columns=class_names,
    )
    df_var = pd.DataFrame(
        var_mat.numpy(),
        index=class_names,
        columns=class_names,
    )

    return df_mean, df_var


def log_classwise_tables_once_per_trial(
    df_mean,
    df_var,
    epoch: int,
    trial_number: int,
    prefix: str = "valid",
):
    # ===== Table =====
    wandb.log({
        f"{prefix}/classwise_mean_table": wandb.Table(
            dataframe=df_mean.reset_index()
        ),
        f"{prefix}/classwise_var_table": wandb.Table(
            dataframe=df_var.reset_index()
        ),
        "epoch": epoch,
    })

    # ===== Artifact =====
    tmp_dir = Path("wandb_tables_tmp")
    tmp_dir.mkdir(exist_ok=True)

    mean_path = tmp_dir / f"classwise_mean_trial{trial_number:03d}_epoch{epoch:03d}.csv"
    var_path  = tmp_dir / f"classwise_var_trial{trial_number:03d}_epoch{epoch:03d}.csv"

    df_mean.to_csv(mean_path)
    df_var.to_csv(var_path)

    artifact = wandb.Artifact(
        name=f"classwise_distance_tables_trial{trial_number:03d}_epoch{epoch:03d}",
        type="analysis",
        metadata={
            "trial": trial_number,
            "epoch": epoch,
        }
    )

    artifact.add_file(str(mean_path))
    artifact.add_file(str(var_path))
    wandb.log_artifact(artifact)
