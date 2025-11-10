import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb
import optuna
from optuna.trial import TrialState
try:
    from optuna.storages import InMemoryStorage 
except Exception:
    InMemoryStorage = None
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import shutil
import csv
from urllib import request as urlrequest, error as urlerror
import gc
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# 必要なファイルをインポート
from utils import FlowNpyDataset, X3DVideoMAEDataset, MAEDataset, discord_notify, set_seed,cleanup_memory
from model import (
    SimpleLinearNet, SimpleMLPNet, ActionLinearNet, ActionMLPNet,
    SpeciesDiscriminator, GatedFusion
)
from pytorch_metric_learning import losses, miners, distances
from dotenv import load_dotenv

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Global settings and small utils
# -------------------------------
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 30
DEFAULT_BATCH_SIZE: int = 64
TOP_K_TO_KEEP_PER_LOSS: int = 3



try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    pass

# ==============================================================
# ヘルパー関数群（データパス取得、データローダ構築、α保存、メモリ解放）
# ==============================================================
def get_data_paths(datatype: str, flow_preprocessing: str) -> Tuple[str, str]:
    """データセット種類と前処理種別から VideoMAE JSON と X3D特徴パスを返す。"""
    vmae_json_path = f"./vector/{datatype}/train/vectors_sliding_base.json"
    x3d_dir_path = f"./x3d_output/{datatype}/train"
    x3d_centered_dir_path = f"./x3d_output_centered/{datatype}/train"
    current_x3d_path = x3d_centered_dir_path if flow_preprocessing == "centered" else x3d_dir_path
    return vmae_json_path, current_x3d_path


def build_datasets_and_loaders(
    config: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    le_act: LabelEncoder,
    le_sp: LabelEncoder,
) -> Tuple[DataLoader, DataLoader, Optional[nn.Module]]:
    """train/val の DataLoader と gated モード用融合モデルを構築する。"""
    vmae_json_path, current_x3d_path = get_data_paths(config["datatype"], config.get("flow_preprocessing", "normal"))

    fusion_model: Optional[nn.Module] = None
    if config["train_mode"] == "mae":
        train_dataset = MAEDataset(train_df, vmae_json_path, le_act, le_sp)
        val_dataset = MAEDataset(val_df, vmae_json_path, le_act, le_sp)
    elif config["train_mode"] == "flow":
        train_dataset = FlowNpyDataset(train_df, current_x3d_path, le_act, le_sp)
        val_dataset = FlowNpyDataset(val_df, current_x3d_path, le_act, le_sp)
    elif config["train_mode"] == "gated":
        train_dataset = X3DVideoMAEDataset(train_df, current_x3d_path, vmae_json_path, le_act, le_sp)
        val_dataset = X3DVideoMAEDataset(val_df, current_x3d_path, vmae_json_path, le_act, le_sp)
        fusion_model = GatedFusion(2048, 768, config["fused_dim"]).to(DEVICE)
    else:
        raise ValueError(f"Unknown train_mode: {config['train_mode']}")

    workers = int(os.getenv("DATALOADER_NUM_WORKERS", "0"))
    pin_env = os.getenv("PIN_MEMORY", "auto").lower()
    pin_mem = torch.cuda.is_available() if pin_env == "auto" else (pin_env in ["1", "true", "yes"]) 

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader, fusion_model


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

def get_loss_fn_and_miner(
    loss_type: str,
    temperature: float = 0.07,
    triplet_margin: float = 0.1,
) -> Tuple[nn.Module, Optional[nn.Module]]:
    """loss_type に応じて損失関数と Triplet マイナーを返す。

    supcon: SupConLoss (温度パラメータあり)
    cosine: CosineSimilarity距離ベースの Triplet + ハードマイニング
    その他: L2距離ベースの Triplet + ハードマイニング
    """
    miner: Optional[nn.Module] = None

    if loss_type == "supcon":
        loss_fn = losses.SupConLoss(temperature=temperature)

    elif loss_type == "cosine":
        distance = distances.CosineSimilarity()
        loss_fn = losses.TripletMarginLoss(distance=distance, margin=triplet_margin)
        miner = miners.TripletMarginMiner(
            margin=triplet_margin, distance=distance, type_of_triplets="hard"
        )

    else:
        loss_fn = losses.TripletMarginLoss(margin=triplet_margin)
        miner = miners.TripletMarginMiner(
            margin=triplet_margin, type_of_triplets="hard"
        )

    return loss_fn, miner


def _build_inference_models(config: Dict[str, Any], D: int, fusion: Optional[nn.Module] = None) -> nn.ModuleDict:
    """推論専用のモデル辞書を構築"""
    models = nn.ModuleDict()
    if config.get('use_adversarial'):
        models['action_encoder'] = ActionMLPNet(D, 256, 256).to(DEVICE)
        if fusion is not None:
            models['fusion'] = fusion.to(DEVICE)
    else:
        models['net'] = SimpleMLPNet(D, 256, 256).to(DEVICE)
        if fusion is not None:
            models['fusion'] = fusion.to(DEVICE)
    return models


def _encode_batch(models: nn.ModuleDict, batch: Tuple[torch.Tensor, ...], config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """入力バッチを適切なエンコーダで特徴量に変換する。

    gated モード: (x3d特徴, vmae特徴, action, species)
    それ以外: (feature, action, species)
    戻り値: (埋め込み, actionラベル, speciesラベル, α(ゲート係数またはNone))
    """
    if config['train_mode'] == 'gated':
        x3d, vmae, a, s = batch
        x3d = x3d.to(DEVICE)
        vmae = vmae.to(DEVICE)
        a = a.to(DEVICE, dtype=torch.long)
        s = s.to(DEVICE, dtype=torch.long)
        fused, alpha = models['fusion'](x3d, vmae)
        x = fused
    else:
        x, a, s = batch
        x = x.to(DEVICE)
        a = a.to(DEVICE, dtype=torch.long)
        s = s.to(DEVICE, dtype=torch.long)
        alpha = None

    encoder = models['action_encoder'] if 'action_encoder' in models else models['net']
    a_vec = encoder(x)
    return a_vec, a, s, alpha

def train_step(
    models: nn.ModuleDict,
    batch: Tuple[torch.Tensor, ...],
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    config: Dict[str, Any],
    le_sp: LabelEncoder,
) -> Tuple[float, Optional[np.ndarray]]:
    """1バッチ分の学習ステップを実行する。
        adversarial が有効な場合は判別器を先に更新し、その後でエンコーダ（+融合層）を
        主タスク損失と（必要なら）逆学習のKL損失で更新する。
    戻り値:
        total_loss値、gated時のα (numpy) あるいは None
    """
    for model in models.values():
        model.train()

    a_vec, a, s, alpha = _encode_batch(models, batch, config)

    if config.get('use_adversarial', False):
        discriminator: nn.Module = models['discriminator']
        optimizer_disc = optimizers['discriminator']

        # 複数回更新（例: 3回）
        for _ in range(config.get('disc_steps_per_batch', 3)):
            optimizer_disc.zero_grad()
            logits_disc = discriminator(a_vec.detach())
            ce_loss = nn.CrossEntropyLoss()(logits_disc, s)
            ce_loss.backward()
            optimizer_disc.step()

    main_optimizer = optimizers.get('encoder') or optimizers.get('main')
    if main_optimizer is None:
        raise RuntimeError("No optimizer found for back-prop")
    main_optimizer.zero_grad()

    if config['loss_type'] == 'supcon':
        main_loss = loss_fn(a_vec, labels=a)
    elif miner is not None:
        hard_triplets = miner(a_vec, a)
        main_loss = loss_fn(a_vec, a, hard_triplets)
    else:
        main_loss = loss_fn(a_vec, a)

    total_loss = main_loss

    if config.get('use_adversarial', False):
        logits = models['discriminator'](a_vec)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        uniform_target = torch.full_like(logits, 1.0 / len(le_sp.classes_))
        adv_loss = nn.KLDivLoss(reduction='batchmean')(log_probs, uniform_target)
        total_loss = total_loss + config['lambda_adv'] * adv_loss

    total_loss.backward()
    # Gradient clipping for stability
    encoder = models['action_encoder'] if 'action_encoder' in models else models['net']
    nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5.0)
    if 'fusion' in models:
        nn.utils.clip_grad_norm_(models['fusion'].parameters(), max_norm=5.0)
    main_optimizer.step()

    # Return loss and optional alpha (as numpy) for logging
    alpha_np: Optional[np.ndarray] = None
    if alpha is not None:
        alpha_np = alpha.detach().cpu().numpy()
    return float(total_loss.item()), alpha_np

def evaluate_model(
    models: nn.ModuleDict,
    loader: DataLoader,
    config: Dict[str, Any],
    loss_fn: nn.Module,
    miner: Optional[nn.Module],
    fusion: Optional[nn.Module] = None,
) -> float:
    """勾配計算を行わずにローダ全体の平均損失を評価する。"""
    for model in models.values():
        model.eval()
    eval_losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            a_vec, a, _s, _alpha = _encode_batch(models, batch, config)
            if config['loss_type'] == 'supcon':
                loss = loss_fn(a_vec, labels=a)
            elif miner is not None:
                hard_triplets = miner(a_vec, a)
                loss = loss_fn(a_vec, a, hard_triplets)
            else:
                loss = loss_fn(a_vec, a)
            eval_losses.append(float(loss.item()))
    return float(np.mean(eval_losses)) if eval_losses else float('inf')

def train_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    le_sp: LabelEncoder,
    trial: optuna.trial.Trial,
    study_name: str,
    fusion: Optional[nn.Module] = None,
    results_root: Optional[Path] = None,
    run_name_override: Optional[str] = None,
    is_ablation: bool = False,
    ablation_subdir: Optional[str] = None,
) -> float:
    """最大 MAX_EPOCHS まで学習し、早期終了を用いて最良モデルを保存する。"""
    S = len(le_sp.classes_)
    sample_data = next(iter(train_loader))[0]
    D = config['fused_dim'] if fusion is not None else sample_data.shape[1]

    models = nn.ModuleDict()
    optimizers: Dict[str, torch.optim.Optimizer] = {}
    wd = float(config['weight_decay'])

    if config['use_adversarial']:
        models['action_encoder'] = (
            ActionMLPNet(D, 256, 256) if config['use_mlp'] else ActionLinearNet(D, 256)
        ).to(DEVICE)
        models['discriminator'] = SpeciesDiscriminator(256, S).to(DEVICE)
        params_enc = list(models['action_encoder'].parameters())
        if fusion is not None:
            models['fusion'] = fusion.to(DEVICE)
            params_enc.extend(models['fusion'].parameters())
        optimizers['encoder'] = torch.optim.Adam(params_enc, lr=float(config['lr']), weight_decay=wd)
        optimizers['discriminator'] = torch.optim.Adam(
            models['discriminator'].parameters(), lr=float(config['lr']), weight_decay=wd
        )
    else:
        models['net'] = (
            SimpleMLPNet(D, 256, 256) if config['use_mlp'] else SimpleLinearNet(D, 256)
        ).to(DEVICE)
        params_to_optimize = list(models['net'].parameters())
        if fusion is not None:
            models['fusion'] = fusion.to(DEVICE)
            params_to_optimize.extend(models['fusion'].parameters())
        optimizers['main'] = torch.optim.Adam(
            params_to_optimize, lr=float(config['lr']), weight_decay=wd
        )

    #損失関数とマイナーの構築
    loss_fn, miner = get_loss_fn_and_miner(
        config["loss_type"],
        temperature=config.get("temperature", 0.07),
        triplet_margin=config.get("triplet_margin", 0.1),   
    )

    run_name = run_name_override or (wandb.run.name if wandb.run else None) or "local-run"
    # Save locations entirely under results_root (train_result/<date>)
    if results_root is not None:
        if is_ablation:
            # ablation models live under results_root/ablations[/<category>]
            model_dir = Path(results_root) / "ablations"
            if ablation_subdir:
                model_dir = model_dir / ablation_subdir
        else:
            # trial checkpoints live under results_root/checkpoints/<study_name>
            model_dir = Path(results_root) / "checkpoints" / study_name
    else:
        # Fallback: local date folder
        date_dir = datetime.now().strftime("%Y-%m-%d")
        if run_name_override and run_name_override.startswith("ablation_"):
            model_dir = Path("./train_result") / date_dir / "ablations"
        else:
            model_dir = Path("./train_result") / date_dir / "checkpoints" / study_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{run_name}_best.pth"

    best_val_loss: float = float('inf')
    best_epoch: int = -1
    no_improve: int = 0

    for epoch in range(MAX_EPOCHS):
        train_losses: List[float] = []
        alpha_epoch_parts: List[np.ndarray] = []
        desc = f"[{config['train_mode'].upper()}][{config['loss_type']}] Epoch {epoch+1:03d}"
        for batch in tqdm(train_loader, desc=desc):
            loss, alpha_np = train_step(models, batch, loss_fn, miner, optimizers, config, le_sp)
            if loss is not None:
                train_losses.append(loss)
            # Collect alpha per batch if in gated mode
            if alpha_np is not None:
                # ensure 2D (N, D)
                alpha_np = alpha_np if alpha_np.ndim > 1 else alpha_np.reshape(-1, 1)
                alpha_epoch_parts.append(alpha_np)
        avg_train_loss = float(np.mean(train_losses)) if train_losses else float('inf')

        fusion_for_eval = models['fusion'] if 'fusion' in models else None
        avg_val_loss = evaluate_model(models, val_loader, config, loss_fn, miner, fusion=fusion_for_eval)

        # --- αの保存（毎エポック、非アブレーション時のみ、一時フォルダに蓄積） ---
        save_alpha_epoch(alpha_epoch_parts, epoch + 1, config, trial, results_root, is_ablation)

        if wandb.run is not None:
            wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            no_improve = 0
            print(f"✅ Best val_loss improved to {best_val_loss:.4f}. Saving model...")
            torch.save(models.state_dict(), save_path)
            trial.set_user_attr("model_save_path", str(save_path))
            trial.set_user_attr("best_epoch", int(best_epoch))
            # no-op for alpha: per-epoch saving is handled above
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        trial.report(avg_val_loss, epoch)

        # Proactive cleanup per epoch to reduce peak memory
        try:
            alpha_epoch_parts.clear()
        except Exception:
            pass
        cleanup_memory()

    trial.set_user_attr("epochs_run", int(epoch + 1))

    return float(best_val_loss)

# --- Optuna 目的関数 ---
def objective(trial: optuna.trial.Trial, full_df: pd.DataFrame, le_act: LabelEncoder, le_sp: LabelEncoder, results_root: Optional[Path] = None):
    # Explore loss_type within a single study
    # loss_type = trial.suggest_categorical("loss_type", ["improved", "cosine", "default", "supcon"])
    loss_type = "improved"
    trial.set_user_attr("loss_type", loss_type)
    train_mode = trial.suggest_categorical("train_mode", ["flow", "mae", "gated"])
    
    flow_preprocessing = 'n/a'
    if train_mode in ['flow', 'gated']:
        flow_preprocessing = trial.suggest_categorical("flow_preprocessing", ["normal", "centered"])

    config: Dict[str, Any] = {
        "use_mlp": trial.suggest_categorical("use_mlp", [True]),
        "loss_type": loss_type, 
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "lambda_adv": trial.suggest_float("lambda_adv", 0.01, 0.5),
        "train_mode": train_mode,
        "use_adversarial": trial.suggest_categorical("use_adversarial", [True]),
        "flow_preprocessing": flow_preprocessing,
        "datatype": 'animalkingdom', "batch_size": DEFAULT_BATCH_SIZE, "fused_dim": 512, "feature_dim": 256
    }
    # Triplet系のマージン（ユーザー要望の"alpha"に相当）を探索対象に追加
    # SupConのときは使われませんが、パラメータとしては保持します
    config["triplet_margin"] = trial.suggest_float("triplet_margin", 0.05, 0.5)
    
    if loss_type == 'supcon':
        config['temperature'] = trial.suggest_float('temperature', 0.05, 0.5)

    run_name_base = f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}_{'mlp' if config['use_mlp'] else 'nomlp'}_{'adv' if config['use_adversarial'] else 'noadv'}"

    if config['train_mode'] in ['flow', 'gated']:
        run_name = f"{run_name_base}_{config['flow_preprocessing']}"
    else:
        run_name = run_name_base

    # Persist key config into trial.user_attrs for downstream selection/naming
    trial.set_user_attr("train_mode", train_mode)
    trial.set_user_attr("flow_preprocessing", flow_preprocessing)
    trial.set_user_attr("use_mlp", bool(config.get('use_mlp')))
    trial.set_user_attr("use_adversarial", bool(config.get('use_adversarial')))
    trial.set_user_attr("triplet_margin", float(config.get("triplet_margin", 0.0)))

    wandb.init(project="optuna", config=config, group=config['train_mode'], name=run_name, reinit=True)
    # Optional per-trial notifications (set DISCORD_NOTIFY_TRIALS=1 to enable)
    if os.getenv("DISCORD_NOTIFY_TRIALS") == "1":
        discord_notify(
            content=(
                f"▶️ Trial #{trial.number} start\n"
                f"loss_type: {loss_type}, mode: {train_mode}, mlp: {config['use_mlp']}, adv: {config['use_adversarial']}\n"
                f"run: {run_name}"
            )
        )
    
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['action'])

    # データローダの構築（重複ロジックを関数化）
    train_loader, val_loader, fusion_model = build_datasets_and_loaders(
        config, train_df, val_df, le_act, le_sp
    )

    study_name = trial.study.study_name
    # Save study_name into trial for later summaries (FrozenTrial lacks .study)
    trial.set_user_attr("study_name", study_name)

    try:
        # 学習本体（早期終了/ベスト保存を内部で実施）
        best_val_loss = train_model(
            config, train_loader, val_loader, le_sp, trial, study_name,
            fusion=fusion_model, results_root=results_root
        )
        # --- Compute clustering metrics (ARI/NMI) on validation embeddings ---
        model_path = trial.user_attrs.get("model_save_path")
        ari_val, nmi_val, combined_val = -1.0, -1.0, -1.0
        if model_path:
            try:
                # Determine encoder input dimension D from val loader (or config)
                try:
                    first_batch = next(iter(val_loader))
                    if config['train_mode'] == 'gated':
                        D_enc = int(config['fused_dim'])
                    else:
                        D_enc = int(first_batch[0].shape[1])
                except Exception:
                    D_enc = int(config.get('fused_dim' if config['train_mode'] == 'gated' else 'feature_dim', 256))
                # Build inference models and load weights
                inf_models = _build_inference_models(config, D_enc, fusion=fusion_model if config['train_mode'] == 'gated' else None)
                state = torch.load(str(model_path), map_location=DEVICE)
                try:
                    inf_models.load_state_dict(state)
                except Exception:
                    inf_models.load_state_dict(state, strict=False)
                ari_val, nmi_val, combined_val = _compute_clustering_metrics(inf_models, val_loader, config)
            except Exception as e:
                print(f"⚠️ Failed to compute clustering metrics for trial #{trial.number}: {e}")
        trial.set_user_attr("metric_ari", float(ari_val))
        trial.set_user_attr("metric_nmi", float(nmi_val))
        trial.set_user_attr("metric_combined", float(combined_val))
    finally:
        # Explicit cleanup to release memory between trials
        try:
            del train_loader, val_loader
        except Exception:
            pass
        try:
            del train_dataset, val_dataset, fusion_model
        except Exception:
            pass
        cleanup_memory()

    wandb.finish()
    # Optional per-trial completion notification
    if os.getenv("DISCORD_NOTIFY_TRIALS") == "1":
        model_path = trial.user_attrs.get("model_save_path", "(no-save)")
        best_epoch = trial.user_attrs.get("best_epoch", "-")
        discord_notify(
            content=(
                f"✅ Trial #{trial.number} done: combined={trial.user_attrs.get('metric_combined', -1):.5f} (ARI={trial.user_attrs.get('metric_ari', -1):.4f}, NMI={trial.user_attrs.get('metric_nmi', -1):.4f})\n"
                f"best_epoch: {best_epoch}, model: {model_path}"
            )
        )
    # Return combined metric for Optuna (maximize). Fall back to -1 if unavailable.
    return float(trial.user_attrs.get("metric_combined", -1.0))

# --- Fixed-config training utilities (no Optuna suggestions) ---
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


# --- Helpers for ablation naming and layout ---
def build_ablation_basename(cfg: Dict[str, Any], category: str) -> str:
    """Return a human-friendly base filename per category.

    Rules from examples:
    - baseline, losstype: include loss + mode + mlp|nomlp + adv|noadv + [preproc if flow/gated]
    - toggle_* and flow_preprocessing_alt: include mode + mlp|nomlp + adv|noadv + [preproc]
    - mode: include mode + mlp|nomlp + adv (no preproc)
    """
    include_loss = category in (
        "baseline",
        "losstype",
        "toggle_use_mlp",
        "toggle_use_adversarial",
        "flow_preprocessing_alt",
    )
    include_preproc = category in (
        "baseline",
        "toggle_use_mlp",
        "toggle_use_adversarial",
        "flow_preprocessing_alt",
        "losstype",
    )

    parts: List[str] = []
    if include_loss:
        parts.append(str(cfg.get("loss_type", "unknown")))
    # mode
    parts.append(str(cfg.get("train_mode", "unknown")))
    # mlp/nomlp
    parts.append("mlp_on" if cfg.get("use_mlp") else "mlp_off")
    # adv/noadv
    parts.append("adv_on" if cfg.get("use_adversarial") else "adv_off")
    # flow preprocessing token when applicable and requested
    if include_preproc and cfg.get("train_mode") in ["flow", "gated"]:
        fp = cfg.get("flow_preprocessing", "normal")
        parts.append(str(fp))
    return "_".join(parts)


def train_with_config(
    config: Dict[str, Any],
    full_df: pd.DataFrame,
    le_act: LabelEncoder,
    le_sp: LabelEncoder,
    results_root: Optional[Path],
    study_name: str = "ablation",
    trial_number: int = -1,
    category: Optional[str] = None,
    ) -> Tuple[float, Optional[str]]:
    """与えられた設定で1度だけ学習を回し、検証損失と保存モデルパスを返す。"""
    # Split data
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['action'])

    train_loader, val_loader, fusion_model = build_datasets_and_loaders(
        config, train_df, val_df, le_act, le_sp
    )

    dummy_trial = DummyTrial(number=trial_number)
    # Human-readable ablation filename base and category subdir
    cat = category or "misc"
    run_name = build_ablation_basename(config, cat)
    try:
        best_val = train_model(
            config,
            train_loader,
            val_loader,
            le_sp,
            dummy_trial,
            study_name,
            fusion=fusion_model,
            results_root=results_root,
            run_name_override=run_name,
            is_ablation=True,
            ablation_subdir=cat,
        )
    finally:
        try:
            del train_loader, val_loader
        except Exception:
            pass
        try:
            del train_dataset, val_dataset, fusion_model
        except Exception:
            pass
        cleanup_memory()
    return best_val, dummy_trial.user_attrs.get("model_save_path")

# --- メイン実行ブロック ---
def main() -> None:
    """エントリポイント。

    - 環境設定の読み込み
    - Optunaスタディの作成と最適化
    - ベスト試行の要約とチェックポイント整理
    - （任意）アブレーションの実行
    """
    # Load .env (optional)
    # .env is already loaded via python-dotenv at import time
    # Storage selection: use in-memory if OPTUNA_INMEMORY=1 or OPTUNA_STORAGE is empty/memory
    env_storage = os.getenv("OPTUNA_STORAGE", "").strip()
    use_inmemory = os.getenv("OPTUNA_INMEMORY", "0") == "1" or env_storage in ("", "memory")
    storage_obj = InMemoryStorage() if use_inmemory and InMemoryStorage is not None else None
    storage_name = None if use_inmemory else env_storage if env_storage else None
    # 任意: 再現性をある程度担保
    seed = int(os.getenv("SEED", "42"))
    set_seed(seed)

    # --- Discord: training start notification ---
    try:
        n_trials = int(os.getenv("N_TRIALS_PER_STUDY", "30"))
    except Exception:
        n_trials = 30
    bs_override = os.getenv("BATCH_SIZE_OVERRIDE", "-")
    dl_workers = os.getenv("DATALOADER_NUM_WORKERS", "0")
    pin_mem = os.getenv("PIN_MEMORY", "auto")
    max_epochs_env = os.getenv("MAX_EPOCHS", "auto")
    matmul_prec = os.getenv("MATMUL_PRECISION", "default")
    wandb_disabled = os.getenv("WANDB_DISABLED", "0")
    msg = (
        "🚀 Training started\n"
        f"Storage: {storage_name}\n"
        f"Seed: {seed}\n"
        f"Trials per study: {n_trials}\n"
        f"BatchSizeOverride: {bs_override}, Workers: {dl_workers}, PinMemory: {pin_mem}\n"
        f"MaxEpochs: {max_epochs_env}, MatmulPrecision: {matmul_prec}, WANDB_DISABLED: {wandb_disabled}\n"
        "Loss types: improved | cosine | default | supcon"
    )
    
    # Discord通知（例外は握りつぶして進行を妨げない）
    try:
        discord_notify(content=msg)
    except Exception:
        pass

    print("Loading initial data...")
    datatype = 'animalkingdom'
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df['action'])
    le_sp = LabelEncoder().fit(full_df['species'])
    print("Data loaded.")

    N_TRIALS_PER_STUDY = int(os.getenv("N_TRIALS_PER_STUDY", "30"))

    # Results root per date (YYYY-MM-DD) and run index (run_001, run_002, ...)
    date_dir = datetime.now().strftime("%Y-%m-%d")
    date_root = Path("./train_result") / date_dir
    date_root.mkdir(parents=True, exist_ok=True)
    # Determine next run id within the date directory
    existing_runs = [p for p in date_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    def _run_index(name: str) -> int:
        try:
            return int(name.split("_")[-1])
        except Exception:
            return 0
    next_idx = 1
    if existing_runs:
        next_idx = max(_run_index(p.name) for p in existing_runs) + 1
    run_dir = date_root / f"run_{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_root = run_dir

    # Create a single mixed study exploring loss_type as a hyperparameter
    today = datetime.now().strftime("%m%d")
    study_name = f"disentangle-study-mixed-{today}"
    print(f"\n\n===== Starting Optuna Study (mixed loss types) =====")
    # Notify study start
    discord_notify(f"🧪 Start study '{study_name}' (loss_type in [improved, cosine, default, supcon])")

    study = optuna.create_study(
        direction="maximize",
        storage=(storage_obj if storage_obj is not None else storage_name),
        study_name=study_name,
        load_if_exists=not use_inmemory
    )

    study.optimize(lambda trial: objective(trial, full_df, le_act, le_sp, results_root), n_trials=N_TRIALS_PER_STUDY, gc_after_trial=True)

    # Extra cleanup after the study finishes
    cleanup_memory()

    print(f"\n--- Best Trial (mixed) ---")
    print(f"Value (combined ARI/NMI): {study.best_value}")
    print(f"Params: {study.best_trial.params}")
    # Notify study completion
    discord_notify(
        content=(
            f"🏁 Study '{study_name}' finished\n"
            f"Best combined metric (ARI/NMI): {study.best_value:.5f} (trial #{study.best_trial.number})\n"
            f"Params: {study.best_trial.params}"
        )
    )


    print("\n--- Starting model selection/cleanup ---")
    # Collect trials for this run only when using in-memory storage; otherwise, aggregate from storage
    if use_inmemory:
        all_trials_info: List[Tuple[optuna.trial.FrozenTrial, str]] = [
            (t, study_name) for t in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        ]
        all_studies = [type("S", (), {"study_name": study_name})]  # minimal shim for loop reuse
    else:
        all_studies = optuna.get_all_study_summaries(storage=storage_name)
        all_trials_info = []
        for summary in all_studies:
            st = optuna.load_study(study_name=summary.study_name, storage=storage_name)
            trials = st.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            all_trials_info.extend((t, summary.study_name) for t in trials)
    
    # -------------------------------
    # 上位N件（全lossタイプ横断）のモデルを保持
    # -------------------------------
    TOP_K = TOP_K_TO_KEEP_PER_LOSS
    best_trials_overall: List[Tuple[optuna.trial.FrozenTrial, str]] = sorted(
        all_trials_info, key=lambda info: info[0].value, reverse=True
    )[:TOP_K]

    # -------------------------------
    # 上位100件の試行をファイル出力
    # -------------------------------
    all_trials_sorted = sorted(all_trials_info, key=lambda info: info[0].value, reverse=True)[:100]
    summary_path = results_root / "results_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        title = "This Run Only" if use_inmemory else "Top 100 by metric (combined ARI/NMI)"
        f.write(f"=== Optuna Trial Summary ({title}) ===\n\n")
        for rank, (trial, study_name_) in enumerate(all_trials_sorted, 1):
            f.write(f"[{rank:03d}] loss_type={trial.user_attrs.get('loss_type')} | metric={trial.value:.6f}\n")
            # Prefer user_attr study_name if present, else use collected name
            sn = trial.user_attrs.get("study_name", study_name_)
            f.write(f"    study_name : {sn}\n")
            f.write(f"    trial_number: {trial.number}\n")
            f.write(f"    params      : {trial.params}\n")
            if "model_save_path" in trial.user_attrs:
                f.write(f"    model_path  : {trial.user_attrs['model_save_path']}\n")
            f.write("\n")
    print(f"✅ Saved summary of top 100 trials to {summary_path}")

    # -------------------------------
    # 保存対象モデルパスの抽出
    # -------------------------------
    # パス比較の不一致（"./models" vs "models"）や相対/絶対の差を避けるため、事前にresolveして揃える
    paths_to_keep: Set[Path] = set()
    for t, _study_name in best_trials_overall:
        path_str = t.user_attrs.get("model_save_path")
        if not path_str:
            continue
        try:
            paths_to_keep.add(Path(path_str).resolve())
        except Exception as e:
            print(f"⚠️ Skipping keep-path '{path_str}': {e}")

    deleted_count = 0

    # Copy kept best models into train_result/<date>/best_model with global rank
    best_model_dir = results_root / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)
    for rank, (t, _sn) in enumerate(best_trials_overall, start=1):
        src_path_str = t.user_attrs.get("model_save_path")
        if not src_path_str:
            continue
        try:
            src_path = Path(src_path_str)
            if not src_path.exists():
                continue
            # Build a readable basename from trial params akin to ablation naming
            params = t.params or {}
            cfg_name: Dict[str, Any] = {
                "loss_type": t.user_attrs.get("loss_type", params.get("loss_type", "unknown")),
                "train_mode": params.get("train_mode", "unknown"),
                "use_mlp": params.get("use_mlp", False),
                "use_adversarial": params.get("use_adversarial", False),
                # Prefer the trial's recorded user_attr when available
                "flow_preprocessing": t.user_attrs.get("flow_preprocessing") 
                           or params.get("flow_preprocessing", "normal"),
            }
            base = build_ablation_basename(cfg_name, category="baseline")
            dst_name = f"{base}_rank{rank}_best.pth"
            dst_path = best_model_dir / dst_name
            shutil.copy2(str(src_path), str(dst_path))
        except Exception as e:
            print(f"⚠️ Failed to copy best model for trial #{t.number}: {e}")

    # Delete non-kept checkpoints under results_root/checkpoints/<study>
    checkpoints_root = results_root / "checkpoints"
    # Limit cleanup scope to this study in in-memory mode
    study_names_to_scan = [study_name] if use_inmemory else [s.study_name for s in all_studies]
    for sn in study_names_to_scan:
        models_dir = checkpoints_root / sn
        if models_dir.exists():
            for model_path in models_dir.glob("**/*.pth"):
                try:
                    if model_path.resolve() not in paths_to_keep:
                        model_path.unlink()
                        deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to process '{model_path}': {e}")
    
    print(f"Cleanup finished. Kept {len(paths_to_keep)} best models overall.")
    print(f"Deleted {deleted_count} other model checkpoints.")

    # Consolidate alpha logs: keep all epochs only for global rank-1 trial
    if best_trials_overall:
        top_trial, _sn = best_trials_overall[0]
        alpha_tmp_root = results_root / 'alpha_logs_tmp'
        alpha_out_dir = results_root / 'alpha_logs'
        alpha_out_dir.mkdir(parents=True, exist_ok=True)
        # Move/copy all temp alphas for the top trial into alpha_logs, remove others
        try:
            if alpha_tmp_root.exists():
                for trial_dir in alpha_tmp_root.glob('trial_*'):
                    trial_num_str = trial_dir.name.split('_')[-1]
                    if trial_num_str.isdigit() and int(trial_num_str) == top_trial.number:
                        # move all files to alpha_logs with same names
                        for f in trial_dir.glob('*.npy'):
                            dst = alpha_out_dir / f.name
                            try:
                                shutil.copy2(str(f), str(dst))
                            except Exception as e:
                                print(f"⚠️ Failed to copy alpha '{f}' -> '{dst}': {e}")
                    # remove the temp directory regardless (we will keep only copied ones)
                    try:
                        for f in trial_dir.glob('*.npy'):
                            try:
                                f.unlink()
                            except Exception:
                                pass
                        trial_dir.rmdir()
                    except Exception:
                        pass
                # remove alpha_logs_tmp if empty
                try:
                    alpha_tmp_root.rmdir()
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Alpha consolidation failed: {e}")

    # Optional ablation runs based on the single best overall trial
    if os.getenv("RUN_ABLATIONS") == "1":
        print("\n--- Running ablation suite (RUN_ABLATIONS=1) ---")
        if not all_trials_info:
            print("No trials to base ablations on.")
        else:
            # global best across all loss types (maximize combined metric)
            best_overall_trial, best_overall_study = max(all_trials_info, key=lambda info: info[0].value)
            # Derive baseline loss_type robustly
            baseline_loss_type = best_overall_trial.user_attrs.get("loss_type")
            if not baseline_loss_type:
                # fallback: parse from study name (format: disentangle-study-<loss_type>-<mmdd>)
                st_name = best_overall_trial.user_attrs.get("study_name", best_overall_study)
                if isinstance(st_name, str) and st_name.startswith("disentangle-study-"):
                    parts = st_name.split("-")
                    if len(parts) >= 4:
                        baseline_loss_type = parts[2]
            if not baseline_loss_type:
                print("⚠️ baseline loss_type not found; falling back to 'default'")
                baseline_loss_type = "default"
            params = best_overall_trial.params
            # Reconstruct baseline config
            baseline_config: Dict[str, Any] = {
                "use_mlp": params.get("use_mlp", False),
                "loss_type": baseline_loss_type,
                "lr": params.get("lr", 1e-4),
                "weight_decay": params.get("weight_decay", 1e-5),
                "lambda_adv": params.get("lambda_adv", 0.1),
                "train_mode": params.get("train_mode", "gated"),
                "use_adversarial": params.get("use_adversarial", False),
                # Prefer user_attr recorded during trial for robustness
                "flow_preprocessing": best_overall_trial.user_attrs.get("flow_preprocessing", params.get("flow_preprocessing", "normal")),
                "datatype": datatype,
                "batch_size": DEFAULT_BATCH_SIZE,
                "fused_dim": 512,
                "feature_dim": 256,
            }
            if baseline_loss_type == 'supcon':
                baseline_config['temperature'] = params.get('temperature', 0.07)

            # Baseline
            baseline_val, baseline_model = train_with_config(
                dict(baseline_config),
                full_df,
                le_act,
                le_sp,
                results_root,
                study_name="ablation",
                trial_number=0,
                category="baseline",
            )
            print(f"Baseline ({baseline_loss_type}) val_loss={baseline_val:.6f} model={baseline_model}")

            # Define ablations
            ablations: List[Tuple[str, str, Dict[str, Any]]] = []
            # MLP のアブレーションはスキップ（基本 ON 固定）

            # Adversarialの有無比較（ベースラインと同じ側は学習しない）
            if baseline_config['use_adversarial']:
                cfg_toggle = dict(baseline_config); cfg_toggle['use_adversarial'] = False
                ablations.append(("adv_off", "use_adversarial", cfg_toggle))
            else:
                cfg_toggle = dict(baseline_config); cfg_toggle['use_adversarial'] = True
                ablations.append(("adv_on", "use_adversarial", cfg_toggle))

            # Flow preprocessing alternative (only meaningful for flow/gated)
            if baseline_config['train_mode'] in ['flow', 'gated']:
                cfg = dict(baseline_config)
                cfg['flow_preprocessing'] = 'centered' if baseline_config['flow_preprocessing'] != 'centered' else 'normal'
                ablations.append(("flow_preprocessing_alt", "flow_preprocessing_alt", cfg))
            # Mode variants (include when different from baseline to avoid duplicates)
            if baseline_config['train_mode'] != 'mae':
                cfg = dict(baseline_config); cfg['train_mode'] = 'mae'; ablations.append(("mode_mae", "mode", cfg))
            if baseline_config['train_mode'] != 'flow':
                cfg = dict(baseline_config); cfg['train_mode'] = 'flow'; ablations.append(("mode_flow", "mode", cfg))
            # Loss type variants (try all other loss types)
            # all_losses = ["improved", "cosine", "default", "supcon"]
            # for lt in all_losses:
            #     if lt == baseline_loss_type:
            #         continue
            #     cfg = dict(baseline_config)
            #     cfg['loss_type'] = lt
            #     if lt == 'supcon':
            #         # carry temperature if present, else default
            #         cfg['temperature'] = params.get('temperature', cfg.get('temperature', 0.07))
            #     else:
            #         # remove temperature if set from baseline
            #         if 'temperature' in cfg:
            #             cfg.pop('temperature', None)
            #     ablations.append((f"losstype_{lt}", "losstype", cfg))

            # Run ablations
            ablation_dir = results_root / "ablations"
            ablation_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            rows.append({
                "name": "baseline",
                "changes": json.dumps({}),
                "val_loss": baseline_val,
                "delta_vs_baseline": 0.0,
                "model_path": baseline_model or "",
            })
            for i, (name, category, cfg) in enumerate(ablations, start=1):
                val, model_path = train_with_config(
                    cfg,
                    full_df,
                    le_act,
                    le_sp,
                    results_root,
                    study_name="ablation",
                    trial_number=i,
                    category=category,
                )
                delta = float(val - baseline_val)
                diff = {k: cfg[k] for k in cfg if cfg[k] != baseline_config.get(k)}
                rows.append({
                    "name": name,
                    "changes": json.dumps(diff),
                    "val_loss": val,
                    "delta_vs_baseline": delta,
                    "model_path": model_path or "",
                })
            csv_path = ablation_dir / "ablation_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=["name", "changes", "val_loss", "delta_vs_baseline", "model_path"])
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"✅ Ablation results saved to {csv_path}")


if __name__ == "__main__":
    main()
