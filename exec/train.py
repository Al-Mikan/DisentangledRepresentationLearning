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
from optuna.pruners import HyperbandPruner
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# 必要なファイルをインポート
from utils import FlowNpyDataset, X3DVideoMAEDataset, MAEDataset
from model import (
    SimpleLinearNet, SimpleMLPNet, ActionLinearNet, ActionMLPNet,
    SpeciesDiscriminator, GatedFusion
)
from pytorch_metric_learning import losses, miners, distances


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Global settings and small utils
# -------------------------------
MAX_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 30
DEFAULT_BATCH_SIZE: int = 64
TOP_K_TO_KEEP_PER_LOSS: int = 3

# PyTorchの高速化設定（サポートGPUのみ）
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    # 古いPyTorchでは未対応
    pass

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility where practical."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Core training utilities (loss/miner, one-step train/eval, main training loop) ---
# 損失関数とマイナーの組み立て
LOSS_FN_MAP = {
    "improved": losses.TripletMarginLoss,
    "cosine": losses.TripletMarginLoss,
    "default": losses.TripletMarginLoss,
    "supcon": losses.SupConLoss
}

def get_loss_fn_and_miner(loss_type: str, temperature: float = 0.07) -> Tuple[nn.Module, Optional[nn.Module]]:
    """Return a metric-learning loss and optional miner based on loss_type.

    - supcon: uses SupConLoss with temperature
    - cosine: Triplet with cosine distance + hard triplet miner
    - improved/default: Triplet (euclidean) + hard triplet miner
    """
    miner: Optional[nn.Module] = None
    if loss_type == "supcon":
        loss_fn: nn.Module = LOSS_FN_MAP[loss_type](temperature=temperature)
    elif loss_type == 'cosine':
        distance = distances.CosineSimilarity()
        loss_fn = LOSS_FN_MAP[loss_type](distance=distance, margin=0.1)
        miner = miners.TripletMarginMiner(margin=0.1, distance=distance, type_of_triplets="hard")
    else:
        loss_fn = LOSS_FN_MAP[loss_type](margin=0.1)
        miner = miners.TripletMarginMiner(margin=0.1, type_of_triplets="hard")
    return loss_fn, miner


def _encode_batch(models: nn.ModuleDict, batch: Tuple[torch.Tensor, ...], config: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Move batch to DEVICE, optionally fuse features, and return encoded vectors and labels.

    Returns (a_vec, action_labels, species_labels).
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
    """One training step over a single batch.

    - Updates discriminator first if adversarial, using detached embeddings.
    - Then updates encoder (and fusion) with main loss (+ adversarial KL if enabled).
    """
    for model in models.values():
        model.train()

    # Encode once and share across losses/updates
    a_vec, a, s, alpha = _encode_batch(models, batch, config)

    # Adversarial update: train species discriminator to classify species from features
    if config.get('use_adversarial', False):
        discriminator: nn.Module = models['discriminator']
        optimizer_disc = optimizers['discriminator']
        optimizer_disc.zero_grad()
        logits_disc = discriminator(a_vec.detach())
        ce_loss = nn.CrossEntropyLoss()(logits_disc, s)
        ce_loss.backward()
        optimizer_disc.step()

    # Main update: encoder (and fusion)
    main_optimizer = optimizers.get('encoder') or optimizers.get('main')
    if main_optimizer is None:
        raise RuntimeError("No optimizer found for back-prop")
    main_optimizer.zero_grad()

    # Main metric-learning loss
    if config['loss_type'] == 'supcon':
        main_loss = loss_fn(a_vec, labels=a)
    elif miner is not None:
        hard_triplets = miner(a_vec, a)
        main_loss = loss_fn(a_vec, a, hard_triplets)
    else:
        main_loss = loss_fn(a_vec, a)

    total_loss = main_loss

    # Adversarial regularization: make species prediction uniform
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
    """Evaluate average loss on a loader without gradients."""
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
) -> float:
    """Train for up to MAX_EPOCHS with early stopping, saving best state_dict per trial."""
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

    loss_fn, miner = get_loss_fn_and_miner(config['loss_type'], config.get('temperature', 0.07))

    run_name = wandb.run.name or "local-run"
    model_dir = Path(f"./models/{config['datatype']}/{study_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{run_name}_best.pth"

    best_val_loss: float = float('inf')
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

        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # --- αの保存（gatedモード時、エポックごとにまとめて保存） ---
        if config.get('train_mode') == 'gated' and alpha_epoch_parts:
            os.makedirs("./alpha_logs", exist_ok=True)
            alpha_epoch = np.concatenate(alpha_epoch_parts, axis=0)
            np.save(
                f"./alpha_logs/alpha_trial{trial.number:03d}_epoch{epoch+1:03d}_{np.random.randint(1e6)}.npy",
                alpha_epoch,
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            print(f"✅ Best val_loss improved to {best_val_loss:.4f}. Saving model...")
            torch.save(models.state_dict(), save_path)
            trial.set_user_attr("model_save_path", str(save_path))
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

        trial.report(avg_val_loss, epoch)
        # 収束が悪い試行を早期打ち切りしたい場合は以下を有効化
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    return float(best_val_loss)

# --- Optuna 目的関数 ---
def objective(trial: optuna.trial.Trial, full_df: pd.DataFrame, le_act: LabelEncoder, le_sp: LabelEncoder, loss_type: str):
    trial.set_user_attr("loss_type", loss_type)
    train_mode = trial.suggest_categorical("train_mode", ["flow", "mae", "gated"])
    
    flow_preprocessing = 'n/a'
    if train_mode in ['flow', 'gated']:
        flow_preprocessing = trial.suggest_categorical("flow_preprocessing", ["normal", "centered"])

    config: Dict[str, Any] = {
        "use_mlp": trial.suggest_categorical("use_mlp", [True, False]),
        "loss_type": loss_type, 
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "lambda_action": 1.0,
        "lambda_adv": trial.suggest_float("lambda_adv", 0.01, 0.5),
        "train_mode": train_mode,
        "use_adversarial": trial.suggest_categorical("use_adversarial", [True, False]),
        "flow_preprocessing": flow_preprocessing,
        "datatype": 'animalkingdom', "batch_size": DEFAULT_BATCH_SIZE, "fused_dim": 512, "feature_dim": 256
    }
    
    if loss_type == 'supcon':
        config['temperature'] = trial.suggest_float('temperature', 0.05, 0.5)

    run_name_base = f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}_{'mlp' if config['use_mlp'] else 'nomlp'}_{'adv' if config['use_adversarial'] else 'noadv'}"

    if config['train_mode'] in ['flow', 'gated']:
        run_name = f"{run_name_base}_{config['flow_preprocessing']}"
    else:
        run_name = run_name_base

    wandb.init(project="optuna_disentangle_supcon", config=config, group=config['train_mode'], name=run_name, reinit=True)
    
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['action'])
    
    datatype = config['datatype']
    vmae_json_path = f"./vector/{datatype}/train/vectors_sliding_base.json"
    x3d_dir_path = f"./x3d_output/{datatype}/train"
    x3d_centered_dir_path = f"./x3d_output_centered/{datatype}/train"
    
    current_x3d_path = x3d_centered_dir_path if config.get('flow_preprocessing') == 'centered' else x3d_dir_path
    
    fusion_model = None
    if config['train_mode'] == 'mae':
        train_dataset = MAEDataset(train_df, vmae_json_path, le_act, le_sp)
        val_dataset = MAEDataset(val_df, vmae_json_path, le_act, le_sp)
    elif config['train_mode'] == 'flow':
        train_dataset = FlowNpyDataset(train_df, current_x3d_path, le_act, le_sp)
        val_dataset = FlowNpyDataset(val_df, current_x3d_path, le_act, le_sp)
    elif config['train_mode'] == 'gated':
        train_dataset = X3DVideoMAEDataset(train_df, current_x3d_path, vmae_json_path, le_act, le_sp)
        val_dataset = X3DVideoMAEDataset(val_df, current_x3d_path, vmae_json_path, le_act, le_sp)
        fusion_model = GatedFusion(2048, 768, config['fused_dim']).to(DEVICE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    study_name = trial.study.study_name
    best_val_loss = train_model(config, train_loader, val_loader, le_sp, trial, study_name, fusion=fusion_model)
    
    wandb.finish()
    return best_val_loss

# --- メイン実行ブロック ---
def main() -> None:
    storage_name = "sqlite:///optuna_study.db"
    # 任意: 再現性をある程度担保
    set_seed(42)

    print("Loading initial data...")
    datatype = 'animalkingdom'
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df['action'])
    le_sp = LabelEncoder().fit(full_df['species'])
    print("Data loaded.")

    loss_types_to_run = ["improved", "cosine", "default", "supcon"]
    N_TRIALS_PER_STUDY = 30

    for loss_type in loss_types_to_run:
        today = datetime.now().strftime("%m%d")
        study_name = f"disentangle-study-{loss_type}-{today}"
        print(f"\n\n===== Starting Optuna Study for loss_type: {loss_type} =====")

        study = optuna.create_study(
            direction="minimize",
            storage=storage_name,
            study_name=study_name,
            load_if_exists=True
        )
        
        study.optimize(lambda trial: objective(trial, full_df, le_act, le_sp, loss_type), n_trials=N_TRIALS_PER_STUDY)
        
        print(f"\n--- Best Trial for {loss_type} ---")
        print(f"Value: {study.best_value}")
        print(f"Params: {study.best_trial.params}")


    print("\n--- Starting model cleanup for all studies ---")
    all_studies = optuna.get_all_study_summaries(storage=storage_name)
    all_trials = []
    for summary in all_studies:
        study = optuna.load_study(study_name=summary.study_name, storage=storage_name)
        all_trials.extend(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
    
    # -------------------------------
    # 上位N件のモデルを保持
    # -------------------------------
    TOP_K = TOP_K_TO_KEEP_PER_LOSS  # loss_typeごとに上位N件保存
    best_trials_by_loss: Dict[str, List[optuna.trial.FrozenTrial]] = {}
    for lt in ["improved", "cosine", "default", "supcon"]:
        relevant_trials = [t for t in all_trials if t.user_attrs.get("loss_type") == lt]
        if not relevant_trials:
            continue
        sorted_trials = sorted(relevant_trials, key=lambda t: t.value)
        best_trials_by_loss[lt] = sorted_trials[:TOP_K]

    # -------------------------------
    # 上位100件の試行をファイル出力
    # -------------------------------
    all_trials_sorted = sorted(all_trials, key=lambda t: t.value)[:100]
    summary_path = Path("./results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Optuna Trial Summary (Top 100 by val_loss) ===\n\n")
        for rank, trial in enumerate(all_trials_sorted, 1):
            f.write(f"[{rank:03d}] loss_type={trial.user_attrs.get('loss_type')} | val_loss={trial.value:.6f}\n")
            f.write(f"    study_name : {trial.study.study_name}\n")
            f.write(f"    trial_number: {trial.number}\n")
            f.write(f"    params      : {trial.params}\n")
            if "model_save_path" in trial.user_attrs:
                f.write(f"    model_path  : {trial.user_attrs['model_save_path']}\n")
            f.write("\n")
    print(f"✅ Saved summary of top 100 trials to {summary_path}")

    # -------------------------------
    # 保存対象モデルパスの抽出
    # -------------------------------
    paths_to_keep: Set[Path] = set()
    for trials in best_trials_by_loss.values():
        for t in trials:
            path_str = t.user_attrs.get("model_save_path")
            if path_str:
                paths_to_keep.add(Path(path_str))

    deleted_count = 0
    for study_summary in all_studies:
        models_dir = Path(f"./models/{datatype}/{study_summary.study_name}")
        if models_dir.exists():
            for model_path in models_dir.glob("**/*.pth"):
                if model_path not in paths_to_keep:
                    model_path.unlink()
                    deleted_count += 1
    
    print(f"Cleanup finished. Kept {len(paths_to_keep)} best models overall.")
    print(f"Deleted {deleted_count} other model checkpoints.")


if __name__ == "__main__":
    main()
