import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import wandb
import optuna
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 必要なファイルをインポート
from utils import FlowNpyDataset, X3DVideoMAEDataset, MAEDataset
from model import (
    SimpleLinearNet, SimpleMLPNet, ActionLinearNet, ActionMLPNet,
    SpeciesDiscriminator, GatedFusion
)
from triplet_losses import ImprovedTripletLoss, CosineTripletLoss
from pytorch_metric_learning import losses

# PyTorchの高速化設定
try:
    torch.set_float32_matmul_precision('high')
except AttributeError:
    print("Warning: torch.set_float32_matmul_precision is not available in this PyTorch version. Skipping.")


# --- 損失関数の定義 ---
LOSS_FN_MAP = {
    "improved": ImprovedTripletLoss,
    "cosine": CosineTripletLoss,
    "default": lambda: nn.TripletMarginLoss(0.1),
    "supcon": losses.SupConLoss
}

def get_loss_fn(loss_type, temperature=0.07):
    """損失関数のインスタンスを取得する"""
    if loss_type == "supcon":
        return LOSS_FN_MAP[loss_type](temperature=temperature)
    return LOSS_FN_MAP.get(loss_type, LOSS_FN_MAP["default"])()

def make_triplets_hard(vectors, labels):
    anchors, positives, negatives = [], [], []
    with torch.no_grad():
        dists = torch.cdist(vectors, vectors, p=2)
    for i in range(len(vectors)):
        label = labels[i]
        pos_idx = torch.where(labels == label)[0]
        neg_idx = torch.where(labels != label)[0]
        pos_idx = pos_idx[pos_idx != i]
        if len(pos_idx) == 0 or len(neg_idx) == 0: continue
        hardest_pos = pos_idx[torch.argmax(dists[i, pos_idx])]
        hardest_neg = neg_idx[torch.argmin(dists[i, neg_idx])]
        anchors.append(vectors[i]); positives.append(vectors[hardest_pos]); negatives.append(vectors[hardest_neg])
    if not anchors: return None, None, None
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def train_step(models, batch, loss_fn, optimizers, config, le_sp):
    for model in models.values(): model.train()

    if config['train_mode'] == 'gated':
        x3d, vmae, a, s = [b.long().cuda() if i >= 2 else b.cuda() for i, b in enumerate(batch)]
        fused, _ = models['fusion'](x3d, vmae); x = fused
    else:
        x, a, s = [b.long().cuda() if i >= 1 else b.cuda() for i, b in enumerate(batch)]

    if config['use_adversarial']:
        action_encoder = models['action_encoder']
        discriminator = models['discriminator']
        optimizer_disc = optimizers['discriminator']
        
        optimizer_disc.zero_grad()
        with torch.no_grad():
            a_vec_detached = action_encoder(x).detach()
        logits = discriminator(a_vec_detached)
        ce_loss = nn.CrossEntropyLoss()(logits, s)
        ce_loss.backward()
        optimizer_disc.step()
    
    main_optimizer = optimizers.get('encoder') or optimizers.get('main')
    main_optimizer.zero_grad()
    
    encoder = models['action_encoder'] if 'action_encoder' in models else models['net']
    a_vec = encoder(x)
    
    if config['loss_type'] == 'supcon':
        main_loss = loss_fn(a_vec, a)
    else:
        a_vec_norm = nn.functional.normalize(a_vec, dim=-1)
        anc, pos, neg = make_triplets_hard(a_vec_norm, a)
        if anc is None: return None
        main_loss = loss_fn(anc, pos, neg)

    total_loss = main_loss
    if config['use_adversarial']:
        logits = discriminator(a_vec)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        uniform_target = torch.full_like(logits, 1.0 / len(le_sp.classes_))
        adv_loss = nn.KLDivLoss(reduction='batchmean')(log_probs, uniform_target)
        total_loss += config['lambda_adv'] * adv_loss

    total_loss.backward()
    main_optimizer.step()
    
    return total_loss.item()


def evaluate_model(models, loader, config, loss_fn, fusion=None):
    for model in models.values(): model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            if config['train_mode'] == 'gated':
                x3d, vmae, a, s = [b.long().cuda() if i >= 2 else b.cuda() for i, b in enumerate(batch)]
                fused, _ = fusion(x3d, vmae); x = fused
            else:
                x, a, s = [b.long().cuda() if i >= 1 else b.cuda() for i, b in enumerate(batch)]

            encoder = models['action_encoder'] if 'action_encoder' in models else models['net']
            a_vec = encoder(x)
            
            if config['loss_type'] == 'supcon':
                loss = loss_fn(a_vec, a)
            else:
                a_vec_norm = nn.functional.normalize(a_vec, dim=-1)
                anc, pos, neg = make_triplets_hard(a_vec_norm, a)
                if anc is None: continue
                loss = loss_fn(anc, pos, neg)
            
            losses.append(loss.item())
            
    return np.mean(losses) if losses else float('inf')


def train_model(config, train_loader, val_loader, le_sp, trial, study_name, fusion=None):
    S = len(le_sp.classes_)
    sample_data = next(iter(train_loader))[0]
    D = config['fused_dim'] if fusion is not None else sample_data.shape[1]
    
    models = nn.ModuleDict()
    optimizers = {}
    
    if config['use_adversarial']:
        models['action_encoder'] = (ActionMLPNet(D, 256, 256) if config['use_mlp'] else ActionLinearNet(D, 256)).cuda()
        models['discriminator'] = SpeciesDiscriminator(256, S).cuda()
        params_enc = list(models['action_encoder'].parameters())
        if fusion:
            models['fusion'] = fusion
            params_enc.extend(fusion.parameters())
        optimizers['encoder'] = torch.optim.Adam(params_enc, lr=config['lr'])
        optimizers['discriminator'] = torch.optim.Adam(models['discriminator'].parameters(), lr=config['lr'])
    else:
        models['net'] = (SimpleMLPNet(D, 256, 256) if config['use_mlp'] else SimpleLinearNet(D, 256)).cuda()
        params_to_optimize = list(models['net'].parameters())
        if fusion:
            models['fusion'] = fusion
            params_to_optimize.extend(fusion.parameters())
        optimizers['main'] = torch.optim.Adam(params_to_optimize, lr=config['lr'])
        
    loss_fn = get_loss_fn(config['loss_type'], config.get('temperature', 0.07))
    
    run_name = wandb.run.name or "local-run"
    model_dir = Path(f"./models/{study_name}")
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"{run_name}_best.pth"
    
    best_val_loss, patience, no_improve = float('inf'), 50, 0
    
    for epoch in range(200):
        train_losses = []
        desc = f"[{config['train_mode'].upper()}][{config['loss_type']}] Epoch {epoch+1:03d}"
        for batch in tqdm(train_loader, desc=desc):
            loss = train_step(models, batch, loss_fn, optimizers, config, le_sp)
            if loss is not None: train_losses.append(loss)
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('inf')
        fusion_for_eval = models['fusion'] if 'fusion' in models else None
        avg_val_loss = evaluate_model(models, val_loader, config, loss_fn, fusion=fusion_for_eval)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            print(f"✅ Best val_loss improved to {best_val_loss:.4f}. Saving model...")
            torch.save(models.state_dict(), save_path)
            trial.set_user_attr("model_save_path", str(save_path))
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    return best_val_loss

# --- Optuna 目的関数 ---
def objective(trial):
    loss_type = trial.suggest_categorical("loss_type", ["improved", "cosine", "default", "supcon"])
    train_mode = trial.suggest_categorical("train_mode", ["flow", "mae", "gated"])
    
    flow_preprocessing = 'n/a'
    if train_mode in ['flow', 'gated']:
        flow_preprocessing = trial.suggest_categorical("flow_preprocessing", ["normal", "centered"])

    config = {
        "use_mlp": trial.suggest_categorical("use_mlp", [True, False]),
        "loss_type": loss_type,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "lambda_action": 1.0,
        "lambda_adv": trial.suggest_float("lambda_adv", 0.01, 0.5),
        "train_mode": train_mode,
        "use_adversarial": trial.suggest_categorical("use_adversarial", [True, False]),
        "flow_preprocessing": flow_preprocessing,
        "datatype": 'animalkingdom', "batch_size": 64, "fused_dim": 512, "feature_dim": 256
    }
    
    if loss_type == 'supcon':
        config['temperature'] = trial.suggest_float('temperature', 0.05, 0.5)

    run_name = f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}"
    wandb.init(project="optuna_disentangle_supcon", config=config, group=config['train_mode'], name=run_name, reinit=True)
    
    datatype = config['datatype']
    full_csv_path = f"./label/{datatype}/train/labels.csv"
    full_df = pd.read_csv(full_csv_path)
    le_act = LabelEncoder().fit(full_df['action'])
    le_sp = LabelEncoder().fit(full_df['species'])
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df['action'])
    
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
        fusion_model = GatedFusion(2048, 768, config['fused_dim']).cuda()
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    study_name = trial.study.study_name
    best_val_loss = train_model(config, train_loader, val_loader, le_sp, trial, study_name, fusion=fusion_model)

    wandb.finish()
    return best_val_loss

# --- メイン実行ブロック ---
def main():
    storage_name = "sqlite:///optuna_study.db"
    study_name = "disentangle-study-0801"
    study = optuna.create_study(direction="minimize", storage=storage_name, study_name=study_name, load_if_exists=True)
    study.optimize(objective, n_trials=100)
    print("Best Trial:", study.best_trial.params)

    print("\n--- Starting model cleanup ---")
    TOP_K_TO_KEEP = 5

    completed_trials = sorted(
        study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]),
        key=lambda t: t.value
    )

    # 削除対象のTrialを取得
    trials_to_delete = completed_trials[TOP_K_TO_KEEP:]

    deleted_count = 0
    for trial in trials_to_delete:
        save_path_str = trial.user_attrs.get("model_save_path")
        if save_path_str:
            save_path = Path(save_path_str)
            if save_path.exists():
                save_path.unlink()
                deleted_count += 1

    print(f"Cleanup finished. Kept top {len(completed_trials) - len(trials_to_delete)} models.")
    print(f"Deleted {deleted_count} model checkpoints.")


if __name__ == "__main__":
    main()
