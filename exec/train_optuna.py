import yaml
import optuna
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
import wandb
from train_core import train_model, build_datasets_and_loaders, _build_inference_models, _compute_clustering_metrics, cleanup_memory

# ===============================
# YAML読み込みとOptunaヘルパー関数
# ===============================

def load_config(yaml_path: str) -> Dict[str, Any]:
    """YAMLファイルを読み込む（固定値＋探索範囲）"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def suggest_from_yml(trial: optuna.trial.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """YAMLに基づき、Optuna探索または固定値を自動生成"""
    config = {}

    for key, val in search_space.items():
        # ネスト対応
        if isinstance(val, dict):
            config[key] = suggest_from_yml(trial, val)
            continue

        # 文字列そのまま
        if isinstance(val, str):
            config[key] = val
            continue

        # リスト処理
        if isinstance(val, list):
            # 要素1つ → 展開
            if len(val) == 1:
                config[key] = val[0]
                continue

            # 数値変換
            def try_float(x):
                try:
                    return float(x)
                except Exception:
                    return x
            val = [try_float(v) for v in val]

            # 2要素 → int or float の範囲探索を自動判定
            if len(val) == 2 and all(isinstance(v, (int, float)) for v in val):
                # 👇ここが重要：整数なら suggest_int、浮動小数なら suggest_float
                if all(isinstance(v, int) for v in val):
                    config[key] = trial.suggest_int(key, int(val[0]), int(val[1]))
                else:
                    config[key] = trial.suggest_float(key, val[0], val[1])
                continue

            # 3要素 + "log" → 対数スケール
            if len(val) == 3 and val[-1] == "log":
                config[key] = trial.suggest_float(key, val[0], val[1], log=True)
                continue

            # その他 → categorical
            config[key] = trial.suggest_categorical(key, val)
            continue

        # その他（数値やboolなど）
        config[key] = val

    return config


# ===============================
# Optuna目的関数
# ===============================

def objective(
    trial: optuna.trial.Trial,
    full_df,
    le_act,
    le_sp,
    results_root,
    search_space,
):
    """Optuna探索用の目的関数（1 trial = 1学習）"""
    if search_space is None:
        # Ablationの場合：Optuna探索はせず固定パラメータを使用
        yaml_cfg = trial.params
    else:
        yaml_cfg = suggest_from_yml(trial, search_space)

    config = {
        # =========================
        # 学習設定・モード関連
        # =========================
        "train_mode": yaml_cfg.get("train_mode", "gated"),  # 学習対象モード（mae / flow / gated）
        "loss_type": yaml_cfg.get("loss_type", "triplet"),  # 損失関数の種類（triplet, supcon, cosineなど）
        "adversarial_mode": yaml_cfg.get("adversarial", "off"),  # 敵対的学習の有効化（off / gan / kl）
        "flow_preprocessing": yaml_cfg.get("flow_preprocessing", "centered"),  # Optical Flow特徴の前処理（normal / centered）

        # =========================
        # 最適化・学習率関連
        # =========================
        "lr_enc": float(yaml_cfg.get("lr_enc", 1e-4)),   # エンコーダ側の学習率
        "lr_disc": float(yaml_cfg.get("lr_disc", 1e-4)), # 識別器(Discriminator)の学習率（GAN・KL使用時のみ）
        "weight_decay": float(yaml_cfg.get("weight_decay", 1e-5)),  # L2正則化（Weight Decay）の強さ
        "lambda_adv": float(yaml_cfg.get("lambda_adv", 0.1)),  # 敵対的損失の重み（adversarial_mode有効時）

        # =========================
        # 損失関数パラメータ
        # =========================
        "triplet_margin": float(yaml_cfg.get("triplet_margin", 0.1)),  # Triplet Loss のマージン値
        "temperature": float(yaml_cfg.get("temperature", 0.07)),       # SupCon Lossなどで使う温度パラメータ

        # =========================
        # バッチ・エポックなどの学習制御
        # =========================
        "batch_size": int(yaml_cfg.get("batch_size", 64)),  # 学習バッチサイズ
        "epochs": int(yaml_cfg.get("epochs", 100)),         # 学習エポック数

        # =========================
        # データセット・モデル構造
        # =========================
        "datatype": "animalkingdom",           # 使用するデータセットの種類（固定：Animal Kingdom）
        "fused_dim": int(yaml_cfg.get("fused_dim", 512)),   # GatedFusionで統合後の特徴ベクトル次元数
        "feature_dim": int(yaml_cfg.get("feature_dim", 256)), # 各モーダルの出力特徴次元数

        # =========================
        # 実験管理・ログ関連
        # =========================
        "project_name": yaml_cfg.get("project_name", "optuna"),  # wandb上のプロジェクト名
        "experiment_name": yaml_cfg.get("experiment_name", "study"),  # wandb上の実験グループ名
    }

    trial.set_user_attr("train_mode", config["train_mode"])
    trial.set_user_attr("adversarial_mode", config["adversarial_mode"])
    trial.set_user_attr("loss_type", config["loss_type"])

    run_name = (
        f"trial_{trial.number}_"
        f"{config['train_mode']}_{config['loss_type']}_"
        f"adv{config['adversarial_mode']}_"
        f"{config['flow_preprocessing']}"
    )

    wandb.init(project=config["project_name"], config=config, name=run_name, reinit=True, mode="disabled")

    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42, stratify=full_df["action"])
    train_loader, val_loader, fusion_model = build_datasets_and_loaders(config, train_df, val_df, le_act, le_sp)

    best_val_loss = train_model(config, train_loader, val_loader, le_sp, trial, study_name="optuna", fusion=fusion_model, results_root=results_root)

    model_path = trial.user_attrs.get("model_save_path")
    ari_val = nmi_val = combined_val = -1.0
    ari_train = nmi_train = combined_train = -1.0

    if model_path:
        try:
            
            inf_models = _build_inference_models(config, D=config["fused_dim"], fusion=fusion_model)
            state = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
            inf_models.load_state_dict(state, strict=False)
            ari_val, nmi_val, combined_val = _compute_clustering_metrics(inf_models, val_loader, config)
            ari_train, nmi_train, combined_train = _compute_clustering_metrics(inf_models, train_loader, config)
        except Exception as e:
            print(f"⚠️ Failed to compute clustering metrics: {e}")
    
    # ===============================
    # train-val差による過学習ペナルティ
    # ===============================
    if combined_train > 0 and combined_val > 0:
        score = combined_val - 0.3 * abs(combined_val - combined_train)
    else:
        score = combined_val  # fallback

    # ログと保存
    trial.set_user_attr("metric_combined_val", combined_val)
    trial.set_user_attr("metric_combined_train", combined_train)
    trial.set_user_attr("score_final", score)
    trial.set_user_attr("metric_ari", float(ari_val))
    trial.set_user_attr("metric_nmi", float(nmi_val))

    wandb.finish()
    cleanup_memory()

    print(f"[Trial {trial.number}] train={combined_train:.3f}, val={combined_val:.3f}, score={score:.3f}")

    return score
