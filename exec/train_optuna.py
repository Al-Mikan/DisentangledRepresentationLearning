import yaml
import optuna
from typing import Any, Dict
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
from train_core import train_model, build_datasets_and_loaders, _build_inference_models, _compute_clustering_metrics, cleanup_memory
from sklearn.model_selection import StratifiedKFold
import numpy as np


# ===============================
# YAML読み込みとOptunaヘルパー関数
# ===============================
import re
from pathlib import Path

def get_base_video_id(video_path: str) -> str:
    """
    xxx_aug0 / xxx_aug1 / xxx_aug2 → xxx
    """
    stem = Path(str(video_path).replace("\\", "/")).stem
    stem = re.sub(r"_aug[0-2]$", "", stem)
    return stem


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
    next_idx,
    search_space=None,
    fixed_config=None,
):
    if fixed_config is not None:
        # suggest_from_yml を通さないので、勝手に値が変わるリスクがゼロになる
        yaml_cfg = fixed_config
        
        # ログ用に trial.params に無理やり登録しておく（可読性のため）
        for k, v in yaml_cfg.items():
            try:
                trial.set_user_attr(k, v)
            except Exception as e: 
                print(f"エラーが出ました: {e}")
    elif search_space is not None:
        yaml_cfg = suggest_from_yml(trial, search_space)
    else:
        yaml_cfg = trial.params


    use_cv = bool(yaml_cfg.get("use_cross_validation", True))
    n_splits = int(yaml_cfg.get("cv_splits", 3))

    use_aug = bool(yaml_cfg.get("use_augmentation", False))

    config = {
        # =========================
        # 学習設定・モード関連
        # =========================
        "train_mode": yaml_cfg.get("train_mode", "gated"),  # 学習対象モード（mae / flow / gated）
        "loss_type": yaml_cfg.get("loss_type", "triplet"),  # 損失関数の種類（triplet, cosineなど）
        "adversarial": yaml_cfg.get("adversarial", "gan"),  # 敵対的学習の有効化（off / gan / kl）
        "flow_preprocessing": yaml_cfg.get("flow_preprocessing", "centered"),  # Optical Flow特徴の前処理（normal / centered）
        "pooling":  bool(yaml_cfg.get("pooling", True)),  # VideoMAE特徴のプーリング使用有無
        "frame_stride": int(yaml_cfg.get("frame_stride", 1)),  # pooling=False時のフレーム間隔（16=重複なし）
        # =========================
        # 最適化・学習率関連
        # =========================
        "lr_enc": float(yaml_cfg.get("lr_enc", 1e-4)),   # エンコーダ側の学習率
        "lr_disc": float(yaml_cfg.get("lr_disc", 1e-4)), # 識別器(Discriminator)の学習率（GAN・KL使用時のみ）
        "weight_decay": float(yaml_cfg.get("weight_decay", 1e-5)),  # L2正則化（Weight Decay）の強さ
        "lambda_adv": float(yaml_cfg.get("lambda_adv", 0.1)),  # 敵対的損失の重み（adversarial有効時）
        "lambda_cls": float(yaml_cfg.get("lambda_cls", 0.0)),  # 行動分類CE損失の重み（0で無効化）

        # =========================
        # 損失関数パラメータ
        # =========================
        "triplet_margin": float(yaml_cfg.get("triplet_margin", 0.1)),  # Triplet Loss のマージン値
        "ms_alpha": float(yaml_cfg.get("ms_alpha", 2.0)),  # MultiSimilarityLoss: positive pairsの指数重み
        "ms_beta": float(yaml_cfg.get("ms_beta", 50.0)),   # MultiSimilarityLoss: negative pairsの指数重み
        "ms_base": float(yaml_cfg.get("ms_base", 0.5)),    # MultiSimilarityLoss: 指数のシフト値

        # =========================
        # バッチ・エポックなどの学習制御
        # =========================
        "batch_size": int(yaml_cfg.get("batch_size", 64)),  # 学習バッチサイズ
        "epochs": int(yaml_cfg.get("epochs", 100)),         # 学習エポック数

        # =========================
        # データセット・モデル構造
        # =========================
        "datatype": yaml_cfg.get("datatype", "animalkingdom"),
        "train_label_paths": yaml_cfg.get("train_label_paths", None),
        "test_label_paths": yaml_cfg.get("test_label_paths", None),

        "fused_dim": int(yaml_cfg.get("fused_dim", 512)),   # GatedFusionで統合後の特徴ベクトル次元数
        "feature_dim": int(yaml_cfg.get("feature_dim", 256)), # 各モーダルの出力特徴次元数

        # =========================
        # 実験管理・ログ関連
        # =========================
        "project_name": yaml_cfg.get("project_name", "optuna"),  # wandb上のプロジェクト名
        "experiment_name": yaml_cfg.get("experiment_name", "study"),  # wandb上の実験グループ名
        
        # =========================
        # Curriculum Learning (段階的学習)
        # =========================
        "curriculum_learning": bool(yaml_cfg.get("curriculum_learning", False)),
        "curriculum_phases": yaml_cfg.get("curriculum_phases", {}),
    }

    run_name = (
            f"trial_{trial.number}_{config['train_mode']}_{config['loss_type']}_adv{config['adversarial']}_pool{config.get('pooling', True)}"
            + (f"_{config['flow_preprocessing']}" if config.get("train_mode") in ["flow", "gated"] else "")
        )

    from datetime import datetime
    date_str = datetime.now().strftime("%m%d")  # MMDD形式
    project_name = config.get("project_name", "optuna") + f"_{date_str}_run{next_idx:03d}"

    wandb.init(project=project_name, config=config, name=run_name, reinit=True, save_code=True)

    trial.set_user_attr("train_mode", config["train_mode"])
    trial.set_user_attr("adversarial", config["adversarial"])
    trial.set_user_attr("loss_type", config["loss_type"])

    seed = 42
    val_scores = []
    fold_logs = []

    if use_aug:
        full_df = full_df.copy()
        full_df["base_video_id"] = full_df["video_path"].apply(get_base_video_id)

        # 🔒 元動画の集合を作る
        orig_df = full_df[~full_df["video_path"].str.contains("_aug")]
        valid_base_ids = set(orig_df["base_video_id"])

        # 🔒 augmentation のうち「元動画があるもの」だけ残す
        full_df = full_df[full_df["base_video_id"].isin(valid_base_ids)]
    else:
        full_df["base_video_id"] = full_df["video_path"]
    # ======================================
    # Cross-Validation ありの場合
    # ======================================
    if use_cv:
        seed = seed + trial.number

        # base_video_id 単位で代表ラベルを作る
        base_df = (
            full_df
            .groupby("base_video_id")
            .first()
            .reset_index()
        )

        kfold = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed
        )

        for fold_idx, (train_base_idx, val_base_idx) in enumerate(
            kfold.split(base_df, base_df["action"])
        ):
            train_base_ids = set(base_df.iloc[train_base_idx]["base_video_id"])
            val_base_ids   = set(base_df.iloc[val_base_idx]["base_video_id"])

            train_df = full_df[full_df["base_video_id"].isin(train_base_ids)].reset_index(drop=True)
            val_df   = full_df[full_df["base_video_id"].isin(val_base_ids)].reset_index(drop=True)

            fold_score = _run_one_fold(
                trial, config,
                train_df=train_df,
                val_df=val_df,
                le_act=le_act, le_sp=le_sp,
                results_root=results_root,
                fold=fold_idx,
            )
            if fold_score is not None:
                val_scores.append(fold_score)

    # ======================================
    # Cross-Validation OFF（単一 split）
    # ======================================
    else:
        base_df = (
        full_df
        .groupby("base_video_id")
        .first()
        .reset_index()
        )

        train_base, val_base = train_test_split(
            base_df,
            test_size=0.1,
            shuffle=True,
            random_state=seed,
            stratify=base_df["action"],
        )

        train_df = full_df[full_df["base_video_id"].isin(train_base["base_video_id"])].reset_index(drop=True)
        val_df   = full_df[full_df["base_video_id"].isin(val_base["base_video_id"])].reset_index(drop=True)

        fold_score = _run_one_fold(
            trial, config,
            train_df, val_df,
            le_act, le_sp,
            results_root, fold=0
        )
        if fold_score is not None:
            val_scores.append(fold_score)


    # ======================================
    # 最終スコアの計算
    # ======================================
    if len(val_scores) == 0:
        print("⚠️ 全fold失敗 → score=0を返す")
        return 0.0

    mean_score = float(np.mean(val_scores))
    print(f"\n📊 Final Score = {mean_score:.4f}")

    trial.set_user_attr("cv_scores", val_scores)
    trial.set_user_attr("cv_mean", mean_score)
    
    # =============================
    # 🔥 wandb.summary にまとめて保存
    # =============================
    wandb.summary["trial_number"] = trial.number
    wandb.summary["lambda_adv"] = float(config.get("lambda_adv", 0.0))
    wandb.summary["lambda_cls"] = float(config.get("lambda_cls", 0.0))
    wandb.summary["cv_scores"] = val_scores
    wandb.summary["cv_mean"] = mean_score
    wandb.summary["fold_logs"] = fold_logs

    wandb.finish()

    return mean_score

def check_dataset_shapes(train_loader, mode):
    batch = next(iter(train_loader))
    print("=== Checking Dataset ===")
    if mode == "gated":
        x3d, mae, a, s = batch
        print("x3d:", x3d.shape)
        print("mae:", mae.shape)
    else:
        x, a, s = batch
        print("x:", x.shape)
    print("=========================")

def _run_one_fold(
    trial,
    config,
    train_df,
    val_df,
    le_act,
    le_sp,
    results_root,
    fold,
):
    LOG_FOLD = 0

    try:
        train_loader, val_loader, fusion_model = build_datasets_and_loaders(
            config, train_df, val_df, le_act, le_sp
        )

        check_dataset_shapes(train_loader, config["train_mode"])

        best_val_loss = train_model(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            le_sp=le_sp,
            le_act=le_act,
            trial=trial,
            study_name="optuna",
            fusion=fusion_model,
            results_root=results_root,
            fold_idx=fold,
            log_fold=LOG_FOLD,
        )

        model_path = trial.user_attrs.get("model_save_path")
        if not model_path:
            return None

        # =========================
        # inference model 構築
        # =========================
        if config["train_mode"] == "gated":
            D = int(config["fused_dim"])
        elif config["train_mode"] == "flow":
            D = 2048
        elif config["train_mode"] == "mae":
            D = 768
        else:
            raise ValueError("Unknown train_mode")

        inf_models = _build_inference_models(config, D=D, fusion=fusion_model)
        state = torch.load(
            model_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        missing, unexpected = inf_models.load_state_dict(state, strict=False)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        # =========================
        # clustering evaluation
        # =========================
        metrics_val = _compute_clustering_metrics(inf_models, val_loader, config)
        metrics_tr  = _compute_clustering_metrics(inf_models, train_loader, config)

        # === main score (Agglomerative) ===
        score = float(metrics_val["agg_nmi"])  # or agg_ari

        print(
            f"Fold{fold+1} | "
            f"train(agg)={metrics_tr['agg_nmi']:.3f}, "
            f"val(agg)={metrics_val['agg_nmi']:.3f}, "
            f"score={score:.3f}, "
            f"val_loss={best_val_loss:.4f}"
        )

        # =========================
        # wandb logging
        # =========================
        if fold == LOG_FOLD:
            wandb.log(
                {
                    "fold": fold + 1,

                    # --- Agglomerative (main) ---
                    "agg_ari_train": metrics_tr["agg_ari"],
                    "agg_nmi_train": metrics_tr["agg_nmi"],
                    "agg_ari_val":   metrics_val["agg_ari"],
                    "agg_nmi_val":   metrics_val["agg_nmi"],
                }
            )

        return score

    except Exception as e:
        print(f"⚠️ Fold {fold+1} failed: {e}")
        return None

    finally:
        if "inf_models" in locals():
            del inf_models
        cleanup_memory()
