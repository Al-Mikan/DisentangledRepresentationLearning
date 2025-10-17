# exec ディレクトリのスクリプト説明

このディレクトリには、特徴抽出・学習・評価・可視化など、実験の主要処理を行うスクリプトが含まれています。主な入出力パスと、Windows PowerShell の簡易実行例も併記します。

## トレーニング・評価

- train.py
  - 目的: flow/x3d・VideoMAE・ゲート融合の各特徴から行動表現を学習。Triplet/Cosine/SupConに対応。Optunaで探索、W&Bでロギング。敵対的正則化で種情報の抑制も可。
  - 入力: `label/<datatype>/train/labels.csv`、`x3d_output/`（または`x3d_output_centered/`）、`vector/<datatype>/train/vectors_sliding_base.json`
  - 出力: `models/<datatype>/<study_name>/<run_name>_best.pth`（ベストのみ）、`optuna_study.db`、`alpha_logs/`（gated時にαを保存）
  - 実行例: `python .\exec\train.py`

- evaluate.py
  - 目的: OptunaのStudyから上位トライアルを自動復元し、テストデータでARI/NMIを評価。t-SNE/UMAPの可視化も保存。
  - 入出力: `optuna_study.db`、`label/<datatype>/(train|test)/*.csv`、`x3d_output(_centered)/`、`vector/.../vectors_sliding_base.json`
  - 出力: `results/<STUDY_NAME>_summary.csv`、`results/<STUDY_NAME>/*_{tsne|umap}.png`

- visualize.py（任意）
  - 目的: 任意のチェックポイントを数本だけ読み、t-SNEの図を素早く作成する軽量スクリプト。
  - 備考: evaluate.pyがあれば必須ではありません。必要ならα（ゲート係数）の可視化コードを追記して使用可能。

## 特徴抽出・前処理

- feature_extractor.py
  - 目的: 動画からフレーム特徴（例: VideoMAE）を抽出し、JSONに保存。
  - 出力: `vector/<datatype>/train/vectors_sliding_base.json` 等
  - 例: `python .\exec\feature_extractor.py --mode adaptive3d`

- optical_flow.py / flow_and_x3d_tensor.py
  - 目的: 光学フローやX3D由来のテンソル出力を作る補助スクリプト。
  - 出力: `x3d_output/<datatype>/.../*.npy`、`x3d_output_centered/<datatype>/.../*.npy`

## モデル・損失・ユーティリティ

- model.py
  - 埋め込み器: SimpleLinear/MLP、ActionLinear/MLP（敵対学習用）、SpeciesDiscriminator（敵対側）、GatedFusion（flow×vmae融合）。
  - 安定化: 埋め込みはL2正規化、GatedFusionにはLayerNorm/Dropoutを導入済み。`forward`は`(fused, alpha)`を返します。

- triplet_losses.py
  - Triplet関連の損失や補助実装（必要に応じて使用）。

- utils.py
  - データセット（MAE/Flow/融合）やユーティリティ関数を提供。train/evaluateから使用されます。

- RAFT/
  - RAFT（光学フロー）実装。`optical_flow.py`等の下支え。

## 補足: 保存ポリシーとαログ

- モデル保存（train.py）
  - 各トライアルで検証損失が改善した時のみ上書き保存。終了後にStudy全体から上位だけを残すクリーンアップを実施します。
  - 保存は`state_dict`（ModuleDict）。非敵対は`net`、敵対は`action_encoder`/`discriminator`、gatedは`fusion`も含みます。

- α（ゲート係数）の保存
  - gatedモード時、各エポックの全バッチ分のαを結合し `alpha_logs/alpha_trial{trial}_epoch{epoch}_*.npy` に保存します。

## 実行ヒント（Windows PowerShell）

- ルートで実行: `python .\exec\train.py` のように、リポジトリルートをカレントにするのが安全です。
- 一時的に環境変数を付けて実行: `$env:OMP_NUM_THREADS='2'; python .\exec\evaluate.py`