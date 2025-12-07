# exec ディレクトリのスクリプト説明

このディレクトリには、特徴抽出・学習・評価・可視化など、実験の主要処理を行うスクリプトが含まれています。主な入出力パスと、Windows PowerShell の簡易実行例も併記します。

## トレーニング・評価

- train.py

  - 目的: flow/x3d・VideoMAE・ゲート融合の各特徴から行動表現を学習。Triplet/Cosine/SupCon に対応。Optuna で探索、W&B でロギング。敵対的正則化で種情報の抑制も可。
  - 入力: `label/<datatype>/train/labels.csv`、`x3d_vector/`（または`x3d_vector_centered/`）、`vector/<datatype>/*`（各動画ごとのフォルダに `avg_pooling.npy` / `sliding_list/*.npy` を配置）
  - 出力: `models/<datatype>/<study_name>/<run_name>_best.pth`（ベストのみ）、`optuna_study.db`、`alpha_logs/`（gated 時に α を保存）
  - 実行例: `python .\exec\train.py`

- evaluate.py

  - 目的: Optuna の Study から上位トライアルを自動復元し、テストデータで ARI/NMI を評価。t-SNE/UMAP の可視化も保存。
  - 入出力: `optuna_study.db`、`label/<datatype>/(train|test)/*.csv`、`x3d_vector(_centered)/`、`vector/<datatype>/*`（各動画フォルダ）
  - 出力: `results/<STUDY_NAME>_summary.csv`、`results/<STUDY_NAME>/*_{tsne|umap}.png`

- visualize.py（任意）
  - 目的: 任意のチェックポイントを数本だけ読み、t-SNE の図を素早く作成する軽量スクリプト。
  - 備考: evaluate.py があれば必須ではありません。必要なら α（ゲート係数）の可視化コードを追記して使用可能。

## 特徴抽出・前処理

- feature_extractor.py

  - 目的: 動画からフレーム特徴（例: VideoMAE）を抽出し、JSON に保存。
  - 出力: 各動画フォルダに `avg_pooling.npy` または `sliding_list/*.npy` を保存
  - 例: `python .\exec\feature_extractor.py --mode adaptive3d`

- optical_flow.py / flow_and_x3d_tensor.py
  - 目的: 光学フローや X3D 由来のテンソル出力を作る補助スクリプト。
  - 出力: `x3d_vector/<datatype>/<video_name>/*.npy`、`x3d_vector_centered/<datatype>/<video_name>/*.npy`

## モデル・損失・ユーティリティ

- model.py

  - 埋め込み器: SimpleLinear/MLP、ActionLinear/MLP（敵対学習用）、SpeciesDiscriminator（敵対側）、GatedFusion（flow×vmae 融合）。
  - 安定化: 埋め込みは L2 正規化、GatedFusion には LayerNorm/Dropout を導入済み。`forward`は`(fused, alpha)`を返します。

- triplet_losses.py

  - Triplet 関連の損失や補助実装（必要に応じて使用）。

- utils.py

  - データセット（MAE/Flow/融合）やユーティリティ関数を提供。train/evaluate から使用されます。

- RAFT/
  - RAFT（光学フロー）実装。`optical_flow.py`等の下支え。

## 補足: 保存ポリシーと α ログ

- モデル保存（train.py）

  - 各トライアルで検証損失が改善した時のみ上書き保存。終了後に Study 全体から上位だけを残すクリーンアップを実施します。
  - 保存は`state_dict`（ModuleDict）。非敵対は`net`、敵対は`action_encoder`/`discriminator`、gated は`fusion`も含みます。

- α（ゲート係数）の保存
  - gated モード時、各エポックの全バッチ分の α を結合し `alpha_logs/alpha_trial{trial}_epoch{epoch}_*.npy` に保存します。

## 実行ヒント（Windows PowerShell）

- ルートで実行: `python .\exec\train.py` のように、リポジトリルートをカレントにするのが安全です。
- 一時的に環境変数を付けて実行: `$env:OMP_NUM_THREADS='2'; python .\exec\evaluate.py`

## Discord 通知（任意）

train.py には、Discord Webhook 経由で進捗を通知する軽量機能を追加しています（標準ライブラリのみ使用・デフォルト無効）。

- 有効化方法

  - 環境変数 `DISCORD_WEBHOOK_URL` に Webhook URL を設定すると有効化されます。
  - 各トライアルの開始/終了も通知したい場合は、`DISCORD_NOTIFY_TRIALS=1` を設定します（未設定なら Study 開始/終了とクリーンアップのみ通知）。
  - 例（PowerShell）: `$env:DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..."; $env:DISCORD_NOTIFY_TRIALS="1"; python .\exec\train.py`

- 送信タイミングと内容

  - Study 開始/終了（loss_type・study 名、ベストスコア・ハイパラ）
  - トライアル開始/終了（任意・trial 番号、val_loss、保存先）
  - クリーンアップ完了（保持/削除数とサマリーのパス）

- 停止方法
  - `DISCORD_WEBHOOK_URL` を未設定にする（通知は送られません）。
