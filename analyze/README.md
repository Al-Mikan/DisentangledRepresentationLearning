# analyze フォルダの使い方

データやラベルの分布、動画の基本統計を手軽に可視化・点検するための補助スクリプト群です。出力画像は `analyze/` 配下に保存されます。

## 依存関係

- Python 3.9+
- 共通: pandas, matplotlib
- 動画関連:
  - OpenCV (cv2): `fps.py`, `time.py`
  - decord: `analyze_label.py`（動画のフレーム数取得に使用）

必要に応じてインストールしてください（例）:
```powershell
pip install pandas matplotlib opencv-python decord
```

## スクリプト一覧

- analyze_label.py
  - 目的: `label/animalkingdom/train/labels.csv` を読み、action/species のクラス数・出現数を表示。動画ファイルの存在チェックも行い、欠損を列挙。
  - 入力: `./label/animalkingdom/train/labels.csv`（列に `video_path`, `action`, `species` を想定）
  - 出力: 標準出力（統計）。フレーム分布のプロット保存コードはコメントアウト済み（必要なら解除）。

- analyze_metadata.py
  - 目的: `AR_metadata.csv` を読み、mammal のみを抽出して 1動画あたりのラベル数分布や（単一ラベルの動画に限定した）species/action 出現数を可視化。
  - 入力: リポジトリルートの `AR_metadata.csv`（列: `list_animal_parent_class`, `list_animal_action`, `list_animal` など、文字列化されたリストを想定）
  - 出力: `analyze/label_count_histogram_mammal.png`, `analyze/single_label_action_counts.png`, `analyze/single_label_species_counts.png`

- fps.py
  - 目的: `./video` 配下の mp4 を走査し、FPS 分布を可視化。
  - 入力: `./video/**/*.mp4`
  - 出力: `analyze/fps_histogram.png`

- time.py
  - 目的: `./video` 配下の mp4 の再生時間（秒）分布を可視化。
  - 入力: `./video/**/*.mp4`
  - 出力: `analyze/video_duration_histogram.png`

## よくある注意点

- 文字化け対策: `analyze_metadata.py` では日本語フォントに Meiryo を指定しています。環境にない場合はコメントアウトするか、お使いのフォント名に変更してください。
- 出力先ディレクトリ: 画像保存前に `os.makedirs('./analyze', exist_ok=True)` を呼ぶよう修正しています。権限エラーが出る場合はカレントディレクトリをリポジトリルートにしてください。
- 入力パス: スクリプトは相対パスを想定しています。リポジトリのルートをカレントにして実行してください。

## 実行例（Windows PowerShell）

```powershell
# FPS 分布
python .\analyze\fps.py

# 動画長分布
python .\analyze\time.py

# ラベル基本統計
python .\analyze\analyze_label.py

# メタデータの分析（mammal 抽出など）
python .\analyze\analyze_metadata.py
```

## トラブルシュート

- `ModuleNotFoundError`: 必要なライブラリをインストールしてください（上の pip コマンド例参照）。
- `FileNotFoundError`: 入力ファイル/ディレクトリのパスが合っているか確認し、必要ならスクリプト内のパスを編集してください。
- `PermissionError`: PowerShell を管理者で実行するか、出力先の権限を確認してください。
