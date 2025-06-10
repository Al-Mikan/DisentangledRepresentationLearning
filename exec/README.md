# exec ディレクトリのスクリプト説明

このディレクトリには、特徴抽出・学習・評価・可視化など、実験の主要な処理を行うスクリプトが含まれています。

## 各スクリプトの概要

- **feature_extractor.py**  
  VideoMAEを用いて動画から特徴ベクトルを抽出し、`exec/video_vectors.json` に保存します。
  1. Simple（均等サンプリング）16フレーム抽出
  `python feature_extractor.py --mode simple`
  2. Adaptive3D（全フレーム → 時間方向に3Dプーリングで16フレームへ圧縮）
  `python feature_extractor.py --mode adaptive3d`
  3. Adaptive1D（16フレームずつ分割 → [CLS] ベクトル → 1Dプーリング）
  `python feature_extractor.py --mode adaptive1d`

- **disentangle_triplet.py**  
  Triplet損失と直交性制約による2軸（行動・種）分離表現の学習を行います。学習済みモデルは `disentangled_triplet.pth` に保存されます。


- **evaluate_clustering.py**  
  抽出した特徴ベクトルに対してクラスタリングを行い、NMIなどの指標で評価します。

- **visualize.py**  
  学習済みモデルを用いて特徴ベクトルを2次元に可視化し、`result/embedding.png` などに保存します。

---

その他の補助的なスクリプトも含まれています。詳細は各ファイルの先頭コメントやdocstringを参照してください。