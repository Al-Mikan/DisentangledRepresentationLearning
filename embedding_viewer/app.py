from __future__ import annotations
import io
import json
import base64
from pathlib import Path

from flask import Flask, request, render_template, jsonify, send_file, Response
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import umap

app = Flask(__name__)

# ======================================================
# 設定
# ======================================================
TRAIN_RESULT_ROOT = Path(
    "/home/asel/Documents/labo/DisentangledRepresentationLearning/train_result"
)
VIDEO_ROOT = Path("/home/asel/Documents/labo/DisentangledRepresentationLearning") # 動画ルート

# ======================================================
# JSONL loader
# ======================================================
def load_embeddings(path: Path):
    vecs, labels, sources, video_paths = [], [], [], []
    with open(path, "r") as f:
        for line in f:
            o = json.loads(line)
            vecs.append(o["vector"])
            labels.append(o["label"])
            sources.append(o.get("source", "unknown"))
            # videopath取得 (絶対パス or 相対パス)
            vp = o.get("videopath", "")
            # windowsパス区切り対応などはここで行う
            vp = vp.replace("\\", "/")
            video_paths.append(vp)
            
    return (
        np.asarray(vecs, np.float32),
        np.asarray(labels),
        np.asarray(sources),
        np.asarray(video_paths),
    )

# ======================================================
# ラベル集合のロード（train / test 別）
# ======================================================
def load_label_set(eval_dir: Path, split: str):
    labels = set()
    # test.jsonl あるいは train.jsonl を探す
    for p in eval_dir.glob("*.jsonl"): 
        # ファイル名末尾判定: {model}_train.jsonl / {model}_test.jsonl
        if p.stem.endswith(f"_{split}"):
            with open(p, "r") as f:
                for line in f:
                    try:
                        labels.add(json.loads(line)["label"])
                    except:
                        pass
    return sorted(labels)

# ======================================================
# ARI / NMI
# ======================================================
def compute_ari_nmi(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return None, None

    # 高速化のため最大サンプル数制限などを入れても良いが今はそのまま
    pred = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(X)

    return (
        adjusted_rand_score(labels, pred),
        normalized_mutual_info_score(labels, pred),
    )

# ======================================================
# 動画配信設定
# ======================================================
@app.route("/video/<path:filepath>")
def serve_video(filepath):
    """
    動画ファイルを配信するエンドポイント
    セキュリティ: 本来はディレクトリトラバーサル対策が必要だが研究用なので簡易実装
    """

    
    p = Path("/" + filepath) if not filepath.startswith("/") else Path(filepath)
    
    if not p.exists():
        # 相対パスかもしれないので VIDEO_ROOT 加味
        p2 = VIDEO_ROOT / filepath
        if p2.exists():
            p = p2
        else:
            return f"Video not found: {p}", 404

    return send_file(p, mimetype="video/mp4")


# ======================================================
# /
# ======================================================
@app.route("/", methods=["GET"])
def index():
    if not TRAIN_RESULT_ROOT.exists():
        return f"Error: TRAIN_RESULT_ROOT not found: {TRAIN_RESULT_ROOT}"

    dates = sorted(p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir())

    selected_date = request.args.get("date")
    selected_run = request.args.get("run")
    selected_test_set = request.args.get("test_set")

    runs = []
    test_sets = []
    model_names = []
    train_labels = []
    test_labels = []

    if selected_date:
        date_dir = TRAIN_RESULT_ROOT / selected_date
        if date_dir.exists():
            runs = sorted(
                p.name for p in date_dir.iterdir()
                if p.is_dir() and p.name.startswith("run_")
            )

    if selected_date and selected_run:
        eval_root = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        if eval_root.exists():
            test_sets = sorted(d.name for d in eval_root.iterdir() if d.is_dir())

    if selected_date and selected_run and selected_test_set:
        current_eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval" / selected_test_set / "jsonl"
        if not current_eval_dir.exists():
            # フォールバック: eval直下にjsonlがある場合
            current_eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        
        if current_eval_dir.exists():
            train_labels = load_label_set(current_eval_dir, "train")
            test_labels = load_label_set(current_eval_dir, "test")
            model_names = sorted({
                p.stem.replace("_train", "").replace("_test", "")
                for p in current_eval_dir.glob("*.jsonl")
            })

    return render_template(
        "index.html",
        dates=dates,
        runs=runs,
        test_sets=test_sets,
        selected_date=selected_date,
        selected_run=selected_run,
        selected_test_set=selected_test_set,
        
        model_names=model_names,
        train_labels=train_labels,
        test_labels=test_labels,
        
        # 初期パラメータ
        perplexity=30,
        learning_rate=200,
        early_exaggeration=12,
        n_neighbors=15,
        min_dist=0.1,
    )


# ======================================================
# /read : ラベル情報とモデル一覧を取得してページを再描画
# ======================================================
@app.route("/read", methods=["POST"])
def read_labels():
    date = request.form["date"]
    run = request.form["run"]

    # evalフォルダを探す (testセットごとにフォルダが切られている可能性あり)
    # 構造: run_xxx/eval/test_set_name/jsonl/*.jsonl
    run_dir = TRAIN_RESULT_ROOT / date / run
    eval_root = run_dir / "eval"
    
    if not eval_root.exists():
         return f"Error: eval dir not found: {eval_root}"

    # testセットフォルダの列挙
    test_sets = [d.name for d in eval_root.iterdir() if d.is_dir()]
    selected_test_set = request.form.get("test_set", test_sets[0] if test_sets else "")
    
    # 選択されたtestセットのフォルダ
    current_eval_dir = eval_root / selected_test_set / "jsonl"
    
    if not current_eval_dir.exists():
        # 古い構造への対応 (eval直下にjsonlがない場合など)
         # もし eval/*.jsonl ならそちらを見る
         if list(eval_root.glob("*.jsonl")):
             current_eval_dir = eval_root
         else:
             # まだ何もない
             train_labels, test_labels, model_names = [], [], []

    if current_eval_dir.exists():
        train_labels = load_label_set(current_eval_dir, "train")
        test_labels = load_label_set(current_eval_dir, "test")
        
        # モデル名: xxx_train.jsonl から xxx を抽出
        model_names = sorted({
            p.stem.replace("_train", "").replace("_test", "")
            for p in current_eval_dir.glob("*.jsonl")
        })
    else:
        train_labels, test_labels, model_names = [], [], []

    # 再度日付リストなどを取得してrender
    dates = sorted(p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir())
    runs = sorted(
        p.name for p in (TRAIN_RESULT_ROOT / date).iterdir()
        if p.is_dir() and p.name.startswith("run_")
    )

    return render_template(
        "index.html",
        dates=dates,
        runs=runs,
        selected_date=date,
        selected_run=run,
        
        test_sets=test_sets,
        selected_test_set=selected_test_set,

        model_names=model_names,
        train_labels=train_labels,
        test_labels=test_labels,
        
        # フォームの値を維持
        perplexity=30,
        learning_rate=200,
        early_exaggeration=12,
        n_neighbors=15,
        min_dist=0.1,
    )

# ======================================================
# /api/plot_data : Plotly用のJSONデータを返す (Ajax用)
# ======================================================
@app.route("/api/plot_data", methods=["POST"])
def get_plot_data():
    data = request.json
    date = data["date"]
    run = data["run"]
    test_set = data.get("test_set", "")
    model = data["model"]
    
    selected_train_labels = set(data.get("train_labels", []))
    selected_test_labels = set(data.get("test_labels", []))
    
    view_mode = data.get("view_mode", "all")
    method = data.get("method", "tsne")
    
    # パス構築
    # run_xxx/eval/{test_set}/jsonl/{model}_{split}.jsonl
    eval_dir = TRAIN_RESULT_ROOT / date / run / "eval"
    if test_set:
        eval_dir = eval_dir / test_set / "jsonl"
    
    paths = []
    if view_mode in ("all", "train"):
        paths.append(eval_dir / f"{model}_train.jsonl")
    if view_mode in ("all", "test"):
        paths.append(eval_dir / f"{model}_test.jsonl")

    vecs, labels, sources, video_paths = [], [], [], []
    for p in paths:
        if not p.exists(): continue
        X, l, s, v = load_embeddings(p)
        vecs.append(X)
        labels.append(l)
        sources.append(s)
        video_paths.append(v)

    if not vecs:
        return jsonify({"error": "No data found"})

    X = np.concatenate(vecs)
    labels = np.concatenate(labels)
    sources = np.concatenate(sources)
    video_paths = np.concatenate(video_paths)

    # フィルタリング
    # train/test それぞれ選択されたラベルに含まれるか
    mask_train = (sources == "train") & np.isin(labels, list(selected_train_labels))
    mask_test  = (sources == "test")  & np.isin(labels, list(selected_test_labels))
    mask = mask_train | mask_test
    
    if np.sum(mask) == 0:
         return jsonify({"error": "No samples matched the filter"})

    X = X[mask]
    labels = labels[mask]
    sources = sources[mask]
    video_paths = video_paths[mask]

    # 次元削減
    if len(X) < 2:
         return jsonify({"error": "Not enough samples to plot"})

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=int(data.get("perplexity", 30)),
            learning_rate=float(data.get("learning_rate", 200)),
            early_exaggeration=float(data.get("early_exaggeration", 12)),
            init=data.get("init", "pca"),
            angle=float(data.get("angle", 0.5)),
            random_state=42
        )
        X2d = reducer.fit_transform(X)
    else:
        reducer = umap.UMAP(
            n_neighbors=int(data.get("n_neighbors", 15)),
            min_dist=float(data.get("min_dist", 0.1)),
            metric="cosine",
            random_state=42
        )
        X2d = reducer.fit_transform(X)

    # ARI / NMI 計算
    ari, nmi = compute_ari_nmi(X, labels)

    # Plotly用データ構築
    # ラベルごとにTraceを分ける（凡例用）と、Web上で重いので
    # 全点を1つのScatter（またはTrain/Testで2つ）にして、色データを持たせるのが軽量だが
    # 凡例クリックでOn/OffしたいならラベルごとにTraceを作るのが定石
    
    unique_labels = sorted(list(set(labels)))
    traces = []
    
    # 色パレット生成 (簡易的にHLSなどで自前生成もできるが、Plotlyのデフォ色が使える)
    # ここではラベルごとにTraceを作る
    
    for lab in unique_labels:
        # Train
        m_tr = (labels == lab) & (sources == "train")
        if np.any(m_tr):
            traces.append({
                "x": X2d[m_tr, 0].tolist(),
                "y": X2d[m_tr, 1].tolist(),
                "mode": "markers",
                "name": f"{lab} (train)",
                "marker": {"symbol": "circle", "size": 8, "opacity": 0.7},
                "text": video_paths[m_tr].tolist(), # ホバー用
                "customdata": video_paths[m_tr].tolist(), # クリックイベント用
                "hovertemplate": f"Label: {lab}<br>Source: Train<br>Path: %{{customdata}}<extra></extra>"
            })

        # Test
        m_te = (labels == lab) & (sources == "test")
        if np.any(m_te):
            traces.append({
                "x": X2d[m_te, 0].tolist(),
                "y": X2d[m_te, 1].tolist(),
                "mode": "markers",
                "name": f"{lab} (test)",
                "marker": {
                    "symbol": "triangle-up", 
                    "size": 10, 
                    "opacity": 0.9,
                    "line": {"width": 1, "color": "white"}
                },
                "text": video_paths[m_te].tolist(),
                "customdata": video_paths[m_te].tolist(),
                "hovertemplate": f"Label: {lab}<br>Source: Test<br>Path: %{{customdata}}<extra></extra>"
            })

    return jsonify({
        "traces": traces,
        "layout": {
            "title": f"{method.upper()} Projection",
            "hovermode": "closest",
            "dragmode": "pan",
            "plot_bgcolor": "#1e1e1e", # ダークモード背景
            "paper_bgcolor": "#1e1e1e",
            "font": {"color": "#e0e0e0"}
        },
        "metrics": {
            "ari": f"{ari:.4f}" if ari else "N/A",
            "nmi": f"{nmi:.4f}" if nmi else "N/A"
        }
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
