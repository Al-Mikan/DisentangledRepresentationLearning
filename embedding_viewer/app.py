from __future__ import annotations
import io
import json
from pathlib import Path
from flask import Flask, request, render_template_string
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------------------------------------------------------
# 設定：train_result の位置（ここだけ変えれば良い）
# --------------------------------------------------------
TRAIN_RESULT_ROOT = Path("/home/asel/Documents/labo/DisentangledRepresentationLearning/train_result")


# --------------------------------------------------------
# HTML テンプレート
# --------------------------------------------------------
PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Embedding Viewer</title>
</head>
<body style="font-family:sans-serif; margin:24px;">
  <h1>Embedding Viewer</h1>

  <!-- 日付フォルダ -->
  <form action="/" method="get">
    <label>日付フォルダ</label><br>
    <select name="date" onchange="this.form.submit()">
      {% for d in dates %}
        <option value="{{d}}" {% if d == selected_date %}selected{% endif %}>{{d}}</option>
      {% endfor %}
    </select>
  </form>
  <br>

  <!-- run_xxx -->
  {% if runs %}
  <form action="/" method="get">
    <input type="hidden" name="date" value="{{selected_date}}">

    <label>Run</label><br>
    <select name="run" onchange="this.form.submit()">
      {% for r in runs %}
        <option value="{{r}}" {% if r == selected_run %}selected{% endif %}>{{r}}</option>
      {% endfor %}
    </select>
  </form>
  <br>
  {% endif %}

  <!-- ファイル選択 -->
  {% if files %}
  <form action="/plot" method="post">
    <input type="hidden" name="date" value="{{selected_date}}">
    <input type="hidden" name="run" value="{{selected_run}}">

    <label>Embedding File (JSONL)</label><br>
    <select name="filename">
      {% for f in files %}
        <option value="{{f}}">{{f}}</option>
      {% endfor %}
    </select>
    <br><br>

    <!-- train/test/both -->
    <label>描画モード</label><br>
    <select name="mode">
      <option value="train">train のみ</option>
      <option value="test">test のみ</option>
      <option value="both">train + test</option>
    </select>
    <br><br>

    <label>Method</label>
    <select name="method">
      <option value="tsne">t-SNE</option>
      <option value="umap">UMAP</option>
    </select>
    <br><br>

    <!-- t-SNE パラメータ -->
    <h3>t-SNE Parameters</h3>
    perplexity: <input type="number" name="perplexity" value="30"><br>
    learning_rate: <input type="number" name="learning_rate" value="200"><br>
    n_iter: <input type="number" name="n_iter" value="1000"><br>
    early_exaggeration: <input type="number" name="early_exaggeration" value="12"><br>
    metric:
    <select name="metric">
      <option value="euclidean">euclidean</option>
      <option value="cosine">cosine</option>
    </select>
    <br><br>

    <!-- UMAP -->
    <h3>UMAP Parameters</h3>
    n_neighbors: <input type="number" name="n_neighbors" value="15"><br>
    min_dist: <input type="number" step="0.01" name="min_dist" value="0.1"><br>

    <br><br>
    <button type="submit">描画</button>
  </form>
  {% endif %}

  {% if image_data %}
  <hr>
  <h2>結果</h2>
  <img src="data:image/png;base64,{{ image_data }}" />
  <pre>{{ summary }}</pre>
  {% endif %}
</body>
</html>
"""


# --------------------------------------------------------
# Utility: JSONL loader
# --------------------------------------------------------
def load_embeddings(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    vecs, labels = [], []
    with open(path, "r") as f:
        for line in f:
            o = json.loads(line)
            vecs.append(o["vector"])
            labels.append(o["label"])

    return np.array(vecs, np.float32), np.array(labels)


# --------------------------------------------------------
# Plot
# --------------------------------------------------------
def plot_embedding(X2d, labels):
    plt.figure(figsize=(8, 6))
    uniq = np.unique(labels)

    for lab in uniq:
        mask = labels == lab
        plt.scatter(X2d[mask, 0], X2d[mask, 1], s=14, alpha=0.7, label=f"{lab}")

    plt.legend(fontsize=7)
    plt.xticks([])
    plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()

    import base64
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------
# Main UI
# --------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    # --- step 1: 日付フォルダ ---
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    selected_date = request.args.get("date", dates[-1] if dates else None)

    # --- step 2: run_xxx ---
    runs = []
    selected_run = None
    if selected_date:
        date_dir = TRAIN_RESULT_ROOT / selected_date
        runs = sorted([p.name for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
        selected_run = request.args.get("run", runs[0] if runs else None)

    # --- step 3: eval/*.jsonl ---
    files = []
    if selected_date and selected_run:
        eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        if eval_dir.exists():
            files = sorted([p.name for p in eval_dir.glob("*.jsonl")])

    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=selected_date,
        runs=runs,
        selected_run=selected_run,
        files=files,
        image_data=None,
        summary=None
    )


# --------------------------------------------------------
# Plot Route
# --------------------------------------------------------
@app.route("/plot", methods=["POST"])
def plot():
    date = request.form.get("date")
    run = request.form.get("run")
    filename = request.form.get("filename")
    mode = request.form.get("mode")  # train / test / both

    method = request.form.get("method")

    # t-SNE params
    perplexity = int(request.form.get("perplexity"))
    lr = float(request.form.get("learning_rate"))
    n_iter = int(request.form.get("n_iter"))
    early_ex = float(request.form.get("early_exaggeration"))
    metric = request.form.get("metric")

    # UMAP params
    n_neighbors = int(request.form.get("n_neighbors"))
    min_dist = float(request.form.get("min_dist"))

    run_dir = TRAIN_RESULT_ROOT / date / run / "eval"

    # === 読み込むファイルを決定 ===
    jsonl_paths = []
    if mode == "train":
        jsonl_paths = [run_dir / filename.replace("_test.jsonl", "_train.jsonl")]
    elif mode == "test":
        jsonl_paths = [run_dir / filename.replace("_train.jsonl", "_test.jsonl")]
    else:  # both
        jsonl_paths = [
            run_dir / filename.replace("_test.jsonl", "_train.jsonl"),
            run_dir / filename.replace("_train.jsonl", "_test.jsonl")
        ]

    # === 埋め込み読み込み ===
    X_list, y_list = [], []
    for p in jsonl_paths:
        if p.exists():
            X, y = load_embeddings(p)
            X_list.append(X)
            y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    labels = np.concatenate(y_list, axis=0)

    # === 次元削減 ===
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=lr,
            n_iter=n_iter,
            early_exaggeration=early_ex,
            metric=metric,
        )
        X2 = reducer.fit_transform(X)
        summary = f"t-SNE / N={len(X)} samples"
    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine"
        )
        X2 = reducer.fit_transform(X)
        summary = f"UMAP / N={len(X)} samples"

    image_data = plot_embedding(X2, labels)

    # UI 再生成
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    runs = sorted([p.name for p in (TRAIN_RESULT_ROOT / date).iterdir() if p.is_dir()])
    files = sorted([p.name for p in (TRAIN_RESULT_ROOT / date / run / "eval").glob("*.jsonl")])

    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=date,
        runs=runs,
        selected_run=run,
        files=files,
        image_data=image_data,
        summary=summary
    )


if __name__ == "__main__":
    app.run(debug=True)
