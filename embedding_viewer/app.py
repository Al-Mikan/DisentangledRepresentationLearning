from __future__ import annotations
import io
import json
from pathlib import Path
from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

app = Flask(__name__)

# ======================================================
# 設定：train_result の位置をここだけ変えれば良い
# ======================================================
TRAIN_RESULT_ROOT = Path("/home/asel/Documents/labo/DisentangledRepresentationLearning/train_result")

# ======================================================
# HTML TEMPLATE
# ======================================================
PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Embedding Viewer</title>
</head>
<body style="font-family:sans-serif; margin:24px;">
  <h1>Embedding Viewer</h1>

  <!-- 日付 -->
  <form action="/" method="get">
    <label>日付フォルダ</label><br>
    <select name="date" onchange="this.form.submit()">
      {% for d in dates %}
        <option value="{{d}}" {% if d == selected_date %}selected{% endif %}>{{d}}</option>
      {% endfor %}
    </select>
  </form>
  <br>

  <!-- Run -->
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

  <!-- モデル名 -->
  {% if model_names %}
  <form action="/plot" method="post">
    <input type="hidden" name="date" value="{{selected_date}}">
    <input type="hidden" name="run" value="{{selected_run}}">

    <label>Model</label><br>
    <select name="model">
      {% for m in model_names %}
        <option value="{{m}}">{{m}}</option>
      {% endfor %}
    </select>

    <br><br>

    <!-- train/test/all -->
    <label>表示モード</label><br>
    <select name="view_mode">
      <option value="all">全体 (train+test)</option>
      <option value="train">train only</option>
      <option value="test">test only</option>
    </select>

    <br><br>

    <!-- Dim Reduction -->
    <label>Method</label>
    <select name="method">
      <option value="tsne">t-SNE</option>
      <option value="umap">UMAP</option>
    </select>

    <h3>t-SNE Parameters</h3>
    <label>Perplexity</label>
    <input type="number" name="perplexity" value="30">

    <label>Learning Rate</label>
    <input type="number" name="learning_rate" value="200">

    <label>Early Exaggeration</label>
    <input type="number" name="early_exaggeration" value="12">

    <label>Init</label>
    <select name="init">
      <option value="pca">pca</option>
      <option value="random">random</option>
    </select>

    <label>Angle (0.0–1.0)</label>
    <input type="number" step="0.01" name="angle" value="0.5">

    <h3>UMAP Parameters</h3>
    <label>n_neighbors</label>
    <input type="number" name="n_neighbors" value="15">

    <label>min_dist</label>
    <input type="number" step="0.01" name="min_dist" value="0.1">

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


# ======================================================
# Embedding Loader
# ======================================================
def load_embeddings(path: Path):
    vecs, labels, sources = [], [], []
    with open(path, "r") as f:
        for line in f:
            o = json.loads(line)
            vecs.append(o["vector"])
            labels.append(o["label"])
            sources.append(o.get("source", "unknown"))
    return (
        np.array(vecs, np.float32),
        np.array(labels),
        np.array(sources)
    )


# ======================================================
# Plot Helper
# ======================================================
def plot_embedding(X2d, labels):
    plt.figure(figsize=(8, 6))
    uniq = np.unique(labels)
    for lab in uniq:
        m = labels == lab
        plt.scatter(X2d[m, 0], X2d[m, 1], s=12, alpha=0.8, label=str(lab))
    plt.legend(fontsize=7)
    plt.xticks([]); plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()

    import base64
    return base64.b64encode(buf.getvalue()).decode()


# ======================================================
# Main UI
# ======================================================
@app.route("/", methods=["GET"])
def index():
    # Step 1: 日付フォルダ
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    selected_date = request.args.get("date", dates[-1] if dates else None)

    # Step 2: run フォルダ
    runs, selected_run = [], None
    if selected_date:
        date_dir = TRAIN_RESULT_ROOT / selected_date
        runs = sorted([p.name for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
        selected_run = request.args.get("run", runs[0] if runs else None)

    # Step 3: モデル名を抽出（train/test 共通部分）
    model_names = []
    if selected_date and selected_run:
        eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        if eval_dir.exists():
            jsonl_files = list(eval_dir.glob("*.jsonl"))
            names = set()
            for p in jsonl_files:
                stem = p.stem
                if stem.endswith("_train"):
                    names.add(stem[:-6])
                elif stem.endswith("_test"):
                    names.add(stem[:-5])
            model_names = sorted(list(names))

    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=selected_date,
        runs=runs,
        selected_run=selected_run,
        model_names=model_names,
        image_data=None,
        summary=None
    )


# ======================================================
# Plot Route
# ======================================================
@app.route("/plot", methods=["POST"])
def plot():
    date = request.form.get("date")
    run = request.form.get("run")
    model = request.form.get("model")
    view_mode = request.form.get("view_mode")

    eval_dir = TRAIN_RESULT_ROOT / date / run / "eval"

    # 必要に応じて読み込むファイルを決める
    paths = []
    if view_mode in ("train", "all"):
        paths.append(eval_dir / f"{model}_train.jsonl")
    if view_mode in ("test", "all"):
        paths.append(eval_dir / f"{model}_test.jsonl")

    vecs_all, labels_all = [], []

    for p in paths:
        X, labels, sources = load_embeddings(p)
        vecs_all.append(X)
        labels_all.append(labels)

    X = np.concatenate(vecs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)

    # t-SNE or UMAP
    method = request.form.get("method")

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=int(request.form.get("perplexity")),
            learning_rate=float(request.form.get("learning_rate")),
            early_exaggeration=float(request.form.get("early_exaggeration")),
            init=request.form.get("init"),
            angle=float(request.form.get("angle")),
        )
        X2d = reducer.fit_transform(X)
        summary = f"t-SNE / samples={len(X)}"

    else:
        reducer = umap.UMAP(
            n_neighbors=int(request.form.get("n_neighbors")),
            min_dist=float(request.form.get("min_dist")),
            metric="cosine"
        )
        X2d = reducer.fit_transform(X)
        summary = f"UMAP / samples={len(X)}"

    image_data = plot_embedding(X2d, labels)

    # 再表示に必要なデータ
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    runs = sorted([p.name for p in (TRAIN_RESULT_ROOT / date).iterdir() if p.is_dir()])
    model_names = sorted([
        p.stem.replace("_train", "").replace("_test", "")
        for p in (TRAIN_RESULT_ROOT / date / run / "eval").glob("*.jsonl")
    ])

    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=date,
        runs=runs,
        selected_run=run,
        model_names=model_names,
        image_data=image_data,
        summary=summary
    )


if __name__ == "__main__":
    app.run(debug=True)
