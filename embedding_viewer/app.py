from __future__ import annotations
import io
import json
import base64
from pathlib import Path
from flask import Flask, request, render_template_string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

app = Flask(__name__)

# ======================================================
# 設定：train_result の位置
# ======================================================
TRAIN_RESULT_ROOT = Path("/home/asel/Documents/labo/DisentangledRepresentationLearning/train_result")

# ======================================================
# HTML テンプレート
# ======================================================
PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Embedding Viewer</title>

  <script>
    function toggleParams() {
      const m = document.getElementById("method_select").value;
      document.getElementById("tsne_params").style.display = (m === "tsne") ? "block" : "none";
      document.getElementById("umap_params").style.display = (m === "umap") ? "block" : "none";
    }

    function toggleTitleBox() {
      const useTitle = document.getElementById("use_title").checked;
      document.getElementById("title_box").style.display = useTitle ? "block" : "none";
    }

    window.onload = function() {
      toggleParams();
      toggleTitleBox();
    };
  </script>
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

<!-- モデル選択 -->
{% if model_names %}
<form action="/plot" method="post">
  <input type="hidden" name="date" value="{{selected_date}}">
  <input type="hidden" name="run" value="{{selected_run}}">

  <label>Model</label><br>
  <select name="model">
    {% for m in model_names %}
      <option value="{{m}}" {% if m == selected_model %}selected{% endif %}>{{m}}</option>
    {% endfor %}
  </select>

  <br><br>

  <!-- タイトル ON/OFF -->
  <label>
    <input type="checkbox" id="use_title" name="use_title"
      {% if use_title %}checked{% endif %}
      onclick="toggleTitleBox()"
    >
    タイトルを付ける
  </label>

  <div id="title_box" style="margin-top:8px; display:none;">
    <input type="text" name="title" value="{{title}}" style="width:300px;"
      placeholder="例：Gated / epoch27">
  </div>

  <br><br>

  <!-- ラベル表示 ON/OFF -->
  <label>
    <input type="checkbox" name="show_labels"
      {% if show_labels %}checked{% endif %}>
    ラベルを表示する
  </label>

  <br><br>

  <!-- train/test/all -->
  <label>表示モード</label><br>
  <select name="view_mode">
    <option value="all" {% if view_mode=='all' %}selected{% endif %}>全体 (train+test)</option>
    <option value="train" {% if view_mode=='train' %}selected{% endif %}>train only</option>
    <option value="test" {% if view_mode=='test' %}selected{% endif %}>test only</option>
  </select>

  <br><br>

  <!-- Dim Reduction -->
  <label>Method</label><br>
  <select id="method_select" name="method" onchange="toggleParams()">
    <option value="tsne" {% if method=='tsne' %}selected{% endif %}>t-SNE</option>
    <option value="umap" {% if method=='umap' %}selected{% endif %}>UMAP</option>
  </select>

  <br><br>

  <!-- t-SNE -->
  <div id="tsne_params" style="display:none;">
    <label>Perplexity</label><br>
    <input type="number" name="perplexity" value="{{perplexity}}"><br><br>

    <label>Learning Rate</label><br>
    <input type="number" name="learning_rate" value="{{learning_rate}}"><br><br>

    <label>Early Exaggeration</label><br>
    <input type="number" name="early_exaggeration" value="{{early_exaggeration}}">
  </div>

  <!-- UMAP -->
  <div id="umap_params" style="display:none;">
    <label>n_neighbors</label><br>
    <input type="number" name="n_neighbors" value="{{n_neighbors}}"><br><br>

    <label>min_dist</label><br>
    <input type="number" step="0.01" name="min_dist" value="{{min_dist}}">
  </div>

  <br>
  <button type="submit">描画</button>
</form>
{% endif %}

{% if image_data %}
<hr>
<img src="data:image/png;base64,{{image_data}}" />
{% endif %}
</body>
</html>
"""

# ======================================================
# Embedding Loader
# ======================================================
def load_embeddings(path: Path):
    vecs, labels, sources = [], [], []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            vecs.append(o["vector"])
            labels.append(o["label"])
            sources.append(o.get("source", "unknown"))
    return np.array(vecs), np.array(labels), np.array(sources)

# ======================================================
# Plot
# ======================================================
def plot_embedding(X2d, labels, sources, title, show_labels):
    plt.figure(figsize=(8, 8))
    if title:
        plt.title(title)

    for lab in np.unique(labels):
        m = labels == lab
        for src, marker in [("train", "o"), ("test", "^")]:
            mask = m & (sources == src)
            if np.any(mask):
                plt.scatter(
                    X2d[mask, 0], X2d[mask, 1],
                    s=30, marker=marker,
                    label=f"{lab} ({src})" if show_labels else None
                )

    if show_labels:
        plt.legend(fontsize=8, bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.xticks([]); plt.yticks([])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# ======================================================
# Routes
# ======================================================
@app.route("/", methods=["GET"])
def index():
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=dates[-1] if dates else None,
        runs=[],
        model_names=[],
        show_labels=True
    )

@app.route("/plot", methods=["POST"])
def plot():
    date = request.form["date"]
    run = request.form["run"]
    model = request.form["model"]
    view_mode = request.form["view_mode"]
    show_labels = "show_labels" in request.form

    paths = []
    base = TRAIN_RESULT_ROOT / date / run / "eval"
    if view_mode in ("all", "train"):
        paths.append(base / f"{model}_train.jsonl")
    if view_mode in ("all", "test"):
        paths.append(base / f"{model}_test.jsonl")

    X, y, s = [], [], []
    for p in paths:
        vx, vy, vs = load_embeddings(p)
        X.append(vx); y.append(vy); s.append(vs)

    X = np.concatenate(X)
    y = np.concatenate(y)
    s = np.concatenate(s)

    reducer = TSNE(n_components=2)
    X2d = reducer.fit_transform(X)

    img = plot_embedding(X2d, y, s, "", show_labels)

    return render_template_string(
        PAGE,
        image_data=img,
        show_labels=show_labels
    )

if __name__ == "__main__":
    app.run(debug=True)
