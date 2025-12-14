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
# 設定：train_result の位置をここだけ変えれば良い
# ======================================================
TRAIN_RESULT_ROOT = Path("/home/asel/Documents/labo/DisentangledRepresentationLearning/train_result")

# ======================================================
# HTML TEMPLATE（パラメータ保持 & 注釈付き）
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

  <!-- タイトル -->
  <label>画像タイトル（PNGファイル名にも使用）</label><br>
  <input type="text" name="title" value="{{title}}" style="width:300px;" placeholder="例：Gated Normal / epoch27">
  <small style="color:gray;">※ 空欄ならタイトルなし & embedding.png</small>
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
  <label>Method</label>
  <select name="method">
    <option value="tsne" {% if method=='tsne' %}selected{% endif %}>t-SNE</option>
    <option value="umap" {% if method=='umap' %}selected{% endif %}>UMAP</option>
  </select>

  <h3>t-SNE Parameters</h3>

  <label>Perplexity</label><br>
  <input type="number" name="perplexity" value="{{perplexity}}">
  <small style="color:gray;">※ クラスタの局所密度推定（5～50）</small><br><br>

  <label>Learning Rate</label><br>
  <input type="number" name="learning_rate" value="{{learning_rate}}">
  <small style="color:gray;">※ 大きいと暴れる・小さいと収束しにくい</small><br><br>

  <label>Early Exaggeration</label><br>
  <input type="number" name="early_exaggeration" value="{{early_exaggeration}}">
  <small style="color:gray;">※ 初期配置の分離度</small><br><br>

  <label>Init</label>
  <select name="init">
    <option value="pca" {% if init=='pca' %}selected{% endif %}>pca</option>
    <option value="random" {% if init=='random' %}selected{% endif %}>random</option>
  </select>
  <small style="color:gray;">※ pcaの方が安定</small><br><br>

  <label>Angle (0.0–1.0)</label><br>
  <input type="number" step="0.01" name="angle" value="{{angle}}">
  <small style="color:gray;">※ 0 に近いほど精度↑ / 1 に近いほど高速</small><br><br>

  <h3>UMAP Parameters</h3>
  <label>n_neighbors</label><br>
  <input type="number" name="n_neighbors" value="{{n_neighbors}}">
  <small style="color:gray;">※ 局所 vs 大域のバランス (小=細かい)</small><br><br>

  <label>min_dist</label><br>
  <input type="number" step="0.01" name="min_dist" value="{{min_dist}}">
  <small style="color:gray;">※ クラスタの密集度（0〜0.5）</small><br><br>

  <br>
  <button type="submit">描画</button>
</form>
{% endif %}

{% if image_data %}
<hr>
<h2>結果</h2>

<img src="data:image/png;base64,{{ image_data }}" />

<br><br>

<a download="{{png_filename}}" href="data:image/png;base64,{{image_data}}">
  <button>🖼 PNG をダウンロード</button>
</a>

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
# Plot Helper（train/test を marker で区別）
# ======================================================
def plot_embedding(X2d, labels, sources, title):
    plt.figure(figsize=(8,6))

    if title:
        plt.title(title, fontsize=16)

    uniq_labels = np.unique(labels)

    for lab in uniq_labels:
        mask = (labels == lab)
        mask_train = mask & (sources == "train")
        mask_test = mask & (sources == "test")

        if np.any(mask_train):
            plt.scatter(X2d[mask_train,0], X2d[mask_train,1],
                        marker="o", s=20, alpha=0.8,
                        label=f"{lab} (train)")
        if np.any(mask_test):
            plt.scatter(X2d[mask_test,0], X2d[mask_test,1],
                        marker="^", s=35, alpha=0.9,
                        label=f"{lab} (test)")

    plt.legend(fontsize=8)
    plt.xticks([]); plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()

    return base64.b64encode(buf.getvalue()).decode()

# ======================================================
# Main UI
# ======================================================
@app.route("/", methods=["GET"])
def index():
    # 日付
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    selected_date = request.args.get("date", dates[-1] if dates else None)

    # run
    runs = []
    selected_run = None
    if selected_date:
        date_dir = TRAIN_RESULT_ROOT / selected_date
        runs = sorted([p.name for p in date_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
        selected_run = request.args.get("run", runs[0] if runs else None)

    # モデル名
    model_names = []
    if selected_date and selected_run:
        eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        if eval_dir.exists():
            names = set()
            for p in eval_dir.glob("*.jsonl"):
                stem = p.stem
                if stem.endswith("_train"):
                    names.add(stem[:-6])
                elif stem.endswith("_test"):
                    names.add(stem[:-5])
            model_names = sorted(names)

    return render_template_string(
        PAGE,
        dates=dates,
        selected_date=selected_date,
        runs=runs,
        selected_run=selected_run,
        model_names=model_names,

        # 初期値（描画後に保持される）
        selected_model=None,
        view_mode="all",
        method="tsne",
        perplexity=30,
        learning_rate=200,
        early_exaggeration=12,
        init="pca",
        angle=0.5,
        n_neighbors=15,
        min_dist=0.1,
        title="",
        image_data=None,
        summary=None,
        png_filename="embedding.png"
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
    title = request.form.get("title")

    method = request.form.get("method")
    perplexity = int(request.form.get("perplexity"))
    learning_rate = float(request.form.get("learning_rate"))
    early_exaggeration = float(request.form.get("early_exaggeration"))
    init = request.form.get("init")
    angle = float(request.form.get("angle"))
    n_neighbors = int(request.form.get("n_neighbors"))
    min_dist = float(request.form.get("min_dist"))

    eval_dir = TRAIN_RESULT_ROOT / date / run / "eval"

    # train / test jsonl
    paths = []
    if view_mode in ("train", "all"):
        paths.append(eval_dir / f"{model}_train.jsonl")
    if view_mode in ("test", "all"):
        paths.append(eval_dir / f"{model}_test.jsonl")

    vecs_all, labels_all, sources_all = [], [], []
    for p in paths:
        X, labels, sources = load_embeddings(p)
        vecs_all.append(X)
        labels_all.append(labels)
        sources_all.append(sources)

    X = np.concatenate(vecs_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    sources = np.concatenate(sources_all, axis=0)

    # =======================================
    # t-SNE / UMAP 実行
    # =======================================
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            init=init,
            angle=angle,
        )
        X2d = reducer.fit_transform(X)
        summary = f"t-SNE (samples={len(X)})"

    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
        )
        X2d = reducer.fit_transform(X)
        summary = f"UMAP (samples={len(X)})"

    # =======================================
    # 描画 + base64
    # =======================================
    image_data = plot_embedding(X2d, labels, sources, title)

    # 保存ファイル名
    png_filename = (title + ".png") if title else "embedding.png"

    # UI 再描画のための情報を復元
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

        selected_model=model,
        view_mode=view_mode,
        method=method,
        perplexity=perplexity,
        learning_rate=learning_rate,
        early_exaggeration=early_exaggeration,
        init=init,
        angle=angle,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        title=title,
        image_data=image_data,
        summary=summary,
        png_filename=png_filename
    )


if __name__ == "__main__":
    app.run(debug=True)
