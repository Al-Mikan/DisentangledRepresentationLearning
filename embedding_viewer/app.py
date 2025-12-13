# app.py (Local File Loading Version)
from __future__ import annotations
import io
import json
import base64
from pathlib import Path

from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

app = Flask(__name__)

# --------------------------------------------------------
# HTML: ファイルパスを入力する方式（uploadではない）
# --------------------------------------------------------
PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <title>Embedding Viewer (Local Files)</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    input[type=text], select { width: 100%; padding: 6px; }
    .btn { padding: 10px; background:#333; color:#fff; border:0; cursor:pointer; }
    .card { margin-top: 20px; padding: 16px; border:1px solid #ddd; border-radius:8px; }
  </style>
</head>
<body>
  <h1>Embedding Viewer (Local JSONL / NPY / CSV)</h1>

  <form action="/plot" method="post">
    <label>埋め込みファイルパス (.jsonl / .npy / .csv)</label>
    <input type="text" name="embed_path" placeholder="/path/to/train.jsonl" required>

    <label>メソッド</label>
    <select name="method">
      <option value="tsne">t-SNE</option>
      <option value="umap">UMAP</option>
    </select>

    <label>n_components</label>
    <input type="number" name="n_components" value="2" min="2" max="3">

    <label>perplexity (t-SNE)</label>
    <input type="number" name="perplexity" value="30">

    <label>n_neighbors (UMAP)</label>
    <input type="number" name="n_neighbors" value="15">

    <label>min_dist (UMAP)</label>
    <input type="number" name="min_dist" value="0.1" step="0.01">

    <button class="btn" type="submit">描画</button>
  </form>

  {% if image_data %}
  <div class="card">
    <h3>結果</h3>
    <img src="data:image/png;base64,{{ image_data }}" />
    <pre>{{ summary }}</pre>
  </div>
  {% endif %}
</body>
</html>
"""

# --------------------------------------------------------
# JSONL / NPY / CSV Loader
# --------------------------------------------------------
def load_embeddings(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    name = path.name.lower()

    # JSONL (推奨): vector, label, source を自動抽出
    if name.endswith(".jsonl"):
        vecs = []
        labels = []
        sources = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                vecs.append(obj["vector"])
                labels.append(obj["label"])
                src = obj.get("source", "unknown")
                sources.append(src)

        return (
            np.array(vecs, dtype=np.float32),
            np.array(labels),
            np.array(sources)
        )

    # NPY
    if name.endswith(".npy"):
        arr = np.load(path)
        return arr.astype(np.float32), None, None

    # CSV
    if name.endswith(".csv"):
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number]).to_numpy()
        return numeric.astype(np.float32), None, None

    raise ValueError("Unsupported file type. Use JSONL / NPY / CSV")

# --------------------------------------------------------
# Plot helper
# --------------------------------------------------------
def plot_embedding(X2d, labels, sources):
    plt.figure(figsize=(8,6))

    if labels is None:
        plt.scatter(X2d[:,0], X2d[:,1], s=10, alpha=0.8)
    else:
        uniq = np.unique(labels)
        for lab in uniq:
            m = labels == lab
            plt.scatter(X2d[m,0], X2d[m,1], s=12, alpha=0.8, label=str(lab))

        plt.legend(fontsize=7)

    plt.xticks([]); plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()

    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------
# Routes
# --------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE, image_data=None, summary=None)


@app.route("/plot", methods=["POST"])
def plot():
    embed_path = Path(request.form.get("embed_path"))
    method = request.form.get("method")
    n_components = int(request.form.get("n_components"))

    perplexity = int(request.form.get("perplexity"))
    n_neighbors = int(request.form.get("n_neighbors"))
    min_dist = float(request.form.get("min_dist"))

    # 1) Load vectors
    X, labels, sources = load_embeddings(embed_path)

    # 2) Dim reduction
    if method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity)
        X2d = reducer.fit_transform(X)
        summary = f"t-SNE\nN={len(X)}, D={X.shape[1]}"
    else:
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric="cosine",
        )
        X2d = reducer.fit_transform(X)
        summary = f"UMAP\nN={len(X)}, D={X.shape[1]}"

    # 3) Plot
    image_data = plot_embedding(X2d, labels, sources)

    return render_template_string(
        PAGE,
        image_data=image_data,
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)
