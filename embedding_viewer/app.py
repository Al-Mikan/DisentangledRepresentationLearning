from __future__ import annotations
import io
import json
import base64
from pathlib import Path

from flask import Flask, request, render_template
import numpy as np
import matplotlib.pyplot as plt
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

# ======================================================
# JSONL Loader
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
        np.asarray(vecs, np.float32),
        np.asarray(labels),
        np.asarray(sources),
    )

# ======================================================
# ARI / NMI
# ======================================================
def compute_ari_nmi(X, labels):
    n_clusters = len(np.unique(labels))
    if n_clusters <= 1:
        return None, None

    pred = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    ).fit_predict(X)

    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi

# ======================================================
# Plot（同一ラベルは同色、train/test は marker）
# ======================================================
def plot_embedding(X2d, labels, sources, title, show_labels):
    plt.figure(figsize=(8, 8))

    if title:
        plt.title(title, fontsize=16)

    uniq_labels = np.unique(labels)
    cmap = plt.get_cmap("tab20")
    color_map = {lab: cmap(i % 20) for i, lab in enumerate(uniq_labels)}

    for lab in uniq_labels:
        color = color_map[lab]
        mask = labels == lab
        m_train = mask & (sources == "train")
        m_test = mask & (sources == "test")

        if np.any(m_train):
            plt.scatter(
                X2d[m_train, 0], X2d[m_train, 1],
                color=color, marker="o",
                s=20, alpha=0.8,
                label=(f"{lab} (train)" if show_labels else None),
            )
        if np.any(m_test):
            plt.scatter(
                X2d[m_test, 0], X2d[m_test, 1],
                color=color, marker="^",
                s=35, alpha=0.9,
                label=(f"{lab} (test)" if show_labels else None),
            )

    if show_labels:
        plt.legend(
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
        )

    plt.xticks([])
    plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# ======================================================
# Index
# ======================================================
@app.route("/", methods=["GET"])
def index():
    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    selected_date = request.args.get("date", dates[-1] if dates else None)

    runs, selected_run = [], None
    if selected_date:
        ddir = TRAIN_RESULT_ROOT / selected_date
        runs = sorted(
            [p.name for p in ddir.iterdir()
             if p.is_dir() and p.name.startswith("run_")]
        )
        selected_run = request.args.get("run", runs[0] if runs else None)

    model_names = []
    if selected_date and selected_run:
        eval_dir = TRAIN_RESULT_ROOT / selected_date / selected_run / "eval"
        if eval_dir.exists():
            names = set()
            for p in eval_dir.glob("*.jsonl"):
                if p.stem.endswith("_train"):
                    names.add(p.stem[:-6])
                elif p.stem.endswith("_test"):
                    names.add(p.stem[:-5])
            model_names = sorted(names)

    return render_template(
        "index.html",
        dates=dates,
        runs=runs,
        model_names=model_names,
        selected_date=selected_date,
        selected_run=selected_run,
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
        use_title=False,
        title="",
        show_labels=True,
        image_data=None,
        all_labels=[],
        selected_labels=[],
        ari=None,
        nmi=None,
        png_filename="embedding.png",
    )

# ======================================================
# Plot
# ======================================================
@app.route("/plot", methods=["POST"])
def plot():
    date = request.form["date"]
    run = request.form["run"]
    model = request.form["model"]
    view_mode = request.form["view_mode"]
    method = request.form["method"]

    show_labels = "show_labels" in request.form
    use_title = "use_title" in request.form
    title = request.form.get("title", "") if use_title else ""

    eval_dir = TRAIN_RESULT_ROOT / date / run / "eval"

    paths = []
    if view_mode in ("all", "train"):
        paths.append(eval_dir / f"{model}_train.jsonl")
    if view_mode in ("all", "test"):
        paths.append(eval_dir / f"{model}_test.jsonl")

    vecs, labels, sources = [], [], []
    for p in paths:
        X, l, s = load_embeddings(p)
        vecs.append(X)
        labels.append(l)
        sources.append(s)

    X = np.concatenate(vecs)
    labels = np.concatenate(labels)
    sources = np.concatenate(sources)

    all_labels = sorted(np.unique(labels).tolist())
    selected_labels = request.form.getlist("selected_labels") or all_labels

    mask = np.isin(labels, selected_labels)
    X, labels, sources = X[mask], labels[mask], sources[mask]

    # Dim reduction
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=int(request.form["perplexity"]),
            learning_rate=float(request.form["learning_rate"]),
            early_exaggeration=float(request.form["early_exaggeration"]),
            init=request.form["init"],
            angle=float(request.form["angle"]),
        )
        X2d = reducer.fit_transform(X)
    else:
        reducer = umap.UMAP(
            n_neighbors=int(request.form["n_neighbors"]),
            min_dist=float(request.form["min_dist"]),
            metric="cosine",
        )
        X2d = reducer.fit_transform(X)

    ari, nmi = compute_ari_nmi(X, labels)
    image_data = plot_embedding(X2d, labels, sources, title, show_labels)
    png_filename = f"{title}.png" if title else "embedding.png"

    dates = sorted([p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir()])
    runs = sorted(
        [p.name for p in (TRAIN_RESULT_ROOT / date).iterdir()
         if p.is_dir() and p.name.startswith("run_")]
    )

    return render_template(
        "index.html",
        dates=dates,
        runs=runs,
        model_names=[model],
        selected_date=date,
        selected_run=run,
        selected_model=model,
        view_mode=view_mode,
        method=method,
        perplexity=request.form.get("perplexity"),
        learning_rate=request.form.get("learning_rate"),
        early_exaggeration=request.form.get("early_exaggeration"),
        init=request.form.get("init"),
        angle=request.form.get("angle"),
        n_neighbors=request.form.get("n_neighbors"),
        min_dist=request.form.get("min_dist"),
        use_title=use_title,
        title=title,
        show_labels=show_labels,
        image_data=image_data,
        all_labels=all_labels,
        selected_labels=selected_labels,
        ari=ari,
        nmi=nmi,
        png_filename=png_filename,
    )

if __name__ == "__main__":
    app.run(debug=True)
