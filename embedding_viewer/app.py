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
# JSONL loader
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
# ラベル集合のロード（train / test 別）
# ======================================================
def load_label_set(eval_dir: Path, split: str):
    labels = set()
    for p in eval_dir.glob(f"*_{split}.jsonl"):
        with open(p, "r") as f:
            for line in f:
                labels.add(json.loads(line)["label"])
    return sorted(labels)

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

    return (
        adjusted_rand_score(labels, pred),
        normalized_mutual_info_score(labels, pred),
    )

# ======================================================
# Plot（同一ラベル同色、train/testはmarker）
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
                color=color, marker="o", s=20, alpha=0.8,
                label=(f"{lab} (train)" if show_labels else None),
            )
        if np.any(m_test):
            plt.scatter(
                X2d[m_test, 0], X2d[m_test, 1],
                c=[color],                 # 塗りつぶしはラベル色
                marker="^",
                s=35,
                alpha=0.9,
                edgecolors="lightgray",    # 枠線を薄いグレーに
                linewidths=1.2,            # 枠線の太さ
                label=(f"{lab} (test)" if show_labels else None),
            )


    if show_labels:
        plt.legend(fontsize=8, bbox_to_anchor=(1.02, 0.5), loc="center left")

    plt.xticks([]); plt.yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

# ======================================================
# /
# ======================================================
@app.route("/", methods=["GET"])
def index():
    dates = sorted(p.name for p in TRAIN_RESULT_ROOT.iterdir() if p.is_dir())

    selected_date = request.args.get("date")
    selected_run = request.args.get("run")

    runs = []
    if selected_date:
        date_dir = TRAIN_RESULT_ROOT / selected_date
        if date_dir.exists():
            runs = sorted(
                p.name for p in date_dir.iterdir()
                if p.is_dir() and p.name.startswith("run_")
            )

    return render_template(
        "index.html",
        dates=dates,
        runs=runs,
        selected_date=selected_date,
        selected_run=selected_run,

        model_names=[],
        train_labels=[],
        test_labels=[],
        selected_train_labels=[],
        selected_test_labels=[],

        # 以下は初期値
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
        summary=None,
        ari=None,
        nmi=None,
        png_filename="embedding.png",
    )


# ======================================================
# /read
# ======================================================
@app.route("/read", methods=["POST"])
def read_labels():
    date = request.form["date"]
    run = request.form["run"]

    eval_dir = TRAIN_RESULT_ROOT / date / run / "eval"

    train_labels = load_label_set(eval_dir, "train")
    test_labels = load_label_set(eval_dir, "test")

    model_names = sorted({
        p.stem.replace("_train", "").replace("_test", "")
        for p in eval_dir.glob("*.jsonl")
    })

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
        model_names=model_names,

        train_labels=train_labels,
        test_labels=test_labels,
        selected_train_labels=train_labels,
        selected_test_labels=test_labels,

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
        summary=None,
        ari=None,
        nmi=None,
        png_filename="embedding.png",
    )

# ======================================================
# /plot
# ======================================================
@app.route("/plot", methods=["POST"])
def plot():
    date = request.form["date"]
    run = request.form["run"]
    model = request.form["model"]

    selected_train = request.form.getlist("train_labels")
    selected_test = request.form.getlist("test_labels")

    view_mode = request.form.get("view_mode", "all")
    method = request.form.get("method", "tsne")
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

    mask = (
        ((sources == "train") & np.isin(labels, selected_train)) |
        ((sources == "test") & np.isin(labels, selected_test))
    )
    X, labels, sources = X[mask], labels[mask], sources[mask]

    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=int(request.form.get("perplexity", 30)),
            learning_rate=float(request.form.get("learning_rate", 200)),
            early_exaggeration=float(request.form.get("early_exaggeration", 12)),
            init=request.form.get("init", "pca"),
            angle=float(request.form.get("angle", 0.5)),
        )
        X2d = reducer.fit_transform(X)
        summary = f"t-SNE (samples={len(X)})"
    else:
        reducer = umap.UMAP(
            n_neighbors=int(request.form.get("n_neighbors", 15)),
            min_dist=float(request.form.get("min_dist", 0.1)),
            metric="cosine",
        )
        X2d = reducer.fit_transform(X)
        summary = f"UMAP (samples={len(X)})"

    ari, nmi = compute_ari_nmi(X, labels)
    image_data = plot_embedding(X2d, labels, sources, title, show_labels)

    train_labels = load_label_set(eval_dir, "train")
    test_labels = load_label_set(eval_dir, "test")

    model_names = sorted({
        p.stem.replace("_train", "").replace("_test", "")
        for p in eval_dir.glob("*.jsonl")
    })

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
        model_names=model_names,

        train_labels=train_labels,
        test_labels=test_labels,
        selected_train_labels=selected_train,
        selected_test_labels=selected_test,

        view_mode=view_mode,
        method=method,
        perplexity=request.form["perplexity"],
        learning_rate=request.form["learning_rate"],
        early_exaggeration=request.form["early_exaggeration"],
        init=request.form["init"],
        angle=request.form["angle"],
        n_neighbors=request.form["n_neighbors"],
        min_dist=request.form["min_dist"],

        use_title=use_title,
        title=title,
        show_labels=show_labels,

        image_data=image_data,
        summary=summary,
        ari=ari,
        nmi=nmi,
        png_filename=(title + ".png") if title else "embedding.png",
    )

if __name__ == "__main__":
    app.run(debug=True)
