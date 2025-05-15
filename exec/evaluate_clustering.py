import json, pandas as pd, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

# --- データ読み込み ---
with open("exec/video_vectors.json") as f:
    vecs = json.load(f)
df = pd.read_csv("labels.csv")
df["video_path"] = df["video_path"].str.replace("\\", "/")
df = df[df["video_path"].apply(lambda p: p in vecs)]

# --- ラベルエンコード ---
le = LabelEncoder().fit(df["action"])
df["act_id"] = le.transform(df["action"])
X = np.array([vecs[p] for p in df["video_path"]])
y_true = df["act_id"].values

# --- KMeansクラスタリング ---
kmeans = KMeans(n_clusters=len(le.classes_), init="k-means++", random_state=0)
y_pred = kmeans.fit_predict(X)

# --- NMIスコア計算 ---
nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"✅ NMI Score: {nmi:.4f}")
