import pandas as pd
import ast
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.rcParams['font.family'] = 'Meiryo'

# CSV読み込み
df = pd.read_csv("AR_metadata.csv")

# mammal フィルター
def has_mammal(label_str):
    try:
        parent_list = ast.literal_eval(label_str)
        return any(cls.lower() == "mammal" for cls in parent_list)
    except:
        return False

df = df[df["list_animal_parent_class"].apply(has_mammal)]

# ✅ 動画本数の表示
print(f"✅ 対象動画数（mammal）: {len(df)} 本\n")

# ラベル数カウント
def count_labels(label_str):
    try:
        return len(ast.literal_eval(label_str))
    except:
        return 0

df["num_labels"] = df["list_animal_action"].apply(count_labels)

# 統計表示
print("mammal の動画におけるラベル数の統計:")
print(df["num_labels"].describe())

# ヒストグラム表示
plt.figure(figsize=(8, 5))
plt.hist(df["num_labels"], bins=range(1, df["num_labels"].max() + 2), color='skyblue', edgecolor='black', align='left')
plt.xlabel("1動画あたりのラベル数（species-action）")
plt.ylabel("動画数")
plt.xticks(range(1, df["num_labels"].max() + 1))
plt.title("🎬 mammal 動画のアノテーション数の分布")
plt.grid(True)
plt.tight_layout()
plt.savefig("./analyze/label_count_histogram_mammal.png")
# plt.show()

# -----------------------------
# ラベルが1つだけの mammal 動画を抽出
# -----------------------------
one_label_df = df[df["num_labels"] == 1].copy()

# 解析しやすくするために list_animal_action を展開
def extract_action(label_str):
    try:
        return ast.literal_eval(label_str)[0][1]  # 1つだけなので[0]で取り出す
    except:
        return "Invalid"

def extract_species(label_str):
    try:
        return ast.literal_eval(label_str)[0]  # 種名だけ
    except:
        return "Invalid"

# 行動・種をそれぞれ抽出
one_label_df["single_action"] = one_label_df["list_animal_action"].apply(extract_action)
one_label_df["single_species"] = one_label_df["list_animal"].apply(lambda x: ast.literal_eval(x)[0] if pd.notnull(x) else "Invalid")

# -----------------------------
# 1. 行動（action）の出現数グラフ
# -----------------------------
action_counts = Counter(one_label_df["single_action"])

plt.figure(figsize=(10, 5))
plt.bar(action_counts.keys(), action_counts.values(), color='orange')
plt.xticks(rotation=45, ha='right')
plt.xlabel("行動（action）")
plt.ylabel("動画数")
plt.title("🐾 ラベルが1つだけの mammal 動画に含まれる行動の数")
plt.tight_layout()
plt.savefig("./analyze/single_label_action_counts.png")
# plt.show()

# -----------------------------
# 2. 種（species）の出現数グラフ
# -----------------------------
species_counts = Counter(one_label_df["single_species"])
print("種の出現数:", species_counts)

lion_df = one_label_df[one_label_df["single_species"] == "Wolf"]
print(lion_df["single_action"].value_counts())

plt.figure(figsize=(10, 5))
plt.bar(species_counts.keys(), species_counts.values(), color='green')
plt.xticks(rotation=45, ha='right')
plt.xlabel("種（species）")
plt.ylabel("動画数")
plt.title("🦣 ラベルが1つだけの mammal 動画に含まれる種の数")
plt.tight_layout()
plt.savefig("./analyze/single_label_species_counts.png")
# plt.show()