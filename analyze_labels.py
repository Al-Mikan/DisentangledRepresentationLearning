import pandas as pd
import ast
import matplotlib.pyplot as plt
from collections import Counter

# CSVの読み込み
df = pd.read_csv("labels.csv")

# ------------------------------
# 1. 親クラスの出現数
# ------------------------------
parent_counts = Counter(df["parent_class"].str.strip().str.lower())

print("📊 親クラスの出現数：")
for cls, count in parent_counts.items():
    print(f"  {cls}: {count}")

# ------------------------------
# 2. 行動ペア（(species, action)）の出現数
# ------------------------------
all_actions = [f"{species}: {action}" for species, action in zip(df["species"], df["action"])]
action_counts = Counter(all_actions)

print("\n🎬 行動ペアの出現数 上位10件：")
for item, count in action_counts.most_common(10):
    print(f"  {item}: {count}")

# ------------------------------
# 3. 特定の親クラス（例：'insect'）ごとの行動分布
# ------------------------------
target_parent_class = "mammal"  # 例として 'mammal' を指定

filtered_rows = df[df["parent_class"].str.strip().str.lower() == target_parent_class]
target_actions = filtered_rows["action"]
action_dist = Counter(target_actions)

print(f"\n🧬 {target_parent_class} の行動分布：")
for action, count in action_dist.most_common():
    print(f"  {action}: {count}")


# ------------------------------
# 4. mammal に含まれる動物の種類（species）
# ------------------------------
species_dist = Counter(filtered_rows["species"])

print(f"\n🦣 {target_parent_class} に含まれる動物の種類（species）:")
for species, count in species_dist.most_common():
    print(f"  {species}: {count}")
print(f"✅ {target_parent_class} の種類数: {len(species_dist)} 種類")