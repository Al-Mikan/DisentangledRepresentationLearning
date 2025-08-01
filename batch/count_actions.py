import pandas as pd

# ===== パス設定 =====
csv_path = "./label/animalkingdom/test/labels_test.csv"  # あなたのパスに合わせて！

# ===== CSV 読み込み =====
df = pd.read_csv(csv_path)

# ===== アクションごとの件数を集計 =====
action_counts = df["action"].value_counts().sort_index()

print("=== アクションごとの件数 ===")
print(action_counts)

# ===== CSVとして保存したい場合 =====
# out_path = "./action_counts.csv"
# action_counts.to_csv(out_path, header=["count"])
# print(f"\n✅ 保存しました: {out_path}")
