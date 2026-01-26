import pandas as pd
from pathlib import Path

train_path = "label/train/animalkingdom_split.csv"
test_path = "label/test/animalkingdom_split.csv"

if not Path(train_path).exists() or not Path(test_path).exists():
    print("CSV not found")
    exit()

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

sp_train = set(df_train["species"].unique())
sp_test = set(df_test["species"].unique())

print(f"Train species count: {len(sp_train)}")
print(f"Test species count: {len(sp_test)}")
print(f"Union count: {len(sp_train | sp_test)}")

only_train = sp_train - sp_test
only_test = sp_test - sp_train

print(f"Only in Train ({len(only_train)}): {only_train}")
print(f"Only in Test ({len(only_test)}): {only_test}")
print(f"Intersection ({len(sp_train & sp_test)}): {len(sp_train & sp_test)}")
