import pandas as pd
import ast
import re

def fix_broken_list_animal_action(s):
    """
    文字列sが壊れていても ('Species', 'Action') の形式のタプルを全て抽出し、リスト化する
    """
    if pd.isna(s):
        return []

    # クォートを正規化
    s = s.replace('"', "'")

    # ('xxx', 'yyy') にマッチするタプルをすべて抽出
    matches = re.findall(r"\('.*?'\s*,\s*'.*?'\)", s)

    # matches は ["('Russell''s Viper', 'Keeping still')", ...] の形式
    # それを安全に ast.literal_eval してリストに変換
    result = []
    for m in matches:
        try:
            tup = ast.literal_eval(m)
            result.append(tup)
        except Exception as e:
            print(f"⚠️ パース失敗: {m} → {e}")
    return result

# CSV読み込み
df = pd.read_csv("AR_metadata.csv")

# 修正
df["list_animal_action_fixed"] = df["list_animal_action"].apply(fix_broken_list_animal_action)

# 保存（必要であれば）
df.to_csv("AR_metadata_fix.csv", index=False)