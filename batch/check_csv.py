import pandas as pd
import sys
import argparse
import os

def analyze_csv(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found -> {csv_path}")
        return

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    for col in ["action", "species"]:
        if col in df.columns:
            unique_vals = df[col].unique()
            print(f"\n===== [ {col} ] Total types: {len(unique_vals)} =====")
            # value_counts() はデフォルトで降順ソートされるので、数が多い順に表示される
            print(df[col].value_counts().to_string())
        else:
            print(f"\n(Column '{col}' not found)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check action/species statistics in a CSV file.")
    parser.add_argument("csv_path", nargs="?", default="./label/train/animalkingdom.csv", help="Path to the CSV file")
    args = parser.parse_args()
    
    analyze_csv(args.csv_path)
