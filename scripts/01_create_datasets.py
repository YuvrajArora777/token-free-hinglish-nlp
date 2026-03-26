"""
01_create_datasets.py
Loads and checks human_annotated Hinglish dataset for correctness.
Author: Yuvraj Arora
"""

import pandas as pd, os

def process_split(path: str, split_name: str):
    """
    Loads and analyzes a single dataset split.

    Args:
        path (str): The path to the data directory.
        split_name (str): The name of the split to process (e.g., "train").
    """
    file_path = f"{path}/{split_name}.tsv"
    if not os.path.exists(file_path):
        print(f"SKIPPING: Missing {split_name}.tsv in {path}")
        return

    try:
        df = pd.read_csv(file_path, sep="\t")
        print(f"--- Analyzing {split_name}.tsv ---")
        print(f"Loaded: {len(df)} rows")
        print("Columns:", list(df.columns))
        if "domain" in df.columns:
            print("Unique domains:", df["domain"].unique())
        print("-" * 50)
    except Exception as e:
        print(f"ERROR processing {split_name}.tsv: {e}")

if __name__ == "__main__":
    path = "data/human_annotated"
    splits = ["train", "validation", "test"]

    for s in splits:
        process_split(path, s)
