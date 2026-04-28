#!/usr/bin/env python3
"""
Combine a source CSV into a target CSV with controlled label distribution.

Rules applied to the source CSV:
  - Drop all rows with label == 3
  - Randomly sample 200 rows with label == 3.5 (or all if fewer than 200)
  - Keep all rows with any other label
  - Add 5000 to the '#' column of every kept source row
The (offset + filtered) source rows are then appended to the target CSV.

Edit the paths below, then run:  python combine_csv.py
"""

import sys
import pandas as pd

# ---- Configure these ----
SOURCE_PATH = "train_llm_augmented.csv"
TARGET_PATH = "train.csv"
OUTPUT_PATH = "train_all_combined.csv"
ID_OFFSET = 5000
DROP_LABEL = 3.0
SAMPLE_LABEL = 3.5
SAMPLE_N = 200
SEED = 42
# -------------------------


def combine(source_path: str, target_path: str, output_path: str, seed: int) -> None:
    src = pd.read_csv(source_path)
    tgt = pd.read_csv(target_path)

    expected_cols = ["#", "text", "label"]
    for df, name in [(src, "source"), (tgt, "target")]:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            sys.exit(f"Error: {name} CSV is missing columns: {missing}")

    # Coerce labels to numeric so 3 and 3.0 compare equally
    src["label"] = pd.to_numeric(src["label"], errors="raise")

    # Drop label == DROP_LABEL
    drop_mask = src["label"] == DROP_LABEL
    dropped = int(drop_mask.sum())
    src = src[~drop_mask]

    # Sample label == SAMPLE_LABEL
    is_sample = src["label"] == SAMPLE_LABEL
    pool = src[is_sample]
    rest = src[~is_sample]

    n_sample = min(SAMPLE_N, len(pool))
    sampled = pool.sample(n=n_sample, random_state=seed) if n_sample > 0 else pool

    kept = pd.concat([rest, sampled], ignore_index=True)

    # Offset the # column
    kept["#"] = kept["#"].astype(int) + ID_OFFSET

    # Append to target and save
    combined = pd.concat([tgt, kept], ignore_index=True)
    combined.to_csv(output_path, index=False)

    # Summary
    print(f"Source rows read:           {len(src) + dropped}")
    print(f"  dropped (label == {DROP_LABEL}):     {dropped}")
    print(f"  sampled (label == {SAMPLE_LABEL}):   {n_sample} (of {len(pool)} available)")
    print(f"  kept from other labels:   {len(rest)}")
    print(f"Source rows added:          {len(kept)}")
    print(f"Target rows (original):     {len(tgt)}")
    print(f"Output rows total:          {len(combined)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    combine(SOURCE_PATH, TARGET_PATH, OUTPUT_PATH, SEED)