import pandas as pd

SOURCE = "universal_cefr.csv"
OUTPUT = "train.csv"

# Set the number of instances to add per label.
# Use None to add all instances of that label
LABEL_COUNTS = {
    1.0: None,
    1.5: None,
    2.0: None,
    2.5: None,
    3.0: 200,
    3.5: None,
    4.0: 500,
    5.0: None,
    6.0: None,
}

df = pd.read_csv(SOURCE)

frames = []
for label, count in LABEL_COUNTS.items():
    subset = df[df["label"] == label]
    if len(subset) == 0:
        print(f"Warning: label {label} not found in {SOURCE}, skipping.")
        continue
    if count is None or count >= len(subset):
        sampled = subset
    else:
        sampled = subset.sample(n=count, random_state=42)
    frames.append(sampled)
    print(f"Label {label}: adding {len(sampled)} instances")

new_rows = pd.concat(frames).reset_index(drop=True)


existing = pd.read_csv(OUTPUT)
combined = pd.concat([existing, new_rows], ignore_index=True)
print(f"\nAppended to existing {OUTPUT} ({len(existing)} -> {len(combined)} rows)")

combined.to_csv(OUTPUT, index=False)
