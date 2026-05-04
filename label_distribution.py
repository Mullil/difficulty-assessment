import pandas as pd

df = pd.read_csv("train_all_combined.csv")
counts = df["label"].value_counts().sort_index()
total = len(df)

print(f"Total samples: {total}\n")
print(f"{'Label':<10} {'Count':<10} {'Percent'}")
print("-" * 30)
for label, count in counts.items():
    print(f"{label:<10} {count:<10} {count / total * 100:.1f}%")
