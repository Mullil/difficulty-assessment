import csv
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(input_path: str, train_path: str, test_path: str,
                     test_size: float = 0.2, seed: int = 42) -> None:
    random.seed(seed)

    # Group rows by label
    buckets: dict[str, list[dict]] = defaultdict(list)
    fieldnames: list[str] = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            buckets[row["label"]].append(row)

    train_rows: list[dict] = []
    test_rows: list[dict] = []

    print(f"\nLabel distribution in source file:")
    print(f"{'Label':>8}  {'Total':>6}  {'Train':>6}  {'Test':>6}")
    print("-" * 34)

    for label in sorted(buckets, key=float):
        rows = buckets[label]
        random.shuffle(rows)
        n_test = max(1, round(len(rows) * test_size))
        n_train = len(rows) - n_test
        test_rows.extend(rows[:n_test])
        train_rows.extend(rows[n_test:])
        print(f"{label:>8}  {len(rows):>6}  {n_train:>6}  {n_test:>6}")

    # Shuffle final sets so labels aren't grouped
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    def write_csv(path: str, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(train_path, train_rows)
    write_csv(test_path, test_rows)

    total = len(train_rows) + len(test_rows)
    print("-" * 34)
    print(f"{'TOTAL':>8}  {total:>6}  {len(train_rows):>6}  {len(test_rows):>6}")
    print(f"\nSaved {len(train_rows)} rows → {train_path}")
    print(f"Saved {len(test_rows)} rows  → {test_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stratified 80/20 train-test split for a labeled CSV."
    )
    parser.add_argument("input", help="Path to the input CSV file")
    parser.add_argument(
        "--train", default=None,
        help="Output path for train CSV (default: train.csv)"
    )
    parser.add_argument(
        "--test", default=None,
        help="Output path for test CSV (default: test.csv)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    train_path = args.train or str(input_path.with_name("train.csv"))
    test_path  = args.test  or str(input_path.with_name("test.csv"))

    stratified_split(
        input_path=str(input_path),
        train_path=train_path,
        test_path=test_path,
        test_size=args.test_size,
        seed=args.seed,
    )