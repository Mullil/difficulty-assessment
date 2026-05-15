"""Strip the raw test CSV down to a single ``text`` column so it can be fed
directly into the prediction script."""

import csv

src = r"raw_testset.csv"
dst = r"preprocessed_testset.csv"

with open(src, newline="", encoding="utf-8") as f_in, open(
    dst, "w", newline="", encoding="utf-8"
) as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=["text"])
    writer.writeheader()
    for row in reader:
        writer.writerow({"text": row["text"]})