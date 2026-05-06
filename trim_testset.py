import csv

src = r"original_testset.csv"
dst = r"original_testset_text_only.csv"

with open(src, newline="", encoding="utf-8") as f_in, open(
    dst, "w", newline="", encoding="utf-8"
) as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.DictWriter(f_out, fieldnames=["text"])
    writer.writeheader()
    for row in reader:
        writer.writerow({"text": row["text"]})