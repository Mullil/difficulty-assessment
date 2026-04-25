import csv
import os
from transformers import pipeline
from tqdm import tqdm

# CEFR label mapping: floats to CEFR strings
# 1.0 -> A1, 1.5 -> A1 (towards edge), 2.0 -> A2, 2.5 -> A2 (towards edge),
# 3.0 -> B1, 3.5 -> B1, 4.0 -> B2, 4.5 -> B2,
# 5.0 -> C1, 5.5 -> C2 (towards edge), 6.0 -> C2
FLOAT_TO_CEFR = {
    1.0: "A1",
    1.5: "A1",  # mapped towards lower edge
    2.0: "A2",
    2.5: "A2",  # mapped towards lower edge (closer to A-band)
    3.0: "B1",
    3.5: "B2",  # mapped towards lower edge
    4.0: "B2",
    4.5: "C1",  # mapped towards lower edge
    5.0: "C1",
    5.5: "C2",  # mapped towards upper edge
    6.0: "C2",
}


def map_label_to_cefr(label_value) -> str:
    """Map a numeric label (possibly intermediate) to a CEFR level string.

    Intermediate levels are mapped towards the edges:
      1.5 -> A1, 5.5 -> C2.
    Other intermediate levels (2.5, 3.5, 4.5) are mapped to the lower
    neighbour so that they cluster towards the closer band edge.
    """
    label = float(label_value)
    if label in FLOAT_TO_CEFR:
        return FLOAT_TO_CEFR[label]
    raise ValueError(f"Unexpected label value: {label_value}")


# Build the generator once
print("Loading model...")
generator = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
)


def paraphrase_with_cefr(text: str, cefr_level: str) -> str:
    cefr_descriptions = {
        "A1": "beginner level — very simple vocabulary, short sentences, basic present tense, concrete topics only",
        "A2": "elementary level — simple vocabulary, short sentences, basic tenses, everyday topics",
        "B1": "intermediate level — common vocabulary, moderate sentence complexity, main tenses, familiar topics",
        "B2": "upper-intermediate level — varied vocabulary, complex sentences, range of tenses, abstract topics",
        "C1": "advanced level — sophisticated vocabulary, complex syntax, nuanced expression, idiomatic language",
        "C2": "mastery level — extensive vocabulary, highly complex syntax, nominalisations, domain-specific terminology, dense subordination",
    }
    level_upper = cefr_level.upper()
    if level_upper not in cefr_descriptions:
        raise ValueError(
            f"Invalid CEFR level '{cefr_level}'. Must be one of: {', '.join(cefr_descriptions)}"
        )
    description = cefr_descriptions[level_upper]
    prompt = f"""<s>[INST] You are an expert linguist specialising in CEFR language proficiency levels.
Your task is to paraphrase the following text while strictly preserving its CEFR difficulty level: {level_upper} ({description}).

Rules:
- Reword sentences and rephrase expressions — do not copy the original phrasing
- Maintain the same language as the input text
- Preserve the meaning and all key information
- Match the vocabulary range, sentence complexity, and grammatical structures typical of {level_upper}
- Do not simplify or elevate the difficulty — stay exactly at {level_upper}
- Output only the paraphrased text, no commentary

Text to paraphrase:
{text} [/INST]"""

    output = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        return_full_text=True,
    )
    return output[0]["generated_text"].split("[/INST]")[-1].strip()


def augment_csv(
    input_path: str = "train.csv",
    output_path: str = "train_llm_augmented.csv",
    resume: bool = True,
) -> None:
    """Read train.csv, paraphrase each row with the LLM, and write augmented CSV.

    The output keeps the original numeric label (mapped to integer band edge)
    so that intermediate values such as 1.5 become 1 and 5.5 become 6.
    """
    # CEFR string -> integer label (1..6) for the output file
    cefr_to_int = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

    # Resume support: skip rows already written
    already_done = set()
    write_header = True
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8", newline="") as f_done:
            reader = csv.reader(f_done)
            try:
                header = next(reader)
                write_header = False
                for row in reader:
                    if row:
                        already_done.add(row[0])
            except StopIteration:
                pass
        print(f"Resuming: {len(already_done)} rows already in {output_path}")

    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    mode = "a" if (resume and not write_header) else "w"
    with open(output_path, mode, encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        if write_header:
            writer.writerow(["#", "text", "label"])

        for row in tqdm(rows, desc="Paraphrasing"):
            row_id = row["#"]
            if row_id in already_done:
                continue

            text = row["text"]
            label_raw = row["label"]
            try:
                cefr = map_label_to_cefr(label_raw)
            except ValueError as e:
                print(f"Skipping row {row_id}: {e}")
                continue

            int_label = float(cefr_to_int[cefr])

            try:
                paraphrased = paraphrase_with_cefr(text, cefr)
            except Exception as e:
                print(f"Error paraphrasing row {row_id}: {e}")
                continue

            writer.writerow([row_id, paraphrased, int_label])
            f_out.flush()  # flush after each row so progress is preserved


if __name__ == "__main__":
    augment_csv()