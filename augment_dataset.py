from pathlib import Path
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ["UniversalCEFR/cefr_asag_en", "UniversalCEFR/elg_cefr_en", "UniversalCEFR/readme_en", "UniversalCEFR/cambridge_exams_en", "UniversalCEFR/icle500_en"]
model_name = "Helsinki-NLP/opus-mt-en-fi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)
model.eval()

CEFR_TO_SCORE = {
    "A1": 1.0,
    "A1+": 1.5,
    "A2": 2.0,
    "A2+": 2.5,
    "B1": 3.0,
    "B1+": 3.5,
    "B2": 4.0,
    "B2+": 4.5,
    "C1": 5.0,
    "C1+": 5.5,
    "C2": 6.0,
}


def translate_data():
    data = [{"label": CEFR_TO_SCORE[sample["cefr_level"].strip().upper()], "source_text": sample["text"].strip()}
            for dataset in datasets for sample in load_dataset(dataset, split="train") if sample["cefr_level"].strip().upper() in CEFR_TO_SCORE]

    batch_size = 8
    max_length = 512
    translated_rows = []

    for batch_start in range(0, len(data), batch_size):
        batch_rows = data[batch_start: batch_start + batch_size]
        source_texts = [row["source_text"] for row in batch_rows]
        inputs = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generated = model.generate(**inputs, max_length=max_length)
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

        for row, translated_text in zip(batch_rows, translations):
            translated_text = translated_text.strip()
            translated_rows.append(
                {
                    "text": translated_text,
                    "label": row["label"],
                    }
            )
    return translated_rows

if __name__ == "__main__":
    translated = translate_data()
    df = pd.DataFrame(translated)

    output_path = Path("universal_cefr.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)