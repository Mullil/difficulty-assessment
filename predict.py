import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

MODEL_DIR = "./best_model"

valid_df = pd.read_csv("valid.csv", usecols=["text"], dtype={"text": "string"})

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

dataset = Dataset.from_pandas(valid_df.reset_index(drop=True), preserve_index=False)
dataset = dataset.map(
    lambda batch: tokenizer(batch["text"], truncation=True),
    batched=True,
)
dataset = dataset.remove_columns(["text"])

args = TrainingArguments(
    output_dir="./tmp_predict",
    per_device_eval_batch_size=8,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer))
output = trainer.predict(dataset)
predictions = np.squeeze(output.predictions)

pd.DataFrame({"prediction": predictions}).to_csv("predictions.csv", index=False)
print("Saved predictions.csv")
