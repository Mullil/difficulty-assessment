import json
import os

import numpy as np
import optuna
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

df = pd.read_csv("train.csv", usecols=["text", "labels"], dtype={"text": "string", "labels": "float32"})
train_df, valid_df = train_test_split(df, test_size=0.2, shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(
        'TurkuNLP/FinBERT',
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )

def tokenize_split(dataframe, tokenizer):
    dataset = Dataset.from_pandas(dataframe.reset_index(drop=True), preserve_index=False)
    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True),
        batched=True,
    )
    return dataset.remove_columns(["text"])


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.squeeze(predictions)
    labels = np.squeeze(labels)
    rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))
    mae = float(np.mean(np.abs(predictions - labels)))
    return {"rmse": rmse, "mae": mae}


def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 4),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [4, 8, 16]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('TurkuNLP/FinBERT')
    train_ds = tokenize_split(train_df, tokenizer)
    eval_ds = tokenize_split(valid_df, tokenizer)

    training_args = TrainingArguments(
        output_dir="output_args",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    search_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    best_run = search_trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=15,
        compute_objective=lambda metrics: metrics["eval_rmse"],
    )

    params = best_run.hyperparameters

    best_training_args = TrainingArguments(
        output_dir="final_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["per_device_train_batch_size"],
        per_device_eval_batch_size=params["per_device_eval_batch_size"],
        num_train_epochs=params["num_train_epochs"],
        weight_decay=params["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    final_trainer = Trainer(
        model=model,
        args=best_training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    metrics = final_trainer.evaluate()

    os.makedirs("best_model", exist_ok=True)
    final_trainer.save_model("best_model")
    tokenizer.save_pretrained("best_model")

    with open(os.path.join("best_model", "metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"best_hyperparameters": best_run.hyperparameters, "metrics": metrics}, file)
