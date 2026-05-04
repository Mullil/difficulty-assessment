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

MODEL_NAME = "TurkuNLP/bert-base-finnish-cased-v1"
N_TRIALS = 20

valid_df = pd.read_csv("valid.csv", usecols=["text", "label"], dtype={"text": "string", "label": "float32"})


def make_model(hidden_dropout: float, attention_dropout: float):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=hidden_dropout,
        attention_probs_dropout_prob=attention_dropout,
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


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eval_ds = tokenize_split(valid_df, tokenizer)

    def objective(trial):
        dataset = trial.suggest_categorical("dataset", ["original_train.csv", "LLM_augmented.csv", "MT_augmented.csv", "train_all_combined.csv"])
        train_df = pd.read_csv(dataset, usecols=["text", "label"], dtype={"text": "string", "label": "float32"})
        train_ds = tokenize_split(train_df, tokenizer)
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        epochs = trial.suggest_int("num_train_epochs", 2, 4)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
        hidden_dropout = trial.suggest_float("hidden_dropout", 0.0, 0.3)
        attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.3)

        args = TrainingArguments(
            output_dir=f"trial_{trial.number}",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=lr,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=make_model(hidden_dropout, attention_dropout),
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        return metrics["eval_rmse"]

    study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study.db", load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    best_args = TrainingArguments(
        output_dir="final_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_params["learning_rate"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=best_params["num_train_epochs"],
        weight_decay=best_params["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    final_trainer = Trainer(
        model=make_model(best_params["hidden_dropout"], best_params["attention_dropout"]),
        args=best_args,
        train_dataset=tokenize_split(pd.read_csv(best_params["dataset"], usecols=["text", "label"], dtype={"text": "string", "label": "float32"}), tokenizer),
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    metrics = final_trainer.evaluate()

    os.makedirs("best_model", exist_ok=True)
    final_trainer.save_model("best_model")
    tokenizer.save_pretrained("best_model")

    with open(os.path.join("best_model", "metrics.json"), "w", encoding="utf-8") as file:
        json.dump({"best_hyperparameters": best_params, "metrics": metrics}, file)
