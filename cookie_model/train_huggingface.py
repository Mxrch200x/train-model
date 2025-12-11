import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "cookies_db"

TABLE = "cookies"
COLUMN_NAME = "name"
COLUMN_DOMAIN = "domain"
COLUMN_LABEL = "label"

LABEL2ID = {
    "Strictly Necessary Cookies": 0,
    "Performance Cookies": 1,
    "Functionality Cookies": 2,
    "Targeting or Advertising Cookies": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PRETRAINED = "distilbert-base-uncased"
SAVE_DIR = "./hf_cookie_model"


class CookieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len
        )
        enc = {k: torch.tensor(v) for k, v in enc.items()}
        enc["labels"] = torch.tensor(self.labels[idx])
        return enc

    def __len__(self):
        return len(self.texts)


def load_labeled_cookies():
    conn = None
    texts, labels = [], []
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT {COLUMN_NAME}, {COLUMN_DOMAIN}, {COLUMN_LABEL}
            FROM {TABLE}
            WHERE {COLUMN_LABEL} IS NOT NULL
        """)
        rows = cursor.fetchall()
        for row in rows:
            raw_label = row[COLUMN_LABEL]
            if raw_label not in LABEL2ID:
                continue
            text = f"{row[COLUMN_NAME]} | {row[COLUMN_DOMAIN]}"
            texts.append(text)
            labels.append(LABEL2ID[raw_label])
    except Error as e:
        print("MySQL error:", e)
    finally:
        if conn:
            conn.close()

    print(f"[INFO] Loaded {len(texts)} labeled cookies usable for training.")
    return texts, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


def main():
    texts, labels = load_labeled_cookies()
    if len(texts) == 0:
        print("[ERROR] No training samples found.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    train_ds = CookieDataset(X_train, y_train, tokenizer)
    val_ds = CookieDataset(X_val, y_val, tokenizer)

    args = TrainingArguments(
        output_dir=SAVE_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("[INFO] Start training...")
    trainer.train()
    print("[INFO] Evaluate...")
    trainer.evaluate()

    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"[DONE] HF model saved → {SAVE_DIR}")


if __name__ == "__main__":
    main()
