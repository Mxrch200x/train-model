# train_sklearn_from_hf.py
# ---------------------------
# ใช้ HuggingFace (จาก hf_cookie_model) ทำ embedding
# แล้ว train sklearn LogisticRegression เป็น final classifier

import mysql.connector
from mysql.connector import Error
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "cookies_db"
TABLE = "cookies"

COLUMN_NAME = "name"
COLUMN_DOMAIN = "domain"
COLUMN_LABEL = "label"

# ใช้ HF model ที่ train แล้ว
HF_MODEL_DIR = "./hf_cookie_model"
SKLEARN_OUTPUT = "cookie_sklearn_clf.joblib"

# label แบบข้อความยาว
LABEL2ID = {
    "Strictly Necessary Cookies": 0,
    "Performance Cookies": 1,
    "Functionality Cookies": 2,
    "Targeting or Advertising Cookies": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_data():
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

        for r in cursor.fetchall():
            lab = r[COLUMN_LABEL]
            if lab not in LABEL2ID:
                # ข้าม unknown หรือ label แปลก ๆ
                continue
            texts.append(f"{r[COLUMN_NAME]} | {r[COLUMN_DOMAIN]}")
            labels.append(LABEL2ID[lab])

        conn.close()
    except Error as e:
        print("MySQL error:", e)

    print(f"[INFO] Loaded {len(texts)} samples for sklearn.")
    return texts, labels


def build_embeddings(texts, tokenizer, model, batch_size=32):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    all_emb = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            all_emb.append(cls.cpu().numpy())

    return np.vstack(all_emb)


def main():
    texts, labels = load_data()
    if not texts:
        print("[ERROR] No data found for sklearn training.")
        return

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    hf_model = AutoModel.from_pretrained(HF_MODEL_DIR)

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("[INFO] Building embeddings for train...")
    X_train = build_embeddings(X_train_t, tokenizer, hf_model)

    print("[INFO] Building embeddings for test...")
    X_test = build_embeddings(X_test_t, tokenizer, hf_model)

    clf = Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1))
    ])

    print("[INFO] Training sklearn classifier...")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    # ใช้เฉพาะ class ที่มีอยู่จริงใน y_test/preds
    unique_ids = sorted(set(list(y_test) + list(preds)))
    target_names = [ID2LABEL[i] for i in unique_ids]

    print("\n=== Classification report (only classes present in test) ===")
    print(
        classification_report(
            y_test,
            preds,
            labels=unique_ids,
            target_names=target_names,
        )
    )

    # เซฟโมเดล + mapping
    joblib.dump(
        {"clf": clf, "id2label": ID2LABEL},
        SKLEARN_OUTPUT
    )
    print(f"[DONE] Saved sklearn model → {SKLEARN_OUTPUT}")


if __name__ == "__main__":
    main()
