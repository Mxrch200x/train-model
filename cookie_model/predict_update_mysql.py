# predict_update_mysql.py
# --------------------------
# ใช้ HF embedding + sklearn classifier
# ทำนาย label ของ cookies ที่ label IS NULL
# แล้วอัปเดตเป็นข้อความเต็ม เช่น "Performance Cookies"

import mysql.connector
from mysql.connector import Error
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel


MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "cookies_db"
TABLE = "cookies"

COLUMN_ID = "id"
COLUMN_NAME = "name"
COLUMN_DOMAIN = "domain"
COLUMN_LABEL = "label"
COLUMN_LABEL_SOURCE = "label_source"

HF_MODEL_DIR = "./hf_cookie_model"
SKLEARN_MODEL_PATH = "cookie_sklearn_clf.joblib"

BATCH_SIZE = 32
LIMIT = 1000  # ต่อรันจะทำกี่แถว


def fetch_unlabeled():
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT {COLUMN_ID}, {COLUMN_NAME}, {COLUMN_DOMAIN}
            FROM {TABLE}
            WHERE {COLUMN_LABEL} IS NULL
            LIMIT {LIMIT}
        """)
        rows = cursor.fetchall()
        conn.close()
        print(f"[INFO] Loaded {len(rows)} unlabeled cookies.")
        return rows
    except Error as e:
        print("MySQL error:", e)
        return []


def build_embeddings(texts, tokenizer, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    all_emb = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
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


def update_label(row_id, label_text):
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE {TABLE}
            SET {COLUMN_LABEL}=%s,
                {COLUMN_LABEL_SOURCE}='hf_sklearn'
            WHERE {COLUMN_ID}=%s
        """, (label_text, row_id))
        conn.commit()
        conn.close()
    except Error as e:
        print(f"[ERROR] update id={row_id}: {e}")


def main():
    print("[INFO] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_DIR)
    hf_model = AutoModel.from_pretrained(HF_MODEL_DIR)

    payload = joblib.load(SKLEARN_MODEL_PATH)
    clf = payload["clf"]
    ID2LABEL = payload["id2label"]

    rows = fetch_unlabeled()
    if not rows:
        print("[DONE] No unlabeled cookies.")
        return

    ids = []
    texts = []
    for r in rows:
        cid = r[COLUMN_ID]
        name = r[COLUMN_NAME] or ""
        domain = r[COLUMN_DOMAIN] or ""
        ids.append(cid)
        texts.append(f"{name} | {domain}")

    print("[INFO] Building embeddings...")
    X = build_embeddings(texts, tokenizer, hf_model)

    print("[INFO] Predicting...")
    pred_ids = clf.predict(X)

    for cid, pid in zip(ids, pred_ids):
        label_text = ID2LABEL[int(pid)]
        print(f"[UPDATE] id={cid} -> '{label_text}'")
        update_label(cid, label_text)

    print("Finished predicting & updating labels.")


if __name__ == "__main__":
    main()
