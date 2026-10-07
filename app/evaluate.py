import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)

df = pd.read_csv("data/synthetic_ids.csv")
FEATURES = [
    "duration_s", "total_packets", "total_bytes", "src_packet_rate",
    "dst_packet_rate", "byte_rate", "mean_packet_bytes",
    "packet_ratio", "byte_ratio", "dst_port"
]
X = df[FEATURES]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

lr = joblib.load("artifacts/model_v1/model.joblib")
pred = lr.predict(X_test)

print("=== MỤC 19 (LAB 4.8): ĐÁNH GIÁ TRÊN TEST SET CÓ NHÃN ===")
print("Data source: synthetic_ids.csv (SIMULATED DATA - NOT REAL NETWORK IDS PERFORMANCE)")
print(confusion_matrix(y_test, pred, labels=["NORMAL", "SIM_ATTACK"]))
print("Accuracy =", accuracy_score(y_test, pred))
print("Precision=", precision_score(y_test, pred, pos_label="SIM_ATTACK"))
print("Recall   =", recall_score(y_test, pred, pos_label="SIM_ATTACK"))
print("F1       =", f1_score(y_test, pred, pos_label="SIM_ATTACK"))
print("MCC      =", matthews_corrcoef(y_test, pred))
