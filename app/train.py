import pandas as pd
import joblib, json, hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, f1_score

# Mục 8: Bước 4 - Chia train/test đúng quy trình
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

# Mục 9: Lab 4.2 - Logistic Regression
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print("=== MỤC 9 (LAB 4.2): KẾT QUẢ LOGISTIC REGRESSION ===")
print(confusion_matrix(y_test, pred, labels=["NORMAL", "SIM_ATTACK"]))
print(classification_report(y_test, pred, digits=4))

# Mục 10: Lab 4.3 - Random Forest và so sánh có kiểm soát
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("=== MỤC 10 (LAB 4.3): SO SÁNH LR VÀ RF ===")
print("LR F1 =", f1_score(y_test, pred, pos_label="SIM_ATTACK"))
print("RF F1 =", f1_score(y_test, pred_rf, pos_label="SIM_ATTACK"))
print("LR MCC=", matthews_corrcoef(y_test, pred))
print("RF MCC=", matthews_corrcoef(y_test, pred_rf))

# Mục 11: Bước 5 - Lưu model artifact
p = Path("artifacts/model_v1")
p.mkdir(parents=True, exist_ok=True)
joblib.dump(lr, p / "model.joblib")
meta = {
    "model_id": "ZONE05-STANDALONE-LR",
    "model_version": "1.0.0-SIM",
    "feature_set_version": "zone05_flow_v1:1.0.0",
    "feature_order": FEATURES,
    "classes": list(lr.classes_),
    "data_source": "synthetic_ids.csv",
    "warning": "SIMULATED DATA - NOT REAL NETWORK IDS PERFORMANCE"
}
(p / "metadata.json").write_text(json.dumps(meta, indent=2))

print("\n=== MỤC 11: MÃ SHA256 CỦA MODEL ARTIFACT ===")
for f in ["model.joblib", "metadata.json"]:
    b = (p / f).read_bytes()
    print(f, hashlib.sha256(b).hexdigest())

# Mục 12: Lab 4.4 - Kiểm tra lưu/nạp model
loaded = joblib.load("artifacts/model_v1/model.joblib")
p1 = lr.predict(X_test.iloc[:20])
p2 = loaded.predict(X_test.iloc[:20])
assert (p1 == p2).all()
print("\n=== MỤC 12 (LAB 4.4) ===")
print("PASS: predictions identical")
