import sys, os
sys.path.insert(0, os.path.abspath("."))

import json, requests
from app.validator import validate

URL = "http://127.0.0.1:8004/api/v1/predict"
META = json.loads(open("artifacts/model_v1/metadata.json", encoding="utf-8").read())

with open("data/replay.jsonl", encoding="utf-8") as f:
    valid_fv = json.loads(f.readline())

print("=== MỤC 17 (LAB 4.6): KIỂM THỬ LỖI ĐẦU VÀO (E1 -> E5) ===")

# E1: Sai feature_set_version -> Kỳ vọng: HTTP 422
e1 = json.loads(json.dumps(valid_fv))
e1["feature_set_version"] = "wrong_version:9.9.9"
r1 = requests.post(URL, json=e1, timeout=2)
print("E1 (Sai version)       -> Expected: HTTP 422 | Actual:", r1.status_code, r1.json())

# E2: Thiếu total_bytes -> Kỳ vọng: HTTP 422
e2 = json.loads(json.dumps(valid_fv))
del e2["features"]["total_bytes"]
r2 = requests.post(URL, json=e2, timeout=2)
print("E2 (Thiếu total_bytes) -> Expected: HTTP 422 | Actual:", r2.status_code, r2.json())

# E3: NaN/Inf qua unit test -> Kỳ vọng: Reject
e3 = json.loads(json.dumps(valid_fv))
e3["features"]["byte_rate"] = float("nan")
try:
    validate(e3, META)
except ValueError as err:
    print("E3 (NaN/Inf unit test) -> Expected: Reject   | Actual: Reject -", err)

# E4: dst_port > 65535 -> Kỳ vọng: Reject (HTTP 422)
e4 = json.loads(json.dumps(valid_fv))
e4["features"]["dst_port"] = 70000
r4 = requests.post(URL, json=e4, timeout=2)
print("E4 (dst_port > 65535)  -> Expected: Reject   | Actual: HTTP", r4.status_code, r4.json())

# E5: flow_id hợp lệ, vector đúng -> Kỳ vọng: HTTP 200
r5 = requests.post(URL, json=valid_fv, timeout=2)
print("E5 (Vector hợp lệ)     -> Expected: HTTP 200 | Actual:", r5.status_code, r5.json())
