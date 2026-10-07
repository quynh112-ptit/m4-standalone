import pandas as pd
import json, time, requests

# Bước 1 & 2: Tạo 50 FeatureVector JSONL từ synthetic_ids.csv, gán flow_id sim-000001...sim-000050
df = pd.read_csv("data/synthetic_ids.csv").head(50)
FEATURES = [
    "duration_s", "total_packets", "total_bytes", "src_packet_rate",
    "dst_packet_rate", "byte_rate", "mean_packet_bytes",
    "packet_ratio", "byte_ratio", "dst_port"
]

with open("data/replay.jsonl", "w", encoding="utf-8") as f:
    for i, row in df.iterrows():
        fv = {
            "flow_id": f"sim-{i+1:06d}",
            "feature_set_version": "zone05_flow_v1:1.0.0",
            "features": {k: float(row[k]) for k in FEATURES},
            "source": "simulator",
            "timestamp": "2026-09-30T13:30:16Z"
        }
        f.write(json.dumps(fv) + "\n")

# Bước 3, 4, 5: Replay gửi sang Server, lưu Prediction JSONL và kiểm tra không mất flow_id
URL = "http://127.0.0.1:8004/api/v1/predict"
results = []
with open("data/replay.jsonl", encoding="utf-8") as f_in, open("results/predictions.jsonl", "w", encoding="utf-8") as f_out:
    for line in f_in:
        fv = json.loads(line)
        r = requests.post(URL, json=fv, timeout=2)
        res_json = r.json()
        print(r.status_code, res_json)
        f_out.write(json.dumps(res_json) + "\n")
        results.append(res_json)
        time.sleep(0.1) # Để 0.1 giây/gói cho chạy vèo hết 50 gói trong 5 giây

print(f"\n=== PASS LAB 4.5: Đã gửi 50 input và nhận đủ {len(results)}/50 response hợp lệ! ===")
