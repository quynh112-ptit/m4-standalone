import requests, time, json, platform
import numpy as np
import pandas as pd

df = pd.read_csv("data/synthetic_ids.csv")
META = json.loads(open("artifacts/model_v1/metadata.json", encoding="utf-8").read())
FEATURES = META["feature_order"]

vectors = []
for i, row in df.head(200).iterrows():
    vectors.append({
        "flow_id": f"sim-{i+1:06d}",
        "feature_set_version": "zone05_flow_v1:1.0.0",
        "features": {k: float(row[k]) for k in FEATURES},
        "source": "simulator",
        "timestamp": "2026-09-30T13:30:16Z"
    })

times = []
n_success = 0
t_start = time.perf_counter()

for i, fv in enumerate(vectors[:200]):
    t0 = time.perf_counter()
    r = requests.post("http://127.0.0.1:8004/api/v1/predict", json=fv, timeout=2)
    times.append((time.perf_counter() - t0) * 1000)
    if r.status_code == 200:
        n_success += 1

t_measurement = time.perf_counter() - t_start
throughput = n_success / t_measurement

print("=== MỤC 18 (LAB 4.7): ĐO INFERENCE LATENCY VÀ THROUGHPUT ===")
print("n         =", len(times))
print("mean_ms   =", np.mean(times))
print("median_ms =", np.median(times))
print("p95_ms    =", np.percentile(times, 95))
print("Throughput=", throughput, "requests/s")
print("Info      : Python", platform.python_version(), "| Model:", META["model_version"])
