# 🛡️ ML-IDS: Hệ Thống Phát Hiện Xâm Nhập Độc Lập (Module 04)

Dự án này triển khai toàn bộ vòng đời của một hệ thống Machine Learning Phát hiện xâm nhập (ML-IDS) độc lập, sử dụng dữ liệu mạng mô phỏng (Synthetic Data) và dataset replay. Kiến trúc dự án được thiết kế tách biệt hoàn toàn phần lõi AI và phần tiếp nhận dữ liệu, giúp sẵn sàng tích hợp với luồng traffic thực tế từ thiết bị mạng (M3) mà không phải viết lại mã nguồn Inference API[cite: 13, 14].

> ** CẢNH BÁO KỸ THUẬT BẮT BUỘC:** 
> `SIMULATED DATA - NOT REAL NETWORK IDS PERFORMANCE`[cite: 13, 14]

## 📁 Cấu trúc Thư mục Dự án
```text
m4-standalone/
├── app/
│   ├── train.py          # Huấn luyện mô hình LR & RF, đóng gói artifact
│   ├── validator.py      # Bộ lọc kiểm tra lỗi đầu vào (Validator)
│   ├── main.py           # FastAPI Inference Service
│   └── evaluate.py       # Đánh giá độc lập trên tập Test
├── simulator/
│   ├── generate_dataset.py  # Sinh tập dataset mô phỏng (2000 dòng)
│   ├── eda_lab4_1.py        # Phân tích khám phá dữ liệu (EDA) & vẽ Histogram
│   ├── replay.py            # Replay FeatureVector thời gian gần thực
│   ├── test_faults_lab4_6.py # Kiểm thử 5 ca lỗi đầu vào (E1-E5)
│   └── benchmark_lab4_7.py   # Đo lường Latency & Throughput
├── config/
│   └── feature_set.yaml     # Khóa hợp đồng dữ liệu (zone05_flow_v1:1.0.0)
├── data/                    # Thư mục chứa synthetic_ids.csv, replay.jsonl
├── artifacts/               # Thư mục lưu mô hình (model.joblib, metadata.json)
├── results/                 # Lưu kết quả dự đoán, predictions.jsonl và biểu đồ
├── report.ipynb             # Báo cáo kết quả chi tiết dưới dạng Jupyter Notebook
└── requirements.txt         # Danh sách thư viện phụ thuộc
Hướng dẫn Cài đặt và Chạy Hệ thống
1. Cài đặt môi trường

python3 -m venv .venv
# Trên Windows PowerShell:
.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
# source .venv/bin/activate

pip install numpy pandas scikit-learn joblib fastapi uvicorn pydantic requests matplotlib ipykernel pyyaml
2. Sinh dữ liệu và Huấn luyện mô hình

python simulator/generate_dataset.py
python simulator/eda_lab4_1.py
python app/train.py
3. Khởi chạy Máy chủ Dự đoán (Inference Service)
Mở một cửa sổ Terminal, kích hoạt môi trường .venv và chạy[cite: 13, 14]:

uvicorn app.main:app --host 127.0.0.1 --port 8004
4. Chạy Replay và Kiểm thử Lỗi (Trên Terminal thứ hai)

python simulator/replay.py
python simulator/test_faults_lab4_6.py
python simulator/benchmark_lab4_7.py
python app/evaluate.py
