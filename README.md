# Lab 16 — Reflexion Agent Scaffold

Repo này cung cấp một khung sườn (scaffold) để xây dựng và đánh giá **Reflexion Agent**.

## 1. Mục tiêu của Repo
- Repo hiện tại đang sử dụng **Mock Data** (`mock_runtime.py`) để giả lập phản hồi từ LLM.
- Mục đích giúp học viên hiểu rõ về **flow**, các bước **loop**, cách thức hoạt động của cơ chế phản chiếu (reflection) và cách đánh giá (evaluation) mà không tốn chi phí API ban đầu.

## 2. Nhiệm vụ của Học viên
Học viên cần thực hiện các bước sau để hoàn thành bài lab:
1. **Xây dựng Agent thật**: Thay thế phần mock bằng việc gọi LLM thật (sử dụng Local LLM như Ollama, vLLM hoặc các Simple LLM API như OpenAI, Gemini).
2. **Chạy Benchmark thực tế**: Chạy đánh giá trên ít nhất **100 mẫu dữ liệu thật** từ bộ dataset **HotpotQA**.
3. **Định dạng báo cáo**: Kết quả chạy phải đảm bảo xuất ra file report (`report.json` và `report.md`) có cùng định dạng (format) với code gốc để có thể chạy được công cụ chấm điểm tự động.
4. **Tính toán Token thực tế**: Thay vì dùng số ước tính, học viên phải cài đặt logic tính toán lượng token tiêu thụ thực tế từ phản hồi của API.

## 3. Cách chạy Lab (Scaffold)
```bash
# Cài đặt môi trường
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Chạy benchmark (với mock data)
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run

# Chạy chấm điểm tự động
python autograde.py --report-path outputs/sample_run/report.json
```

## 4. Tiêu chí chấm điểm (Rubric)
- **80% số điểm (80 điểm)**: Hoàn thiện đúng và đủ luồng (flow) cho Reflexion Agent, chạy thành công với LLM thật và dataset thật.
- **20% số điểm (20 điểm)**: Thực hiện thêm ít nhất một trong các phần **Bonus** được nhắc đến trong mã nguồn (ví dụ: `structured_evaluator`, `reflection_memory`, `adaptive_max_attempts`, `memory_compression`, v.v. - xem chi tiết tại `autograde.py`).

## Thành phần mã nguồn
- `src/reflexion_lab/schemas.py`: Định nghĩa các kiểu dữ liệu trace, record.
- `src/reflexion_lab/prompts.py`: Nơi chứa các template prompt cho Actor, Evaluator và Reflector.
- `src/reflexion_lab/mock_runtime.py`: (Cần thay thế) Logic giả lập phản hồi LLM.
- `src/reflexion_lab/agents.py`: Cấu trúc chính của ReAct và Reflexion Agent.
- `src/reflexion_lab/reporting.py`: Logic xuất báo cáo benchmark.
- `run_benchmark.py`: Script chính để chạy đánh giá.
- `autograde.py`: Công cụ hỗ trợ chấm điểm nhanh dựa trên report.
- `src/reflexion_lab/llm_runtime.py`: Runtime thật dùng Ollama (OllamaClient + actor/evaluator/reflector).
- `scripts/prepare_hotpotqa.py`: Script tải và chuyển đổi HotpotQA dev-distractor.

---

## 5. How to Run REAL Benchmark (Ollama + HotpotQA)

### Bước 1 — Cài đặt & khởi động Ollama

```bash
# Cài Ollama (macOS / Linux)
# https://ollama.com/download

# Khởi động Ollama server (để chạy ngầm)
ollama serve &

# Tải model Llama 3.1 8B (~4.7GB, chạy 1 lần)
ollama pull llama3.1:8b
```

### Bước 2 — Tải & chuẩn bị HotpotQA dataset

```bash
# Tải hotpot_dev_distractor_v1.json (~47MB) và chọn 120 mẫu (seed=42)
python scripts/prepare_hotpotqa.py --n 120 --seed 42 --out data/hotpot_real_120.json
```

### Bước 3 — Chạy benchmark thật

```bash
# Chạy 100 mẫu với cả ReAct và Reflexion (llama3.1:8b)
python run_benchmark.py \
  --dataset data/hotpot_real_120.json \
  --out-dir outputs/real_run \
  --mode real \
  --model llama3.1:8b \
  --limit 100

# Hoặc mock mode để test nhanh (không cần Ollama)
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock
```

### Bước 4 — Chấm điểm tự động

```bash
python autograde.py --report-path outputs/real_run/report.json
# Expected: Auto-grade total: >= 95/100
```

---

## 6. Troubleshoot Ollama

| Vấn đề | Giải pháp |
|---|---|
| `connection refused` port 11434 | Chạy `ollama serve` trước khi benchmark |
| `model not found` | Chạy `ollama pull llama3.1:8b` để tải model |
| OOM / crash trên 16GB RAM | Thử `ollama pull llama3.2:3b` (nhỏ hơn) và thêm `--model llama3.2:3b` |
| Timeout quá 180s | Tăng `timeout` trong `OllamaClient.__init__` lên 300 |
| Port conflict | `lsof -i :11434` để kiểm tra, hoặc `OLLAMA_HOST=0.0.0.0:11435 ollama serve` rồi cập nhật `host` trong `OllamaClient` |
