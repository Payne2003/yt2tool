# 🎙️ YT2Dataset — Tự Động Xây Dựng Dataset TTS

**YT2Dataset** chuyển đổi video YouTube (hoặc audio có sẵn) thành dataset TTS chất lượng cao (định dạng LJSpeech) chỉ với một lệnh.

---

## ✨ Tính Năng

| Tính năng | Mô tả |
|-----------|-------|
| 🔗 YouTube → Dataset | Dán link, tool tự tải audio + subtitle |
| 📁 File có sẵn | Hỗ trợ WAV/MP3 trong thư mục `input/` |
| 🎤 Whisper transcribe | Tự động nhận diện giọng nói với word-level timestamps |
| ✂️ Auto-split | Đoạn quá dài (> max_duration) tự cắt đôi, không mất data |
| 🤖 GPT correction | (Tùy chọn) Dùng OpenAI sửa lỗi phiên âm Whisper |
| 🌐 Đa ngôn ngữ | Tiếng Việt, Anh, Nhật, Hàn, Filipino |
| 📊 LJSpeech format | `segment_name\|text\|normalized` sẵn cho training |

---

## 🚀 Bắt Đầu Nhanh

### 1. Cài đặt

```bash
# Cài các thư viện cần thiết
pip install yt-dlp openai-whisper pydub pyyaml openai

# ffmpeg phải có trong PATH (tải tại https://ffmpeg.org)
```

### 2. Cấu hình API Key (tùy chọn)

Mở file `config.yaml`, thêm OpenAI key nếu muốn sửa lỗi tự động:

```yaml
openai:
  api_key: "sk-..."   # bỏ trống nếu không dùng
```

### 3. Chạy tool

**Cách 1 — Double-click** (dễ nhất):
```
run_auto.bat
```

**Cách 2 — Command line**:
```bash
python auto_app.py
```

---

## 📖 Hướng Dẫn Sử Dụng

### Chế độ 1: Dán link YouTube

Khi tool hỏi, nhập link (nhiều link cách nhau bằng dấu phẩy hoặc mỗi link một dòng):

```
🔗 Nhập URL YouTube (Enter để xử lý file có sẵn):
https://www.youtube.com/watch?v=xxxxx
```

Tool sẽ tự động:
1. **Tải audio** (`.wav`) và subtitle (`.txt`) → lưu vào `input/`
2. **Transcribe** bằng Whisper (với word timestamps)
3. **Phân đoạn** thành các clip 2–13 giây
4. **Sửa lỗi** bằng GPT (nếu có API key)
5. **Xuất dataset** vào `output/`

### Chế độ 2: Xử lý file có sẵn

Đặt file audio (`.wav`/`.mp3`) vào thư mục `input/` → nhấn Enter (bỏ trống URL).

```
input/
├── ten_video.wav
└── ten_video.txt   ← subtitle reference (tùy chọn, dùng để sửa lỗi)
```

---

## 📂 Cấu Trúc Thư Mục

```
yt2dataset/
├── input/                  ← Audio & subtitle đầu vào
├── output/
│   ├── wavs/               ← Các file WAV đã cắt (~2–13 giây)
│   ├── metadata.csv        ← Dataset chính (LJSpeech format)
│   ├── original_texts/     ← Text gốc từ Whisper (mỗi segment 1 file)
│   └── corrected_texts/    ← Text đã sửa bởi GPT
├── config.yaml             ← Cấu hình toàn bộ tool
├── run_auto.bat            ← Chạy nhanh trên Windows
└── logs/yt2dataset.log     ← Log chi tiết
```

---

## ⚙️ Cấu Hình `config.yaml`

```yaml
download:
  quality: "bestaudio"      # chất lượng audio tải về
  subtitle_lang: "vi"       # ngôn ngữ subtitle ưu tiên (vi/en/ja/ko/tl)

whisper:
  model: "medium"           # tiny | base | small | medium | large | large-v3
  language: "auto"          # auto | vi | en | ja | ko | tl

segmentation:
  min_duration: 2.0         # giây — đoạn ngắn nhất (nên: 1.5–2.0)
  max_duration: 13.0        # giây — đoạn dài nhất (nên: 12.0–14.0)

dataset:
  output_dir: "output"      # thư mục output
  append_mode: true         # true = thêm vào dataset có sẵn
  sample_rate: 22050        # Hz (chuẩn TTS)

openai:
  api_key: ""               # OpenAI API key (để trống nếu không dùng)
```

---

## 📊 Định Dạng Output

File `output/metadata.csv` theo chuẩn **LJSpeech**:
```
segment_name|text gốc|text đã normalize
```

Ví dụ:
```
video_segment_00001_4.23s|khủng long xuất hiện từ rất lâu trước đây|khung long xuat hien tu rat lau truoc day
video_segment_00002_5.10s|con người bắt đầu nghiên cứu hóa thạch|con nguoi bat dau nghien cuu hoa thach
```

---

## 💡 Mẹo Sử Dụng

| Tình huống | Giải pháp |
|-----------|-----------|
| Video có subtitle (cc) | Đặt file `.txt` cùng tên với WAV trong `input/` → GPT sửa lỗi chính xác hơn |
| Muốn dataset nhanh (không cần GPT) | Bỏ trống `api_key` trong config |
| Video dài (> 1 tiếng) | Dùng model Whisper `large` để chính xác hơn |
| Muốn thêm vào dataset cũ | Để `append_mode: true` (mặc định) |
| Muốn tạo mới hoàn toàn | Đổi `append_mode: false` |
| Nhiều ngôn ngữ | Đổi `language: "en"` hoặc `"ja"` trong config |

---

## 🔄 Luồng Hoạt Động

```
Link YT ──► Tải audio+sub ──► input/
                                 │
File WAV ────────────────────────┘
                                 │
                          Whisper Transcribe
                          (word timestamps)
                                 │
                          Phân đoạn 2–13s
                          (auto-split nếu dài)
                                 │
                     [Có API key?] ──► GPT correction
                          │               │
                          └───────────────┘
                          Fallback raw nếu GPT từ chối
                                 │
                          Normalize text
                                 │
                         output/metadata.csv
                         output/wavs/*.wav
```

---

## 🧪 Kiểm Tra

Chạy test suite (không cần internet, không cần Whisper thật):

```bash
python -X utf8 tests/test_pipeline_mock.py
```

Kết quả mong đợi: **42 tests passed**

---

## 🐛 Xử Lý Lỗi Thường Gặp

| Lỗi | Nguyên nhân | Giải pháp |
|-----|------------|-----------|
| `c10.dll initialization failed` | PyTorch/CUDA conflict trong venv | Chạy bằng `run_auto.bat` (dùng system Python) |
| `No module named 'whisper'` | Chưa cài OpenAI Whisper | `pip install openai-whisper` |
| `ffmpeg not found` | ffmpeg chưa trong PATH | Tải ffmpeg và thêm vào PATH |
| `0 segments created` | Audio quá ngắn hoặc lỗi Whisper | Kiểm tra log tại `logs/yt2dataset.log` |
| GPT correction 0% | API key sai hoặc hết quota | Kiểm tra key, tool vẫn dùng raw Whisper |

---

## 📌 Yêu Cầu Hệ Thống

- Python 3.9+
- ffmpeg (trong PATH)
- RAM: ≥ 8GB (Whisper medium)
- Tùy chọn: GPU CUDA để Whisper chạy nhanh hơn
"# yt2tool" 
