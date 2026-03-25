"""
auto_app.py — Pipeline TTS Dataset tự động hoàn toàn

Luồng:
  1. Dán YouTube URL vào ô input
  2. Click "▶ Bắt đầu"
  3. Tự động:
     - Tải audio + subtitle → ./input/<title>.wav + .txt
     - Whisper transcribe → cắt đoạn 2-12s
     - Normalize text
     - Ghi ./output/wavs/*.wav + ./output/metadata.csv
"""
from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Bootstrap sys.path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QSplitter,
    QTextEdit, QVBoxLayout, QWidget, QProgressBar,
)

from src.config_loader import load_config
from src.downloader import YouTubeDownloader
from src.transcriber import Transcriber
from src.dataset_builder import DatasetBuilder, DatasetEntry
from src.normalizer import TextNormalizer
from src.logger import setup_logger, get_logger

# ── Đường dẫn cố định ────────────────────────────────────────────────────────
INPUT_DIR  = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# ── API key mặc định ─────────────────────────────────────────────────────────
DEFAULT_API_KEY = ""

log = get_logger("auto_app")


# ─────────────────────────────────────────────────────────────────────────────
# Worker — chạy toàn bộ pipeline trong background
# ─────────────────────────────────────────────────────────────────────────────

class AutoPipelineWorker(QThread):
    log_sig      = pyqtSignal(str)
    progress_sig = pyqtSignal(int)          # 0–100
    done_sig     = pyqtSignal(int)          # tổng số segments
    error_sig    = pyqtSignal(str)

    def __init__(
        self,
        urls: list[str],
        local_files: list[str],
        language: str,
        whisper_model: str,
        openai_key: str,
        subtitle_lang: str,
        parallel_count: int = 1,   # 1 = lần lượt, 2+ = song song
        parent=None,
    ):
        super().__init__(parent)
        self.urls           = urls
        self.local_files    = local_files
        self.language       = language
        self.whisper_model  = whisper_model
        self.openai_key     = openai_key
        self.subtitle_lang  = subtitle_lang
        self.parallel_count = parallel_count
        self._stop          = False
        self._log_lock      = threading.Lock()   # serialise cb() from threads

    def stop(self):
        self._stop = True

    def run(self):
        cb = self.log_sig.emit
        try:
            INPUT_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            cfg = load_config()
            cfg["download"]["subtitle_lang"] = self.subtitle_lang
            cfg["whisper"]["model"]          = self.whisper_model
            if self.language != "auto":
                cfg["whisper"]["language"]   = self.language
            if self.openai_key:
                cfg["openai"]["api_key"]     = self.openai_key

            transcriber = Transcriber(cfg, progress_cb=cb)
            lang        = self.language if self.language != "auto" else None
            eff_lang    = lang or "vi"

            all_entries: list[DatasetEntry] = []

            def _process_source(wav: Path, subtitle_text: str, slug: str, label: str) -> None:
                """Core pipeline: transcribe → build → ghi ra output/<slug>/."""
                # Per-source output dir
                src_out = OUTPUT_DIR / slug
                src_out.mkdir(parents=True, exist_ok=True)

                # Per-source cfg so DatasetBuilder writes to output/<slug>/
                src_cfg = dict(cfg)
                src_cfg["dataset"] = dict(cfg.get("dataset", {}))
                src_cfg["dataset"]["output_dir"] = str(src_out)
                src_cfg["dataset"]["append_mode"] = False  # fresh per source

                builder = DatasetBuilder(src_cfg)

                cb(f"🎤 Transcribe bằng Whisper…")
                try:
                    segments = transcriber.transcribe(str(wav), language=lang)
                    cb(f"✅ {len(segments)} segments")
                except Exception as exc:
                    cb(f"❌ Whisper lỗi: {exc}")
                    return

                if not segments:
                    cb("⚠ Không có segment — bỏ qua.")
                    return

                cb("⚙ Cắt đoạn + normalize + ghi metadata…")
                entries = builder.build(
                    audio_path=str(wav),
                    segments=segments,
                    language=eff_lang,
                    prefix=slug,
                    subtitle_text=subtitle_text,
                    progress_cb=cb,
                )
                all_entries.extend(entries)
                cb(f"✅ {len(entries)} entries → output/{slug}/metadata.csv")
                cb(f"📁 Output: {src_out}")

            # ── Xử lý file local có sẵn trong input/ ────────────────────────────────
            if self.local_files:
                total_local = len(self.local_files)
                cb(f"\n📂 Xử lý {total_local} file có sẵn trong input/…")
                for i, wav_path in enumerate(self.local_files):
                    if self._stop:
                        cb("🛑 Đã dừng.")
                        break
                    wav = Path(wav_path)
                    slug = DatasetBuilder._slugify(wav.stem)

                    cb(f"\n{'─'*60}")
                    cb(f"📂 [{i+1}/{total_local}] {wav.name}")
                    cb(f"📁 Slug: {slug}")

                    # Move/copy wav into input/<slug>/ if not already there
                    src_input = INPUT_DIR / slug
                    src_input.mkdir(parents=True, exist_ok=True)
                    target_wav = src_input / wav.name
                    if wav.resolve() != target_wav.resolve():
                        import shutil
                        shutil.copy2(str(wav), str(target_wav))
                        cb(f"📦 Copied → input/{slug}/{wav.name}")
                        wav = target_wav

                    # Move .txt subtitle nếu có
                    subtitle_text = ""
                    orig_txt = Path(wav_path).with_suffix(".txt")
                    if orig_txt.exists():
                        target_txt = src_input / orig_txt.name
                        if orig_txt.resolve() != target_txt.resolve():
                            import shutil
                            shutil.copy2(str(orig_txt), str(target_txt))
                        subtitle_text = target_txt.read_text(encoding="utf-8", errors="replace")
                        cb(f"📄 Subtitle: input/{slug}/{orig_txt.name}")

                    _process_source(wav, subtitle_text, slug, label=wav.name)

                    if not self.urls:
                        self.progress_sig.emit(int((i + 1) / total_local * 100))

            # ── Xử lý URLs ─────────────────────────────────────────────────────────
            if self.urls:
                total      = len(self.urls)
                n_parallel = max(1, self.parallel_count)
                mode_txt   = f"{n_parallel} song song" if n_parallel > 1 else "lần lượt"
                cb(f"🔗 {total} URL | chế độ: {mode_txt}")

                def _download_and_process(url: str, idx: int) -> None:
                    """Tải xuống và process 1 URL (có thể chạy song song)."""
                    if self._stop:
                        return
                    with self._log_lock:
                        cb(f"\n{'─'*60}")
                        cb(f"🔗 [{idx}/{total}] {url}")

                    tmp_cfg = dict(cfg)
                    tmp_cfg["dataset"] = dict(cfg.get("dataset", {}))
                    tmp_cfg["dataset"]["output_dir"] = str(OUTPUT_DIR)
                    downloader = YouTubeDownloader(tmp_cfg)

                    cb_safe = lambda m: (self._log_lock.acquire(), cb(m), self._log_lock.release())

                    cb_safe(f"📥 Tải audio + subtitle…")
                    result = downloader.download_to_input_format(
                        url=url, dest_dir=INPUT_DIR,
                        speaker_name="", index=idx, progress_cb=cb_safe,
                    )
                    if not result.success:
                        cb_safe(f"❌ Tải thất bại: {result.error}")
                        return

                    slug = DatasetBuilder._slugify(result.title)
                    cb_safe(f"✅ Audio: {Path(result.audio_path).name} | Slug: {slug}")

                    import shutil
                    src_input = INPUT_DIR / slug
                    src_input.mkdir(parents=True, exist_ok=True)

                    wav_src    = Path(result.audio_path)
                    target_wav = src_input / wav_src.name
                    if wav_src.resolve() != target_wav.resolve():
                        shutil.move(str(wav_src), str(target_wav))
                    wav = target_wav

                    subtitle_text = ""
                    if result.subtitle_path:
                        sub_src    = Path(result.subtitle_path)
                        target_sub = src_input / sub_src.name
                        if sub_src.resolve() != target_sub.resolve():
                            shutil.move(str(sub_src), str(target_sub))
                        subtitle_text = target_sub.read_text(encoding="utf-8", errors="replace")

                    cb_safe(f"🎤 Transcribe + Build dataset…")
                    _process_source(wav, subtitle_text, slug, label=result.title)

                    done_count[0] += 1
                    self.progress_sig.emit(done_count[0] * 100 // total)

                done_count = [0]   # mutable counter shared across threads

                if n_parallel == 1:
                    # Tuần tự: chạy từng URL
                    for i, url in enumerate(self.urls):
                        if self._stop:
                            cb("🛑 Đã dừng.")
                            break
                        _download_and_process(url, i + 1)
                else:
                    # Song song: N worker threads
                    with ThreadPoolExecutor(max_workers=n_parallel) as ex:
                        futures = {
                            ex.submit(_download_and_process, url, i + 1): url
                            for i, url in enumerate(self.urls)
                        }
                        for fut in as_completed(futures):
                            if self._stop:
                                ex.shutdown(wait=False, cancel_futures=True)
                                cb("🛑 Đã dừng.")
                                break
                            try:
                                fut.result()
                            except Exception as exc:
                                cb(f"❌ Lỗi: {exc}")

            # ── Hoàn thành ──────────────────────────────────────────────────────
            cb(f"\n{'═'*60}")
            cb(f"🎉 HOÀN THÀNH! Tổng: {len(all_entries)} segments")
            cb(f"📁 Input  : {INPUT_DIR}")
            cb(f"📁 Output : {OUTPUT_DIR}")
            cb(f"   ↳ Mỗi link ở thư mục riêng: output/<tên_video>/")
            self.done_sig.emit(len(all_entries))

        except Exception as exc:
            import traceback
            self.error_sig.emit(f"❌ Lỗi nghiêm trọng:\n{exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

STYLE = """
QMainWindow, QWidget {
    background-color: #0f0f1a;
    color: #e0e0f0;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: bold;
    color: #7c9ef8;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QTextEdit, QLineEdit {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e0e0f0;
}
QTextEdit:focus, QLineEdit:focus { border-color: #7c9ef8; }
QComboBox {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    padding: 4px 10px;
    color: #e0e0f0;
    min-width: 120px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #1a1a2e;
    color: #e0e0f0;
    selection-background-color: #7c9ef8;
    selection-color: #0f0f1a;
}
QPushButton#start_btn {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #4f7cf7, stop:1 #9b59f5);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 40px;
    font-size: 15px;
    font-weight: bold;
    min-width: 220px;
}
QPushButton#start_btn:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #6b8ff8, stop:1 #b06af9);
}
QPushButton#start_btn:disabled {
    background: #2a2a4a;
    color: #555580;
}
QPushButton#stop_btn {
    background-color: #c0392b;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 30px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#stop_btn:hover { background-color: #e74c3c; }
QPushButton#stop_btn:disabled { background-color: #2a2a4a; color: #555580; }
QPushButton#clear_btn {
    background-color: #1a3a2a;
    color: #4ecca3;
    border: 1px solid #2a5a3a;
    border-radius: 6px;
    padding: 6px 18px;
}
QPushButton#clear_btn:hover { background-color: #1f4a34; }
QProgressBar {
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    background: #1a1a2e;
    text-align: center;
    color: #e0e0f0;
    height: 22px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #4f7cf7, stop:1 #9b59f5);
    border-radius: 5px;
}
QLabel#info { color: #555580; font-size: 11px; font-style: italic; }
QLabel#path_label {
    color: #4ecca3;
    font-size: 11px;
    padding: 2px 0;
}
QListWidget {
    background-color: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 6px;
    color: #e0e0f0;
    padding: 4px;
}
QListWidget::item { padding: 4px 8px; border-radius: 4px; }
QListWidget::item:selected {
    background-color: #3a3a6a;
    color: #a29bfe;
}
QListWidget::item:hover { background-color: #25254a; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

class AutoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🤖 Auto TTS Dataset — Dán link → Tự động xây dataset")
        self.setStyleSheet(STYLE)
        self._worker: AutoPipelineWorker | None = None
        self._build_ui()
        self.showMaximized()

    def _build_ui(self):
        root_w = QWidget()
        self.setCentralWidget(root_w)
        root = QVBoxLayout(root_w)
        root.setSpacing(12)
        root.setContentsMargins(24, 16, 24, 16)

        # ── Header ────────────────────────────────────────────────────────────
        header = QLabel("🤖  Auto TTS Dataset Builder")
        header.setFont(QFont("Segoe UI", 20, QFont.Bold))
        header.setStyleSheet("color: #7c9ef8; padding-bottom: 4px;")
        root.addWidget(header)

        sub = QLabel("Dán YouTube URL → Tự động tải về, transcribe, xuất dataset LJSpeech")
        sub.setObjectName("info")
        sub.setStyleSheet("color: #9090b0; font-size: 12px; padding-bottom: 8px;")
        root.addWidget(sub)

        # ── Paths info ────────────────────────────────────────────────────────
        path_row = QHBoxLayout()
        lbl_in  = QLabel(f"📁 Input:  {INPUT_DIR}")
        lbl_out = QLabel(f"📁 Output: {OUTPUT_DIR}")
        lbl_in.setObjectName("path_label")
        lbl_out.setObjectName("path_label")
        path_row.addWidget(lbl_in)
        path_row.addSpacing(30)
        path_row.addWidget(lbl_out)
        path_row.addStretch()
        root.addLayout(path_row)

        # ── URL Input ─────────────────────────────────────────────────────────
        url_grp = QGroupBox("🔗  YouTube URLs  (mỗi dòng 1 URL | hoặc import file .txt)")
        ul = QVBoxLayout(url_grp)

        # Toolbar: số URL + import + xóa
        ut = QHBoxLayout()
        self._url_count_lbl = QLabel("0 URL")
        self._url_count_lbl.setStyleSheet("color:#9090b0;font-size:11px;")
        import_btn = QPushButton("📂 Import .txt")
        import_btn.setFixedWidth(110)
        import_btn.setStyleSheet(
            "background:#1a2a1a;border:1px solid #2a5a2a;"
            "border-radius:5px;padding:3px 8px;color:#55d98d;"
        )
        import_btn.clicked.connect(self._import_url_file)
        clear_url_btn = QPushButton("🗑 Xóa tất cả")
        clear_url_btn.setFixedWidth(95)
        clear_url_btn.setStyleSheet(
            "background:#2a1a1a;border:1px solid #5a2a2a;"
            "border-radius:5px;padding:3px 8px;color:#ff7675;"
        )
        clear_url_btn.clicked.connect(lambda: self.url_edit.clear())
        ut.addWidget(self._url_count_lbl)
        ut.addStretch()
        ut.addWidget(import_btn)
        ut.addWidget(clear_url_btn)
        ul.addLayout(ut)

        self.url_edit = QTextEdit()
        self.url_edit.setPlaceholderText(
            "Dán nhiều URL, mỗi dòng 1 URL:\n"
            "https://www.youtube.com/watch?v=...\n"
            "https://youtu.be/..."
        )
        self.url_edit.setFont(QFont("Consolas", 12))
        self.url_edit.setMinimumHeight(110)
        self.url_edit.textChanged.connect(self._update_url_count)
        ul.addWidget(self.url_edit)

        # ── Local files trong input/ ──────────────────────────────────────
        local_grp = QGroupBox("📂  Hoặc chọn file có sẵn trong input/  (Ctrl+Click để chọn nhiều)")
        ll = QVBoxLayout(local_grp)

        # Toolbar: count label + refresh btn
        lt = QHBoxLayout()
        self.local_count_lbl = QLabel("0 file WAV")
        self.local_count_lbl.setStyleSheet("color:#9090b0; font-size:11px;")
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(95)
        refresh_btn.setStyleSheet(
            "background:#1a2a3a;border:1px solid #2a3a5a;"
            "border-radius:5px;padding:3px 8px;color:#74b9ff;"
        )
        refresh_btn.clicked.connect(self._refresh_local_files)
        select_all_btn = QPushButton("☑ Tất cả")
        select_all_btn.setFixedWidth(80)
        select_all_btn.setStyleSheet(
            "background:#1a2a3a;border:1px solid #2a3a5a;"
            "border-radius:5px;padding:3px 8px;color:#a29bfe;"
        )
        select_all_btn.clicked.connect(lambda: self.local_list.selectAll())
        desel_btn = QPushButton("☐ Bỏ chọn")
        desel_btn.setFixedWidth(85)
        desel_btn.setStyleSheet(
            "background:#1a2a3a;border:1px solid #2a3a5a;"
            "border-radius:5px;padding:3px 8px;color:#636e72;"
        )
        desel_btn.clicked.connect(lambda: self.local_list.clearSelection())
        lt.addWidget(self.local_count_lbl)
        lt.addStretch()
        lt.addWidget(select_all_btn)
        lt.addWidget(desel_btn)
        lt.addWidget(refresh_btn)
        ll.addLayout(lt)

        self.local_list = QListWidget()
        self.local_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.local_list.setFixedHeight(130)
        ll.addWidget(self.local_list)

        # Splitter để 2 group nằm cạnh nhau
        input_splitter = QSplitter(Qt.Horizontal)
        input_splitter.addWidget(url_grp)
        input_splitter.addWidget(local_grp)
        input_splitter.setSizes([500, 500])
        root.addWidget(input_splitter)

        # Load file lần đầu
        self._refresh_local_files()

        # ── Settings row ──────────────────────────────────────────────────────
        settings_grp = QGroupBox("⚙️  Cài đặt")
        sl = QHBoxLayout(settings_grp)
        sl.setSpacing(20)

        # Language
        sl.addWidget(QLabel("Ngôn ngữ:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "auto", "vi", "en", "ja", "ko", "tl"
        ])
        self.lang_combo.setCurrentText("vi")
        sl.addWidget(self.lang_combo)

        # Subtitle lang
        sl.addSpacing(10)
        sl.addWidget(QLabel("Subtitle:"))
        self.sub_lang_combo = QComboBox()
        self.sub_lang_combo.addItems(["vi", "en", "ja", "ko", "tl"])
        sl.addWidget(self.sub_lang_combo)

        # Chế độ chạy
        sl.addSpacing(10)
        sl.addWidget(QLabel("🔀 Chế độ:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["🟢 Lần lượt", "⚡ Song song (2)", "⚡⚡ Song song (3)"])
        self.mode_combo.setToolTip(
            "Lần lượt: an toàn, dễ debug\n"
            "Song song 2: download 2 link cùng lúc (nhanh hơn)\n"
            "Song song 3: 3 link cùng lúc (nhanh nhất, tốn RAM)"
        )
        sl.addWidget(self.mode_combo)

        # Whisper model
        sl.addSpacing(10)
        sl.addWidget(QLabel("Whisper:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "large-v3"])
        self.model_combo.setCurrentText("medium")
        sl.addWidget(self.model_combo)

        # OpenAI Key
        sl.addSpacing(10)
        sl.addWidget(QLabel("API Key:"))
        self.api_edit = QLineEdit()
        self.api_edit.setText(DEFAULT_API_KEY)
        self.api_edit.setEchoMode(QLineEdit.Password)
        show_btn = QPushButton("👁")
        show_btn.setFixedWidth(32)
        show_btn.setCheckable(True)
        show_btn.setStyleSheet("background:#1a1a2e;border:1px solid #2a2a4a;border-radius:4px;")
        show_btn.toggled.connect(
            lambda v: self.api_edit.setEchoMode(QLineEdit.Normal if v else QLineEdit.Password)
        )
        sl.addWidget(self.api_edit, 1)
        sl.addWidget(show_btn)

        root.addWidget(settings_grp)

        # ── Progress ──────────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)

        # ── Log ───────────────────────────────────────────────────────────────
        log_grp = QGroupBox("📋  Processing Log")
        lg = QVBoxLayout(log_grp)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Consolas", 11))
        lg.addWidget(self.log_edit)
        root.addWidget(log_grp, 1)   # stretch to fill remaining space

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.start_btn = QPushButton("▶  Bắt đầu tự động")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self._on_start)

        self.stop_btn = QPushButton("⏹  Dừng")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)

        clear_btn = QPushButton("🗑  Xóa Log")
        clear_btn.setObjectName("clear_btn")
        clear_btn.clicked.connect(self.log_edit.clear)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)
        root.addLayout(btn_row)

    # ── Slots ─────────────────────────────────────────────────────────────────

    # ── URL helpers ────────────────────────────────────────────────────────────

    def _import_url_file(self):
        """Mở file .txt (mỗi dòng 1 URL) và thêm vào ô URL."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file .txt chứa danh sách URL",
            str(Path.home()), "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            links = [l.strip() for l in content.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            existing = self.url_edit.toPlainText().strip()
            combined = (existing + "\n" if existing else "") + "\n".join(links)
            self.url_edit.setPlainText(combined.strip())
            self._log(f"📂 Import {len(links)} URL từ {Path(path).name}")
        except Exception as exc:
            self._log(f"❌ Không thể đọc file: {exc}")

    def _update_url_count(self):
        """Cập nhật nhãn số URL."""
        urls = [l.strip() for l in self.url_edit.toPlainText().splitlines()
                if l.strip() and not l.strip().startswith("#")]
        self._url_count_lbl.setText(f"{len(urls)} URL")

    def _refresh_local_files(self):
        """Scan INPUT_DIR for .wav files and populate the list."""
        self.local_list.clear()
        if INPUT_DIR.exists():
            wavs = sorted(INPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            for wav in wavs:
                size_mb = wav.stat().st_size / 1_048_576
                has_txt = wav.with_suffix(".txt").exists()
                txt_icon = "📄" if has_txt else "  "
                item = QListWidgetItem(f"🔊 {wav.name}  {txt_icon}  [{size_mb:.1f} MB]")
                item.setData(Qt.UserRole, str(wav))
                self.local_list.addItem(item)
            self.local_count_lbl.setText(f"{len(wavs)} file WAV")
        else:
            self.local_count_lbl.setText("0 file WAV (chưa có thư mục input/)")

    def _on_start(self):
        urls = []
        for line in self.url_edit.toPlainText().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

        # Lấy file local đã chọn
        local_files = []
        for item in self.local_list.selectedItems():
            local_files.append(item.data(Qt.UserRole))

        if not urls and not local_files:
            self._log("❌ Chưa nhập URL hoặc chưa chọn file nào.")
            return

        total = len(urls) + len(local_files)
        self._log(f"🚀 Bắt đầu xử lý {total} mục ({len(urls)} URL + {len(local_files)} file local)…")
        self._log(f"   Input  → {INPUT_DIR}")
        self._log(f"   Output → {OUTPUT_DIR}")

        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Đang xử lý {total} mục…")

        mode_idx       = self.mode_combo.currentIndex()  # 0=seq,1=par2,2=par3
        parallel_count = [1, 2, 3][mode_idx]

        self._worker = AutoPipelineWorker(
            urls=urls,
            local_files=local_files,
            language=self.lang_combo.currentText(),
            whisper_model=self.model_combo.currentText(),
            openai_key=self.api_edit.text().strip(),
            subtitle_lang=self.sub_lang_combo.currentText(),
            parallel_count=parallel_count,
        )
        self._worker.log_sig.connect(self._log)
        self._worker.progress_sig.connect(self.progress.setValue)
        self._worker.done_sig.connect(self._on_done)
        self._worker.error_sig.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._log("🛑 Đang dừng, chờ task hiện tại kết thúc…")
            self._worker.stop()
            self.stop_btn.setEnabled(False)

    def _on_done(self, total: int):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(100)
        self.statusBar().showMessage(f"✅ Xong — {total} segments")
        self._refresh_local_files()  # Cập nhật lại danh sách file local

    def _on_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log(msg)
        self.statusBar().showMessage("❌ Lỗi")

    def _log(self, msg: str):
        if "❌" in msg or "lỗi" in msg.lower() or "error" in msg.lower():
            color = "#ff6b6b"
        elif "✅" in msg or "🎉" in msg or "hoàn" in msg.lower():
            color = "#4ecca3"
        elif "⚠" in msg or "warning" in msg.lower():
            color = "#f9ca24"
        elif msg.startswith("  ") or "─" in msg or "═" in msg:
            color = "#74b9ff"
        elif "📥" in msg or "🔗" in msg or "📁" in msg or "📋" in msg:
            color = "#a29bfe"
        else:
            color = "#dfe6e9"

        escaped = (msg.replace("&", "&amp;")
                      .replace("<", "&lt;")
                      .replace(">", "&gt;"))
        self.log_edit.append(f'<span style="color:{color}">{escaped}</span>')
        cur = self.log_edit.textCursor()
        cur.movePosition(QTextCursor.End)
        self.log_edit.setTextCursor(cur)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logger(console=False)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = AutoApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

