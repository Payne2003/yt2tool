"""
app.py — YT2Dataset GUI (PyQt5)

Giao diện đồ họa để download YouTube → build TTS dataset.
Thiết kế tương tự build_dataset_app_final.py từ tts_project.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPalette, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

# ── Bootstrap sys.path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config
from src.dataset_builder import DatasetEntry
from src.downloader import YouTubeDownloader
from src.logger import setup_logger
from src.pipeline import YT2DatasetPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """Runs YT2DatasetPipeline in a background thread and emits signals."""

    log_signal     = pyqtSignal(str)           # progress message
    progress_signal = pyqtSignal(int)          # 0–100
    finished_signal = pyqtSignal(list)         # list[DatasetEntry]
    error_signal   = pyqtSignal(str)           # error message

    def __init__(
        self,
        urls: list[str],
        language: str,
        whisper_model: str,
        output_dir: str,
        openai_key: str,
        append_mode: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.urls = urls
        self.language = language
        self.whisper_model = whisper_model
        self.output_dir = output_dir
        self.openai_key = openai_key
        self.append_mode = append_mode
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            cfg = load_config()
            cfg.setdefault("dataset", {})["append_mode"] = self.append_mode
            if self.output_dir:
                cfg["dataset"]["output_dir"] = self.output_dir

            def progress_cb(msg: str):
                self.log_signal.emit(msg)

            pipeline = YT2DatasetPipeline(
                cfg=cfg,
                language=self.language if self.language != "auto" else None,
                whisper_model=self.whisper_model,
                output_dir=self.output_dir or None,
                openai_key=self.openai_key or None,
            )

            # Monkey-patch progress callback into downloader / transcriber
            pipeline._transcriber._cb = progress_cb
            pipeline._downloader  # no-op, logging goes to logger

            all_entries: list[DatasetEntry] = []
            total = len(self.urls)

            for i, url in enumerate(self.urls):
                if self._stop:
                    self.log_signal.emit("🛑 Processing stopped by user.")
                    break
                self.log_signal.emit(f"\n── URL {i+1}/{total}: {url}")
                entries = pipeline.run(url)
                all_entries.extend(entries)
                self.progress_signal.emit(int((i + 1) / total * 100))

            self.finished_signal.emit(all_entries)

        except Exception as exc:
            import traceback
            self.error_signal.emit(f"❌ Error: {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Download Raw Worker
# ─────────────────────────────────────────────────────────────────────────────

class DownloadRawWorker(QThread):
    """
    Downloads YouTube audio + subtitles into a folder formatted as the
    expected input for build_dataset_app_final.py:
        dest_dir/<speaker>_001.wav + <speaker>_001.txt
    """
    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(list)   # list of (wav_path, txt_path or "")
    error_signal    = pyqtSignal(str)

    def __init__(
        self,
        urls: list[str],
        dest_dir: str,
        speaker_name: str,
        subtitle_lang: str,
        parent=None,
    ):
        super().__init__(parent)
        self.urls         = urls
        self.dest_dir     = dest_dir
        self.speaker_name = speaker_name.strip()
        self.subtitle_lang = subtitle_lang
        self._stop        = False

    def stop(self):
        self._stop = True

    def run(self):
        from pathlib import Path
        try:
            cfg = load_config()
            cfg.setdefault("download", {})["subtitle_lang"] = self.subtitle_lang
            # Point output_dir to dest_dir so downloader uses it as base
            cfg.setdefault("dataset", {})["output_dir"] = self.dest_dir

            dl = YouTubeDownloader(cfg)
            results = []
            total = len(self.urls)

            for i, url in enumerate(self.urls):
                if self._stop:
                    self.log_signal.emit("🛑 Dừng theo yêu cầu.")
                    break

                self.log_signal.emit(f"\n── [{i+1}/{total}] {url}")

                result = dl.download_to_input_format(
                    url=url,
                    dest_dir=Path(self.dest_dir),
                    speaker_name=self.speaker_name,
                    index=i + 1,
                    progress_cb=self.log_signal.emit,
                )

                if result.success:
                    results.append((result.audio_path, result.subtitle_path))
                else:
                    self.log_signal.emit(f"❌ Thất bại: {result.error}")

                self.progress_signal.emit(int((i + 1) / total * 100))

            self.finished_signal.emit(results)

        except Exception as exc:
            import traceback
            self.error_signal.emit(f"❌ Lỗi: {exc}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Window
# ─────────────────────────────────────────────────────────────────────────────

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QLineEdit:focus, QTextEdit:focus {
    border-color: #89b4fa;
}
QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 6px 16px;
    font-weight: bold;
}
QPushButton:hover  { background-color: #b4befe; }
QPushButton:disabled { background-color: #45475a; color: #6c7086; }
QPushButton#stop_btn {
    background-color: #f38ba8;
}
QPushButton#stop_btn:hover { background-color: #eba0ac; }
QPushButton#clear_btn {
    background-color: #a6e3a1;
    color: #1e1e2e;
}
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #cdd6f4;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #313244;
    color: #cdd6f4;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}
QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    background-color: #313244;
    color: #cdd6f4;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #89b4fa, stop:1 #cba6f7);
    border-radius: 3px;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #45475a;
    border-radius: 3px;
    background-color: #313244;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
}
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
QTabBar::tab {
    background: #313244; color: #6c7086;
    padding: 6px 18px; border-radius: 4px 4px 0 0;
    margin-right: 2px;
}
QTabBar::tab:selected { background: #1e1e2e; color: #89b4fa; font-weight: bold; }
QScrollBar:vertical {
    background: #1e1e2e; width: 10px;
}
QScrollBar::handle:vertical {
    background: #45475a; border-radius: 5px;
}
QLabel#hint { color: #6c7086; font-style: italic; font-size: 11px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YT2Dataset — YouTube → TTS Dataset Builder")
        self.setMinimumSize(900, 720)
        self.setStyleSheet(DARK_STYLE)
        self.showMaximized()
        self._worker: PipelineWorker | None = None
        self._dl_worker: DownloadRawWorker | None = None
        self._build_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(12, 12, 12, 12)

        # ── Header ────────────────────────────────────────────────────────────
        header = QLabel("🎬  YT2Dataset — YouTube → TTS Dataset Builder")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #cba6f7; padding: 4px 0 8px 0;")
        root.addWidget(header)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs = QTabWidget()
        root.addWidget(tabs)

        tabs.addTab(self._tab_input(),        "🚀  Build Dataset")
        tabs.addTab(self._tab_download_raw(), "📥  Download Raw")
        tabs.addTab(self._tab_settings(),     "⚙️  Settings")

        # ── Progress ──────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFixedHeight(20)
        root.addWidget(self.progress_bar)

        # ── Log ───────────────────────────────────────────────────────────────
        log_group = QGroupBox("📋  Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 11))
        self.log_text.setMinimumHeight(250)
        log_layout.addWidget(self.log_text)
        root.addWidget(log_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Processing")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        clear_btn = QPushButton("🗑  Clear Log")
        clear_btn.setObjectName("clear_btn")
        clear_btn.clicked.connect(self.log_text.clear)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(clear_btn)
        root.addLayout(btn_row)

        # ── Status bar ────────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready")

    # ── Tab: Input ────────────────────────────────────────────────────────────

    def _tab_input(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # URL text area
        url_group = QGroupBox("YouTube URLs (mỗi URL một dòng)")
        ul = QVBoxLayout(url_group)
        self.url_edit = QTextEdit()
        self.url_edit.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\n"
            "https://youtu.be/...\n"
            "https://www.youtube.com/watch?v=..."
        )
        self.url_edit.setFont(QFont("Consolas", 12))
        self.url_edit.setMinimumHeight(160)
        ul.addWidget(self.url_edit)

        hint1 = QLabel("Paste một hoặc nhiều YouTube URL, mỗi URL một dòng.")
        hint1.setObjectName("hint")
        ul.addWidget(hint1)
        layout.addWidget(url_group)

        # OR: load from file
        file_group = QGroupBox("Hoặc tải từ file URLs (.txt)")
        fl = QHBoxLayout(file_group)
        self.url_file_edit = QLineEdit()
        self.url_file_edit.setPlaceholderText("(tùy chọn) path tới file urls.txt...")
        browse_url_btn = QPushButton("Browse…")
        browse_url_btn.clicked.connect(self._browse_url_file)
        fl.addWidget(self.url_file_edit)
        fl.addWidget(browse_url_btn)
        layout.addWidget(file_group)

        layout.addStretch()
        return w

    # ── Tab: Download Raw ─────────────────────────────────────────────────────

    def _tab_download_raw(self) -> QWidget:
        """Tab for downloading YouTube audio+subtitles as input for dataset builder."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # ── URL input ──────────────────────────────────────────────────────────
        url_grp = QGroupBox("YouTube URLs (mỗi URL một dòng)")
        ul = QVBoxLayout(url_grp)
        self.dl_url_edit = QTextEdit()
        self.dl_url_edit.setPlaceholderText(
            "https://www.youtube.com/watch?v=...\n"
            "https://youtu.be/..."
        )
        self.dl_url_edit.setFont(QFont("Consolas", 12))
        self.dl_url_edit.setMinimumHeight(120)
        ul.addWidget(self.dl_url_edit)
        layout.addWidget(url_grp)

        # ── Destination folder ─────────────────────────────────────────────────
        dest_grp = QGroupBox("📁  Thư mục đích (Input folder cho build_dataset)")
        dl = QVBoxLayout(dest_grp)
        dest_row = QHBoxLayout()
        self.dl_dest_edit = QLineEdit()
        self.dl_dest_edit.setPlaceholderText("Chọn thư mục lưu WAV + TXT...")
        browse_dest = QPushButton("Browse…")
        browse_dest.clicked.connect(self._browse_dl_dest)
        dest_row.addWidget(self.dl_dest_edit)
        dest_row.addWidget(browse_dest)
        dl.addLayout(dest_row)
        hint_dest = QLabel("Sau khi tải xong, dùng thư mục này làm Input Folder trong build_dataset_app_final.py")
        hint_dest.setObjectName("hint")
        dl.addWidget(hint_dest)
        layout.addWidget(dest_grp)

        # ── Speaker / prefix name ──────────────────────────────────────────────
        spk_grp = QGroupBox("🎤  Tên Speaker / Prefix (tùy chọn)")
        sl = QVBoxLayout(spk_grp)
        spk_row = QHBoxLayout()
        spk_row.addWidget(QLabel("Speaker name:"))
        self.dl_speaker_edit = QLineEdit()
        self.dl_speaker_edit.setPlaceholderText("VB4  (nếu để trống → dùng tên video)")
        self.dl_speaker_edit.setFixedWidth(180)
        spk_row.addWidget(self.dl_speaker_edit)
        spk_row.addStretch()
        sl.addLayout(spk_row)
        hint_spk = QLabel("VD: speaker='VB4', 3 URLs → VB4_001.wav, VB4_002.wav, VB4_003.wav")
        hint_spk.setObjectName("hint")
        sl.addWidget(hint_spk)
        layout.addWidget(spk_grp)

        # ── Subtitle language ──────────────────────────────────────────────────
        lang_grp = QGroupBox("🌐  Ngôn ngữ Subtitle ưu tiên")
        ll = QHBoxLayout(lang_grp)
        ll.addWidget(QLabel("Subtitle lang:"))
        self.dl_lang_combo = QComboBox()
        self.dl_lang_combo.addItems(["vi", "en", "ja", "ko", "tl"])
        ll.addWidget(self.dl_lang_combo)
        ll.addStretch()
        layout.addWidget(lang_grp)

        # ── Progress + Log ─────────────────────────────────────────────────────
        self.dl_progress = QProgressBar()
        self.dl_progress.setRange(0, 100)
        self.dl_progress.setFixedHeight(18)
        layout.addWidget(self.dl_progress)

        log_grp = QGroupBox("📋  Log")
        lg = QVBoxLayout(log_grp)
        self.dl_log = QTextEdit()
        self.dl_log.setReadOnly(True)
        self.dl_log.setFont(QFont("Consolas", 10))
        self.dl_log.setMinimumHeight(150)
        lg.addWidget(self.dl_log)
        layout.addWidget(log_grp)

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.dl_start_btn = QPushButton("📥  Download Only (không Whisper)")
        self.dl_start_btn.clicked.connect(self._on_dl_start)
        self.dl_stop_btn = QPushButton("⏹  Stop")
        self.dl_stop_btn.setObjectName("stop_btn")
        self.dl_stop_btn.setEnabled(False)
        self.dl_stop_btn.clicked.connect(self._on_dl_stop)
        self.dl_clear_btn = QPushButton("🗑  Clear")
        self.dl_clear_btn.setObjectName("clear_btn")
        self.dl_clear_btn.clicked.connect(self.dl_log.clear)
        btn_row.addWidget(self.dl_start_btn)
        btn_row.addWidget(self.dl_stop_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.dl_clear_btn)
        layout.addLayout(btn_row)

        layout.addStretch()
        return w

    # ── Tab: Settings ─────────────────────────────────────────────────────────

    def _tab_settings(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # Output directory
        out_group = QGroupBox("📁  Output Directory")
        ol = QVBoxLayout(out_group)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Chọn thư mục lưu dataset (mặc định: output/)")
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.output_edit)
        out_row.addWidget(browse_out)
        ol.addLayout(out_row)

        self.append_check = QCheckBox("Append mode — thêm vào dataset có sẵn (không ghi đè)")
        self.append_check.setChecked(True)
        ol.addWidget(self.append_check)
        layout.addWidget(out_group)

        # Language settings
        lang_group = QGroupBox("🌐  Language Settings")
        ll = QVBoxLayout(lang_group)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems([
            "Auto-detect", "Vietnamese (vi)", "English (en)",
            "Japanese (ja)", "Korean (ko)", "Filipino (tl)"
        ])
        self.lang_combo.currentTextChanged.connect(self._on_lang_changed)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        ll.addLayout(lang_row)

        self.lang_hint = QLabel("Tự động phát hiện ngôn ngữ từ audio")
        self.lang_hint.setObjectName("hint")
        ll.addWidget(self.lang_hint)
        layout.addWidget(lang_group)

        # Whisper settings
        whisper_group = QGroupBox("🎙  Whisper Settings")
        wl = QVBoxLayout(whisper_group)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large", "large-v3"])
        self.model_combo.setCurrentText("medium")
        model_row.addWidget(self.model_combo)
        model_row.addStretch()
        wl.addLayout(model_row)

        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel("Segment min (s):"))
        self.seg_min = QLineEdit("2.0")
        self.seg_min.setFixedWidth(60)
        seg_row.addWidget(self.seg_min)
        seg_row.addSpacing(20)
        seg_row.addWidget(QLabel("Segment max (s):"))
        self.seg_max = QLineEdit("12.0")
        self.seg_max.setFixedWidth(60)
        seg_row.addWidget(self.seg_max)
        seg_row.addStretch()
        wl.addLayout(seg_row)
        layout.addWidget(whisper_group)

        # OpenAI settings
        api_group = QGroupBox("🤖  OpenAI Correction (tùy chọn)")
        al = QVBoxLayout(api_group)

        api_row = QHBoxLayout()
        api_row.addWidget(QLabel("OpenAI API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("sk-... (tùy chọn, để trống nếu không cần)")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        show_key_btn = QPushButton("👁")
        show_key_btn.setFixedWidth(32)
        show_key_btn.setCheckable(True)
        show_key_btn.toggled.connect(
            lambda v: self.api_key_edit.setEchoMode(
                QLineEdit.Normal if v else QLineEdit.Password
            )
        )
        api_row.addWidget(self.api_key_edit)
        api_row.addWidget(show_key_btn)
        al.addLayout(api_row)

        api_hint = QLabel("Nếu có API key: transcription sẽ được sửa lỗi tự động bằng GPT-4.1-nano")
        api_hint.setObjectName("hint")
        al.addWidget(api_hint)
        layout.addWidget(api_group)

        layout.addStretch()
        return w

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _browse_dl_dest(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục đích")
        if folder:
            self.dl_dest_edit.setText(folder)
            self._dl_log(f"📁 Dest: {folder}")

    def _on_dl_start(self):
        urls = []
        for line in self.dl_url_edit.toPlainText().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
        if not urls:
            self._dl_log("❌ Chưa nhập URL nào.")
            return
        dest = self.dl_dest_edit.text().strip()
        if not dest:
            self._dl_log("❌ Chưa chọn thư mục đích.")
            return
        speaker = self.dl_speaker_edit.text().strip()
        lang    = self.dl_lang_combo.currentText()

        self._dl_log(f"🚀 Bắt đầu tải {len(urls)} URL(s)…")
        self._dl_log(f"   Dest: {dest} | Speaker: '{speaker or '(video title)'}' | Subtitle lang: {lang}")

        self.dl_progress.setValue(0)
        self.dl_start_btn.setEnabled(False)
        self.dl_stop_btn.setEnabled(True)

        self._dl_worker = DownloadRawWorker(
            urls=urls,
            dest_dir=dest,
            speaker_name=speaker,
            subtitle_lang=lang,
        )
        self._dl_worker.log_signal.connect(self._dl_log)
        self._dl_worker.progress_signal.connect(self.dl_progress.setValue)
        self._dl_worker.finished_signal.connect(self._on_dl_finished)
        self._dl_worker.error_signal.connect(self._on_dl_error)
        self._dl_worker.start()

    def _on_dl_stop(self):
        if self._dl_worker:
            self._dl_log("🛑 Đang dừng…")
            self._dl_worker.stop()
            self.dl_stop_btn.setEnabled(False)

    def _on_dl_finished(self, results: list):
        self.dl_start_btn.setEnabled(True)
        self.dl_stop_btn.setEnabled(False)
        self.dl_progress.setValue(100)
        dest = self.dl_dest_edit.text().strip()
        self._dl_log(f"\n✅ Hoàn thành! Đã tải {len(results)} file(s) vào: {dest}")
        self._dl_log("📋 Files:")
        for wav, txt in results:
            from pathlib import Path
            wav_name = Path(wav).name if wav else "?"
            txt_name = Path(txt).name if txt else "(no subtitle)"
            self._dl_log(f"   🔊 {wav_name}  +  📄 {txt_name}")
        if results:
            self._dl_log(f"\n💡 Dùng '{dest}' làm Input Folder trong build_dataset_app_final.py")

    def _on_dl_error(self, msg: str):
        self.dl_start_btn.setEnabled(True)
        self.dl_stop_btn.setEnabled(False)
        self._dl_log(f"\n❌ Lỗi:\n{msg}")

    def _dl_log(self, msg: str):
        """Append message to download log with color coding."""
        if "❌" in msg or "Error" in msg.lower() or "Lỗi" in msg:
            color = "#f38ba8"
        elif "✅" in msg or "✓" in msg or "Hoàn" in msg:
            color = "#a6e3a1"
        elif "⚠" in msg or "Warning" in msg.lower():
            color = "#f9e2af"
        elif msg.startswith("  ") or "─" in msg:
            color = "#89dceb"
        else:
            color = "#cdd6f4"
        escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.dl_log.append(f'<span style="color:{color}">{escaped}</span>')
        from PyQt5.QtGui import QTextCursor
        cursor = self.dl_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.dl_log.setTextCursor(cursor)

    def _browse_url_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file URLs", filter="Text files (*.txt);;All files (*)"
        )
        if path:
            self.url_file_edit.setText(path)
            self._log(f"📂 URL file: {path}")

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục output")
        if folder:
            self.output_edit.setText(folder)
            self._log(f"📁 Output dir: {folder}")

    def _on_lang_changed(self, text: str):
        hints = {
            "Auto-detect": "Tự động phát hiện ngôn ngữ từ audio",
            "Vietnamese (vi)": "Xử lý tối ưu cho tiếng Việt — dùng OptimizedVietnameseTTSNormalizer",
            "English (en)": "English — dùng Coqui english_cleaners",
            "Japanese (ja)": "日本語 — dùng OptimizedJapaneseTTSNormalizer",
            "Korean (ko)": "한국어 — dùng OptimizedKoreanTTSNormalizer",
            "Filipino (tl)": "Filipino — dùng OptimizedFilipinoTTSNormalizer",
        }
        self.lang_hint.setText(hints.get(text, ""))

    def _on_start(self):
        # Collect URLs
        urls: list[str] = []
        raw = self.url_edit.toPlainText().strip()
        if raw:
            for line in raw.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

        url_file = self.url_file_edit.text().strip()
        if url_file and Path(url_file).exists():
            for line in Path(url_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

        if not urls:
            self._log("❌ Chưa nhập URL nào. Paste URL vào ô Input URLs hoặc chọn file .txt.")
            return

        # Map language
        lang_map = {
            "Auto-detect": "auto",
            "Vietnamese (vi)": "vi",
            "English (en)": "en",
            "Japanese (ja)": "ja",
            "Korean (ko)": "ko",
            "Filipino (tl)": "tl",
        }
        language = lang_map.get(self.lang_combo.currentText(), "auto")

        # Read settings
        whisper_model = self.model_combo.currentText()
        output_dir = self.output_edit.text().strip() or "output"
        openai_key = self.api_key_edit.text().strip()
        append_mode = self.append_check.isChecked()

        # Update config with segmentation values
        try:
            seg_min = float(self.seg_min.text())
            seg_max = float(self.seg_max.text())
        except ValueError:
            seg_min, seg_max = 2.0, 12.0

        self._log(f"🚀 Bắt đầu xử lý {len(urls)} URL(s)…")
        self._log(f"   Language: {language} | Whisper: {whisper_model} | Output: {output_dir}")
        if openai_key:
            self._log("   OpenAI correction: ✅ bật")
        else:
            self._log("   OpenAI correction: ⬜ tắt (không có API key)")

        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Processing {len(urls)} URL(s)…")

        self._worker = PipelineWorker(
            urls=urls,
            language=language,
            whisper_model=whisper_model,
            output_dir=output_dir,
            openai_key=openai_key,
            append_mode=append_mode,
        )
        self._worker.log_signal.connect(self._log)
        self._worker.progress_signal.connect(self.progress_bar.setValue)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.error_signal.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker:
            self._log("🛑 Đang dừng…")
            self._worker.stop()
            self.stop_btn.setEnabled(False)

    def _on_finished(self, entries: list):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage(f"Done — {len(entries)} segments")

        self._log(f"\n✅ Hoàn thành! Đã tạo {len(entries)} segments.")
        if entries:
            out_dir = self.output_edit.text().strip() or "output"
            metadata = Path(out_dir) / "metadata.csv"
            self._log(f"📋 Metadata: {metadata}")
            self._log("📁 Ví dụ files:")
            for e in entries[:5]:
                self._log(f"   {e.wav_rel}  |  {e.text[:60]}")
            if len(entries) > 5:
                self._log(f"   … và {len(entries)-5} files khác")
            # Count total metadata entries
            if metadata.exists():
                total = sum(1 for l in metadata.read_text(encoding="utf-8").splitlines() if l.strip())
                self._log(f"📊 Tổng entries trong metadata.csv: {total}")

    def _on_error(self, msg: str):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Error!")
        self._log(f"\n❌ Lỗi:\n{msg}")

    def _log(self, msg: str):
        """Append message to log with color coding."""
        # Simple color via HTML
        if "❌" in msg or "Error" in msg.lower():
            color = "#f38ba8"
        elif "✅" in msg or "✓" in msg or "Done" in msg:
            color = "#a6e3a1"
        elif "⚠" in msg or "Warning" in msg.lower():
            color = "#f9e2af"
        elif msg.startswith("  ") or "segment" in msg.lower():
            color = "#89dceb"
        else:
            color = "#cdd6f4"

        escaped = (msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        self.log_text.append(f'<span style="color:{color}">{escaped}</span>')
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    setup_logger(console=False)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
