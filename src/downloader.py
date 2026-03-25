"""
src/downloader.py — Download YouTube audio + subtitles via yt-dlp
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .logger import get_logger

log = get_logger("downloader")


@dataclass
class DownloadResult:
    url: str
    video_id: str = ""
    title: str = ""
    audio_path: str = ""      # Path to downloaded WAV audio
    subtitle_path: str = ""   # Path to subtitle .txt (if available)
    error: str = ""

    @property
    def success(self) -> bool:
        return bool(self.audio_path) and not self.error


class YouTubeDownloader:
    """
    Download YouTube audio and subtitles using yt-dlp.

    yt-dlp is preferred over JDownloader for this use case because:
    - Direct audio extraction (no re-encode video)
    - Built-in subtitle download in many formats
    - No dependency on running GUI app
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._dl_cfg = cfg.get("download", {})
        self._output_dir = Path(cfg.get("dataset", {}).get("output_dir", "output"))
        # Downloads go to input/ (same as local files), NOT output/audio/
        _base = self._output_dir.parent  # e.g. d:/aiphattrien/yt2dataset
        self._audio_dir = _base / "input"
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._subtitle_lang = self._dl_cfg.get("subtitle_lang", "vi")
        self._cookies_file = self._dl_cfg.get("cookies_file", "")
        self._sample_rate = cfg.get("dataset", {}).get("sample_rate", 22050)

    def _yt_dlp_cmd(self) -> list[str]:
        """Base yt-dlp command with common flags."""
        cmd = [sys.executable, "-m", "yt_dlp"]
        if self._cookies_file:
            cmd += ["--cookies", self._cookies_file]
        return cmd

    def download(self, url: str) -> DownloadResult:
        """Download audio and subtitles for a YouTube URL."""
        result = DownloadResult(url=url)
        log.info("Starting download: %s", url)

        # ── Step 1: Get video info ────────────────────────────────────────────
        try:
            info = self._get_info(url)
            result.video_id = info.get("id", "")
            result.title = info.get("title", "video")
            log.info("Video: '%s' [%s]", result.title, result.video_id)
        except Exception as exc:
            result.error = f"Info fetch failed: {exc}"
            log.error(result.error)
            return result

        # Safe title for filesystem
        safe_title = self._sanitize(result.title)
        out_base = self._audio_dir / safe_title

        # ── Step 2: Download audio ────────────────────────────────────────────
        audio_path = self._download_audio(url, out_base)
        if not audio_path:
            result.error = "Audio download failed"
            log.error(result.error)
            return result
        result.audio_path = str(audio_path)
        log.info("Audio saved: %s", audio_path)

        # ── Step 3: Download subtitles (optional) ─────────────────────────────
        sub_path = self._download_subtitles(url, out_base)
        if sub_path:
            result.subtitle_path = str(sub_path)
            log.info("Subtitle saved: %s", sub_path)
        else:
            log.info("No subtitle available — will use Whisper transcription only")

        return result

    def _get_info(self, url: str) -> dict:
        """Get video metadata via yt-dlp JSON dump."""
        import json
        cmd = self._yt_dlp_cmd() + [
            "--dump-json",
            "--no-playlist",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip())
        return json.loads(proc.stdout.strip())

    def _download_audio(self, url: str, out_base: Path) -> Path | None:
        """Download best audio and convert to WAV 22050Hz mono."""
        wav_path = out_base.with_suffix(".wav")
        if wav_path.exists():
            log.info("Audio already exists, skipping download: %s", wav_path)
            return wav_path

        # Download to temp m4a/webm, then convert with ffmpeg via yt-dlp
        cmd = self._yt_dlp_cmd() + [
            "--no-playlist",
            "-f", "bestaudio/best",
            "-x",                           # extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", f"-ar {self._sample_rate} -ac 1",
            "-o", str(out_base.with_suffix(".%(ext)s")),
            url,
        ]
        log.debug("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            log.error("yt-dlp audio error: %s", proc.stderr[-500:])
            return None

        # yt-dlp may output .wav or intermediary — find it
        for ext in (".wav", ".m4a", ".webm", ".mp3", ".ogg"):
            candidate = out_base.with_suffix(ext)
            if candidate.exists():
                if ext != ".wav":
                    # Convert to WAV
                    converted = self._ffmpeg_to_wav(candidate, wav_path)
                    candidate.unlink(missing_ok=True)
                    return converted if converted else None
                return candidate

        return None

    def _ffmpeg_to_wav(self, src: Path, dst: Path) -> Path | None:
        """Convert any audio file to WAV 22050Hz mono using ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-ar", str(self._sample_rate),
            "-ac", "1",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            log.error("ffmpeg error: %s", proc.stderr[-300:])
            return None
        return dst

    def _download_subtitles(self, url: str, out_base: Path) -> Path | None:
        """Try to download auto/manual subtitles; convert to plain .txt."""
        txt_path = out_base.with_suffix(".txt")
        if txt_path.exists():
            log.info("Subtitle already exists: %s", txt_path)
            return txt_path

        # Try manual subs first, then auto subs
        for auto in ("", "--write-auto-subs"):
            for lang in (self._subtitle_lang, "en", ""):
                sub_base = out_base.with_suffix("")
                cmd = self._yt_dlp_cmd() + [
                    "--no-playlist",
                    "--skip-download",
                    "--write-subs",
                    *(["--write-auto-subs"] if auto else []),
                    "--sub-langs", lang if lang else "all",
                    "--sub-format", "vtt/srt/best",
                    "--convert-subs", "srt",
                    "-o", str(sub_base) + ".%(ext)s",
                    url,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                # Find downloaded .srt file
                srt_files = list(self._audio_dir.glob(f"{out_base.stem}*.srt"))
                if srt_files:
                    txt = self._srt_to_txt(srt_files[0], txt_path)
                    # clean up .srt
                    srt_files[0].unlink(missing_ok=True)
                    return txt if txt else None

        return None

    def _srt_to_txt(self, srt_path: Path, txt_path: Path) -> Path | None:
        """Convert .srt subtitle file to plain text (one sentence per line)."""
        import re
        try:
            content = srt_path.read_text(encoding="utf-8", errors="replace")
            # Remove SRT timestamps and numbers
            lines = []
            for line in content.splitlines():
                line = line.strip()
                # Skip index lines (pure numbers)
                if re.match(r"^\d+$", line):
                    continue
                # Skip timestamp lines
                if re.match(r"\d{2}:\d{2}:\d{2}", line):
                    continue
                # Remove HTML tags
                line = re.sub(r"<[^>]+>", "", line)
                if line:
                    lines.append(line)

            # Merge duplicate consecutive lines (auto-sub artifact)
            merged: list[str] = []
            for line in lines:
                if not merged or line != merged[-1]:
                    merged.append(line)

            txt_path.write_text("\n".join(merged), encoding="utf-8")
            log.info("Converted SRT → TXT: %d lines", len(merged))
            return txt_path
        except Exception as exc:
            log.warning("SRT→TXT failed: %s", exc)
            return None

    @staticmethod
    def _sanitize(name: str, max_len: int = 60) -> str:
        """Make a filesystem-safe name."""
        import re
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe[:max_len]

    # ── Download to input format ───────────────────────────────────────────────

    def download_to_input_format(
        self,
        url: str,
        dest_dir: Path,
        speaker_name: str = "",
        index: int = 1,
        progress_cb=None,
    ) -> DownloadResult:
        """
        Download audio + subtitle and place them into dest_dir using the
        exact input format expected by build_dataset_app_final.py:

            dest_dir/
            ├── <name>.wav
            └── <name>.txt   (if subtitle available)

        If speaker_name is given:
            dest_dir/<speaker_name>_<index:03d>.wav  (e.g. VB4_001.wav)
        Otherwise:
            dest_dir/<sanitized_video_title>.wav

        Args:
            url:          YouTube URL
            dest_dir:     Target directory (input folder for dataset builder)
            speaker_name: Speaker/prefix name (e.g. "VB4") — optional
            index:        File index when speaker_name is set (1-based)
            progress_cb:  Optional callback(str) for status messages
        """
        cb = progress_cb or (lambda m: log.info(m))
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        cb(f"📥 Downloading: {url}")

        # Use temp subdir inside dest_dir to avoid polluting dest_dir
        tmp_dir = dest_dir / "_tmp_dl"
        tmp_dir.mkdir(exist_ok=True)

        # Temporarily redirect audio_dir to tmp_dir
        original_audio_dir = self._audio_dir
        self._audio_dir = tmp_dir

        try:
            result = self.download(url)
        finally:
            self._audio_dir = original_audio_dir

        if not result.success:
            cb(f"❌ Download failed: {result.error}")
            # Clean up tmp
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return result

        # Determine destination filename
        if speaker_name:
            dest_stem = f"{speaker_name}_{index:03d}"
        else:
            dest_stem = Path(result.audio_path).stem

        # Move wav
        src_wav = Path(result.audio_path)
        dst_wav = dest_dir / f"{dest_stem}.wav"
        if dst_wav.exists():
            # Avoid overwrite: increment suffix
            counter = 2
            while dst_wav.exists():
                dst_wav = dest_dir / f"{dest_stem}_{counter:02d}.wav"
                counter += 1
        try:
            import shutil
            shutil.move(str(src_wav), str(dst_wav))
            result.audio_path = str(dst_wav)
            cb(f"✅ Audio → {dst_wav.name}")
        except Exception as exc:
            cb(f"⚠ Could not move wav: {exc}")

        # Move txt (subtitle) if available
        if result.subtitle_path:
            src_txt = Path(result.subtitle_path)
            dst_txt = dest_dir / f"{dest_stem}.txt"
            try:
                shutil.move(str(src_txt), str(dst_txt))
                result.subtitle_path = str(dst_txt)
                cb(f"✅ Subtitle → {dst_txt.name}")
            except Exception as exc:
                cb(f"⚠ Could not move txt: {exc}")
        else:
            cb("ℹ No subtitle — only WAV downloaded")

        # Clean up tmp dir (remove any leftover files)
        import shutil as _shutil
        _shutil.rmtree(tmp_dir, ignore_errors=True)

        return result
