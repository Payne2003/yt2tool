"""
src/transcriber.py — Whisper transcription + word-level segmentation
Ported from tts_project/build_dataset_app_final.py (WhisperWorker logic)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .logger import get_logger

log = get_logger("transcriber")


@dataclass
class Segment:
    """One audio segment with transcript."""
    start: float
    end: float
    text: str
    words: list[dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Transcriber:
    """
    Transcribe audio using OpenAI Whisper with word-level timestamps,
    then segment into 2–12 second chunks suitable for TTS training.

    Logic ported directly from tts_project/build_dataset_app_final.py.
    """

    SENTENCE_ENDINGS = {
        "vi":  {".", "!", "?", ":", ";", "…", "。"},
        "en":  {".", "!", "?", ":", ";", "…"},
        "ja":  {"。", "！", "？", "…", "．"},
        "ko":  {".", "!", "?", ":", ";", "…", "。"},
        "tl":  {".", "!", "?", ":", ";", "…"},
    }
    PHRASE_ENDINGS = {
        "vi":  {",", "、", "，"},
        "en":  {","},
        "ja":  {"、", "，", "：", "；", "・"},
        "ko":  {",", "、", "，"},
        "tl":  {","},
    }

    def __init__(self, cfg: dict, progress_cb: Callable[[str], None] | None = None):
        whisper_cfg = cfg.get("whisper", {})
        self._model_name: str = whisper_cfg.get("model", "medium")
        self._language: str = whisper_cfg.get("language", "auto")
        seg_cfg = cfg.get("segmentation", {})
        self._min_dur: float = seg_cfg.get("min_duration", 2.0)
        self._max_dur: float = seg_cfg.get("max_duration", 12.0)
        self._model = None  # lazy load
        self._cb = progress_cb or (lambda msg: log.debug(msg))

    # ── Public ────────────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str, language: str | None = None) -> list[Segment]:
        """
        Transcribe audio and return list of Segments.
        language overrides config if provided.
        """
        lang = language or self._language
        if lang == "auto":
            lang = self._detect_language(audio_path)
            self._cb(f"Detected language: {lang}")

        self._cb(f"Loading Whisper model '{self._model_name}'…")
        model = self._load_model()

        self._cb(f"Transcribing with Whisper ({lang})…")
        options: dict = {"word_timestamps": True, "verbose": False}
        if lang and lang != "auto":
            options["language"] = lang

        result = model.transcribe(audio_path, **options)

        # Collect word-level segments
        word_segs: list[dict] = []
        for seg in result.get("segments", []):
            word_segs.extend(seg.get("words", []))

        if not word_segs:
            self._cb("⚠ No word-level timestamps found — using sentence segments")
            return self._fallback_segments(result)

        self._cb(f"Got {len(word_segs)} words, segmenting…")
        word_segs = self._improve_boundaries(word_segs)
        segments = self._group_into_segments(word_segs, lang)
        self._cb(f"✓ Created {len(segments)} segments")
        return segments

    # ── Private: model ────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_name)
        return self._model

    def _detect_language(self, audio_path: str) -> str:
        import whisper
        try:
            model = whisper.load_model("base")
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected = max(probs, key=probs.get)
            supported = {"vi", "en", "ja", "ko", "tl"}
            return detected if detected in supported else "en"
        except Exception as exc:
            log.warning("Language detection failed: %s — defaulting to en", exc)
            return "en"

    # ── Private: segmentation (ported from tts_project) ───────────────────────

    def _improve_boundaries(self, words: list[dict]) -> list[dict]:
        """Improve word boundary detection (ported 1:1 from build_dataset_app_final.py)."""
        improved: list[dict] = []
        for i, word in enumerate(words):
            w = word.copy()
            # tts_project uses word_segments[i-1] (original list), NOT improved[-1]
            if i > 0:
                prev = words[i - 1]  # original prev word, matching tts_project exactly
                gap = w["start"] - prev["end"]
                if gap > 0.1:  # gap > 100ms → add 50ms padding
                    w["start"] = prev["end"] + 0.05
            # Ensure minimum word duration of 100ms
            if w["end"] - w["start"] < 0.1:
                w["end"] = w["start"] + 0.1
            improved.append(w)
        return improved

    def _group_into_segments(self, words: list[dict], lang: str) -> list[Segment]:
        sentence_ends = self.SENTENCE_ENDINGS.get(lang, self.SENTENCE_ENDINGS["en"])
        phrase_ends   = self.PHRASE_ENDINGS.get(lang, self.PHRASE_ENDINGS["en"])
        min_dur = self._min_dur
        max_dur = self._max_dur

        grouped: list[Segment] = []
        current_words: list[dict] = []
        current_start = words[0]["start"]

        for i, word in enumerate(words):
            current_words.append(word)
            duration = word["end"] - current_start
            word_text = word["word"].strip()
            is_last = i == len(words) - 1
            has_sent = any(p in word_text for p in sentence_ends)
            has_phr  = any(p in word_text for p in phrase_ends)

            should_end = (
                (has_sent and duration >= min_dur)
                or (has_phr and duration >= max(min_dur, 4.0))
                or duration >= max_dur
                or is_last
            )

            if should_end and current_words:
                seg = Segment(
                    start=current_words[0]["start"],
                    end=current_words[-1]["end"],
                    text=" ".join(w["word"].strip() for w in current_words),
                    words=list(current_words),
                )
                # Only keep if long enough (or last)
                min_threshold = min_dur if lang == "vi" else min_dur * 0.7
                if seg.duration >= min_threshold or is_last:
                    grouped.append(seg)
                    self._cb(f"  ✓ Segment {seg.duration:.1f}s: {seg.text[:40]}…")
                elif grouped:
                    # Merge with previous
                    prev = grouped[-1]
                    merged_dur = seg.end - prev.start
                    if merged_dur <= max_dur * 1.2:
                        prev.words.extend(seg.words)
                        prev.end = seg.end
                        prev.text = " ".join(w["word"].strip() for w in prev.words)

                current_words = []
                if not is_last:
                    current_start = words[i + 1]["start"]

        return self._cleanup_segments(grouped, lang)

    def _cleanup_segments(self, segs: list[Segment], lang: str) -> list[Segment]:
        """Dispatch to per-language cleanup (ported from build_dataset_app_final.py)."""
        if lang == "en":
            return self._cleanup_english(segs)
        elif lang == "ja":
            return self._cleanup_japanese(segs)
        elif lang == "tl":
            return self._cleanup_filipino(segs)
        elif lang == "ko":
            return self._cleanup_korean(segs)
        else:  # vi and fallback
            return self._cleanup_vietnamese(segs)

    # ── Per-language cleanup (ported 1:1 from build_dataset_app_final.py) ─────

    def _cleanup_vietnamese(self, segs: list[Segment]) -> list[Segment]:
        sentence_endings = {".", "!", "?", ":", ";", "…", "。"}
        final: list[Segment] = []
        for seg in segs:
            ends_sent = any(seg.text.endswith(p) for p in sentence_endings)
            if seg.duration >= self._min_dur or ends_sent:
                final.append(seg)
            elif final:
                prev = final[-1]
                merged = seg.end - prev.start
                if merged <= self._max_dur:
                    prev.words.extend(seg.words)
                    prev.end = seg.end
                    prev.text = " ".join(w["word"].strip() for w in prev.words)
                else:
                    final.append(seg)
            else:
                final.append(seg)
        return final

    def _cleanup_english(self, segs: list[Segment]) -> list[Segment]:
        sentence_endings = {".", "!", "?", ":", ";", "…"}
        final: list[Segment] = []
        for i, seg in enumerate(segs):
            is_last = i == len(segs) - 1
            ends_sent = any(seg.text.endswith(p) for p in sentence_endings)
            if seg.duration >= self._min_dur * 0.8 or ends_sent or is_last:
                final.append(seg)
            elif final:
                prev = final[-1]
                merged = seg.end - prev.start
                if merged <= self._max_dur * 1.3:
                    prev.words.extend(seg.words)
                    prev.end = seg.end
                    prev.text = " ".join(w["word"].strip() for w in prev.words)
                else:
                    final.append(seg)
            else:
                final.append(seg)
        return final

    def _cleanup_japanese(self, segs: list[Segment]) -> list[Segment]:
        strong = {"。", "！", "？", "…", "．"}
        weak   = {"、", "，", "・"}
        closers = '」』）】》〉〙〗〟\'"'

        def strip_c(s): return s.rstrip(closers)
        def ends_strong(t): s = strip_c(t.strip()); return bool(s) and s[-1] in strong
        def ends_weak(t):   s = t.strip(); return bool(s) and s[-1] in weak

        final: list[Segment] = []
        for i, seg in enumerate(segs):
            is_last = i == len(segs) - 1
            ok = seg.duration >= self._min_dur * 0.9
            if ok or ends_strong(seg.text) or is_last:
                final.append(seg)
                continue
            if final:
                prev = final[-1]
                merged = seg.end - prev.start
                limit = self._max_dur * 1.15 if ends_weak(prev.text) else self._max_dur * 1.05
                if merged <= limit:
                    prev.words.extend(seg.words)
                    prev.end = seg.end
                    prev.text = "".join(w["word"].strip() for w in prev.words)
                else:
                    final.append(seg)
            else:
                final.append(seg)
        return final

    def _cleanup_filipino(self, segs: list[Segment]) -> list[Segment]:
        sentence_endings = {".", "!", "?", ":", ";", "…"}
        final: list[Segment] = []
        for seg in segs:
            ends_sent = any(seg.text.endswith(p) for p in sentence_endings)
            if seg.duration >= self._min_dur or ends_sent:
                final.append(seg)
            elif final:
                prev = final[-1]
                merged = seg.end - prev.start
                if merged <= self._max_dur:
                    prev.words.extend(seg.words)
                    prev.end = seg.end
                    prev.text = " ".join(w["word"].strip() for w in prev.words)
                else:
                    final.append(seg)
            else:
                final.append(seg)
        return final

    def _cleanup_korean(self, segs: list[Segment]) -> list[Segment]:
        strong = {".", "!", "?", "…", "。", "！", "？"}
        weak   = {",", ";", ":", "、", "，", "~", "～"}
        closers = '」』）】》〉〙〗〟'

        def strip_c(s): return s.rstrip(closers)
        def ends_strong(t): s = strip_c(t.strip()); return bool(s) and s[-1] in strong
        def ends_weak(t):   s = t.strip(); return bool(s) and s[-1] in weak
        def ends_kor(t):
            s = strip_c(t.strip())
            # Full list from tts_project build_dataset_app_final.py
            for e in ["다", "요", "죠", "네", "데", "지", "까", "나", "니", "야",
                      "군", "구나", "군요", "네요", "죠요", "거든", "거든요"]:
                if s.endswith(e):
                    return True
            return False

        final: list[Segment] = []
        for i, seg in enumerate(segs):
            is_last = i == len(segs) - 1
            ok = seg.duration >= self._min_dur * 0.85
            if ok or ends_strong(seg.text) or ends_kor(seg.text) or is_last:
                final.append(seg)
                continue
            if final:
                prev = final[-1]
                merged = seg.end - prev.start
                limit = self._max_dur * 1.2 if (ends_weak(prev.text) or not ends_kor(prev.text)) else self._max_dur * 1.1
                if merged <= limit:
                    prev.words.extend(seg.words)
                    prev.end = seg.end
                    prev.text = " ".join(w["word"].strip() for w in prev.words)
                else:
                    final.append(seg)
            else:
                final.append(seg)
        return final

    def _fallback_segments(self, result: dict) -> list[Segment]:
        """Fallback when word timestamps unavailable: use sentence segments."""
        segs = []
        for s in result.get("segments", []):
            seg = Segment(
                start=float(s.get("start", 0)),
                end=float(s.get("end", 0)),
                text=s.get("text", "").strip(),
            )
            if seg.duration > 0 and seg.text:
                segs.append(seg)
        return segs
