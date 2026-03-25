"""
src/normalizer.py — Text normalization dispatcher

Delegates to the normalizer classes copied from tts_project.
Falls back to basic cleaning if normalizer is unavailable.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .logger import get_logger

log = get_logger("normalizer")

# ── Try to import tts_project normalizers ─────────────────────────────────────
# We add tts_project to the path so we can reuse its normalizers directly
_TTS_PROJECT_DIR = Path(__file__).parent.parent.parent / "tts_project"
if _TTS_PROJECT_DIR.exists() and str(_TTS_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_TTS_PROJECT_DIR))

try:
    from normalize_vi import OptimizedVietnameseTTSNormalizer
    _VI_NORM = OptimizedVietnameseTTSNormalizer()
    log.debug("Loaded Vietnamese normalizer")
except ImportError:
    _VI_NORM = None
    log.warning("normalize_vi not found — Vietnamese normalization will be basic")

try:
    from normalize_jp import OptimizedJapaneseTTSNormalizer
    _JA_NORM = OptimizedJapaneseTTSNormalizer()
except ImportError:
    _JA_NORM = None

try:
    from normalize_ko import OptimizedKoreanTTSNormalizer
    _KO_NORM = OptimizedKoreanTTSNormalizer()
except ImportError:
    _KO_NORM = None

try:
    from normalize_fil import OptimizedFilipinoTTSNormalizer
    _FIL_NORM = OptimizedFilipinoTTSNormalizer()
except ImportError:
    _FIL_NORM = None

try:
    from TTS.tts.utils.text.cleaners import english_cleaners as _en_cleaners
except ImportError:
    _en_cleaners = None


class TextNormalizer:
    """Normalize text for TTS training based on language."""

    def normalize(self, text: str, language: str) -> str:
        """Return normalized text for given language."""
        text = text.lower().strip()

        if language == "vi" and _VI_NORM:
            return _VI_NORM.normalize_for_tts(text)
        elif language == "ja" and _JA_NORM:
            return _JA_NORM.normalize_for_tts(text)
        elif language == "ko" and _KO_NORM:
            return _KO_NORM.normalize_for_tts(text)
        elif language == "tl" and _FIL_NORM:
            return _FIL_NORM.normalize_for_tts(text)
        elif language == "en":
            return self._normalize_english(text)
        else:
            return self._basic_normalize(text)

    def clean_reference(self, text: str, language: str) -> str:
        """Clean subtitle/reference text before alignment."""
        if language == "vi" and _VI_NORM:
            return _VI_NORM.clean_reference_input_text(text)
        elif language == "ja" and _JA_NORM:
            return _JA_NORM.clean_reference_input_text_ja(text)
        elif language == "ko" and _KO_NORM:
            return _KO_NORM.clean_reference_input_text_kor(text)
        elif language == "tl" and _FIL_NORM:
            return _FIL_NORM.clean_reference_input_text_fil(text)
        else:
            return self._basic_clean_reference(text)

    def _normalize_english(self, text: str) -> str:
        if _en_cleaners:
            try:
                return _en_cleaners(text).strip()
            except Exception as exc:
                log.warning("english_cleaners failed: %s", exc)
        return self._basic_normalize(text)

    @staticmethod
    def _basic_normalize(text: str) -> str:
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _basic_clean_reference(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)   # Remove HTML tags
        text = re.sub(r"\s+", " ", text)
        return text.strip()
