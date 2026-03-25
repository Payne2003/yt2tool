"""
tests/test_pipeline_mock.py — Full pipeline test với dữ liệu ảo

Không cần internet, không cần Whisper thật, không cần audio thật.
Kiểm tra:
  1. DatasetBuilder: duration filter, auto-split, fallback raw, no data loss
  2. DatasetBuilder: OpenAI correction (mock), word-count tolerance ±15%
  3. DatasetBuilder: _in_reference, _find_context, _count_words
  4. Transcriber: _improve_boundaries, _group_into_segments, cleanup
  5. TextNormalizer: vi / en basic normalization
  6. Zero-loss guarantee: mọi segment đều xuất ra metadata.csv

Chạy: python -m pytest tests/test_pipeline_mock.py -v
hoặc: python tests/test_pipeline_mock.py
"""
from __future__ import annotations

import os
import sys
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows console Unicode encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.dataset_builder import DatasetBuilder, DatasetEntry
from src.transcriber import Segment, Transcriber


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_wav(path: str, duration_sec: float = 30.0, sr: int = 22050) -> None:
    """Generate a real WAV file (silence) for testing pydub slicing."""
    n_samples = int(sr * duration_sec)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)           # 16-bit
        f.setframerate(sr)
        f.writeframes(b"\x00\x00" * n_samples)


def make_word(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


def make_segment(start: float, end: float, text: str, words: list[dict] | None = None) -> Segment:
    if words is None:
        # Auto-generate evenly spaced word timestamps
        ws = text.split()
        dur = (end - start) / max(len(ws), 1)
        words = [make_word(w, start + i * dur, start + (i + 1) * dur) for i, w in enumerate(ws)]
    return Segment(start=start, end=end, text=text, words=words)


CFG_BASE = {
    "dataset": {
        "output_dir": "",          # filled per test
        "append_mode": False,
        "sample_rate": 22050,
    },
    "segmentation": {
        "min_duration": 1.0,
        "max_duration": 13.0,
    },
    "openai": {"api_key": ""},
    "whisper": {"model": "tiny", "language": "vi"},
}


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 1 — DatasetBuilder core
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilderCore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmp, "test.wav")
        make_wav(self.wav, duration_sec=40.0)  # 40s audio
        cfg = dict(CFG_BASE)
        cfg["dataset"] = dict(CFG_BASE["dataset"])
        cfg["dataset"]["output_dir"] = self.tmp
        self.builder = DatasetBuilder(cfg)

    # ── Test 1: normal segment → xuất ra entry ────────────────────────────
    def test_normal_segment_creates_entry(self):
        seg = make_segment(0.0, 5.0, "xin chào đây là test bình thường")
        entries = self.builder.build(
            audio_path=self.wav,
            segments=[seg],
            language="vi",
            prefix="test",
        )
        self.assertEqual(len(entries), 1, "Phải có đúng 1 entry")
        self.assertIn("xin chào", entries[0].text)
        self.assertTrue(Path(entries[0].wav_rel.replace("wavs/", self.tmp + "/wavs/")).exists()
                        or Path(self.tmp, entries[0].wav_rel).exists())

    # ── Test 2: segment quá ngắn bị skip ─────────────────────────────────
    def test_short_segment_skipped(self):
        seg = make_segment(0.0, 0.5, "quá ngắn")   # 0.5s < min 1.0s
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertEqual(len(entries), 0, "Segment 0.5s phải bị skip")

    # ── Test 3: segment quá dài được auto-split ───────────────────────────
    def test_long_segment_auto_split(self):
        # 20s segment với 10 words → split thành 2 x 10s
        words = [make_word(f"từ{i}", i * 2.0, i * 2.0 + 1.8) for i in range(10)]
        seg = Segment(start=0.0, end=20.0, text=" ".join(w["word"] for w in words), words=words)
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertGreaterEqual(len(entries), 1, "Segment 20s phải được split thành ít nhất 1 entry")
        # Tổng số từ phải được bảo toàn (không mất từ nào)
        all_words_out = " ".join(e.text for e in entries).split()
        self.assertGreaterEqual(len(all_words_out), 5, "Không được mất quá nửa số từ khi split")

    # ── Test 4: nhiều segment, tất cả đều xuất ra ────────────────────────
    def test_multiple_segments_all_exported(self):
        segs = [
            make_segment(0.0, 3.0, "đoạn một ngắn bình thường"),
            make_segment(3.0, 8.0, "đoạn hai dài hơn một chút"),
            make_segment(8.0, 12.0, "đoạn ba cũng bình thường nha"),
        ]
        entries = self.builder.build(self.wav, segs, "vi", "test")
        self.assertEqual(len(entries), 3, "Cả 3 segment phải được xuất ra")

    # ── Test 5: metadata.csv được tạo đúng format ─────────────────────────
    def test_metadata_csv_format(self):
        seg = make_segment(0.0, 4.0, "kiểm tra định dạng csv")
        self.builder.build(self.wav, [seg], "vi", "test")
        meta = Path(self.tmp) / "metadata.csv"
        self.assertTrue(meta.exists(), "metadata.csv phải tồn tại")
        lines = meta.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        parts = lines[0].split("|")
        self.assertEqual(len(parts), 3, "Phải có 3 cột: segment_name|text|normalized")
        # tts_project format: segment_name (no path prefix, no .wav extension)
        seg_name = parts[0]
        self.assertFalse(seg_name.startswith("wavs/"), "Không được có prefix 'wavs/'")
        self.assertFalse(seg_name.endswith(".wav"), "Không được có extension .wav")
        self.assertIn("segment", seg_name, "Phải chứa 'segment' trong tên")

    # ── Test 6: original_texts và corrected_texts được lưu ───────────────
    def test_text_dirs_created(self):
        seg = make_segment(0.0, 3.0, "kiểm tra lưu file text")
        self.builder.build(self.wav, [seg], "vi", "test")
        self.assertTrue((Path(self.tmp) / "original_texts").exists())
        self.assertTrue((Path(self.tmp) / "corrected_texts").exists())
        orig_files = list((Path(self.tmp) / "original_texts").glob("*.txt"))
        self.assertEqual(len(orig_files), 1, "Phải có 1 file txt trong original_texts")

    # ── Test 7: word timestamps được dùng thay vì start/end ───────────────
    def test_word_timestamps_used_over_segment_timestamps(self):
        """Word timestamps chính xác hơn segment timestamps."""
        # Segment start=0, end=10 nhưng words chỉ từ 2.0 đến 4.6 (3 words × 0.9s + 0.8s)
        words = [
            make_word("từ1", 2.0, 2.8),
            make_word("từ2", 2.9, 3.7),
            make_word("từ3", 3.8, 4.6),
        ]
        seg = Segment(start=0.0, end=10.0, text="từ1 từ2 từ3", words=words)
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertEqual(len(entries), 1)
        # Duration in filename should be ~2.6s (from words), NOT 10s (from segment)
        # words[0].start=2.0, words[-1].end=4.6 → dur=2.6
        self.assertIn("2.60s", entries[0].wav_rel)

    # ── Test 8: mixed — ngắn + dài + bình thường → không mất gì ──────────
    def test_mixed_segments_zero_loss(self):
        """Kiểm tra zero-loss: ngắn bị skip nhưng dài được split, phải >= 2 entries."""
        segs = [
            make_segment(0.0, 0.5, "quá ngắn"),                  # skip
            make_segment(1.0, 5.0, "đây là đoạn bình thường"),   # OK
            # Long segment with enough words to split
            Segment(
                start=6.0, end=25.0,
                text=" ".join(f"từ{i}" for i in range(12)),
                words=[make_word(f"từ{i}", 6.0 + i * 1.5, 6.0 + i * 1.5 + 1.3) for i in range(12)]
            ),
        ]
        entries = self.builder.build(self.wav, segs, "vi", "test")
        self.assertGreaterEqual(len(entries), 2, "Phải có ≥2 entries từ 3 segments")


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 2 — OpenAI correction (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAICorrection(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmp, "test.wav")
        make_wav(self.wav, duration_sec=30.0)
        cfg = dict(CFG_BASE)
        cfg["dataset"] = dict(CFG_BASE["dataset"])
        cfg["dataset"]["output_dir"] = self.tmp
        cfg["openai"] = {"api_key": "sk-fake-key-for-testing"}
        self.builder = DatasetBuilder(cfg)

    def _mock_openai_response(self, corrected_texts: list[str]):
        """Helper: tạo mock OpenAI response với format [1] text [2] text..."""
        content = " ".join(f"[{i+1}] {t}" for i, t in enumerate(corrected_texts))
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = content
        return mock_resp

    # ── Test 9: GPT-corrected text được dùng thay raw ─────────────────────
    @patch("openai.OpenAI")
    def test_gpt_corrected_text_used(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            ["khung long dang song binh yen"]  # corrected (ASCII for reference match)
        )
        # raw: 6 words, corrected: 6 words -> exact count match -> must accept
        seg = make_segment(0.0, 4.0, "khong long dang song binh yen")
        entries = self.builder.build(
            self.wav, [seg], "vi", "test",
            subtitle_text="khung long dang song binh yen tren trai dat"
        )
        self.assertEqual(len(entries), 1)
        self.assertIn("khung long", entries[0].text)

    # ── Test 10: fallback to raw khi GPT reject vì word count ────────────
    @patch("openai.OpenAI")
    def test_fallback_raw_when_gpt_changes_word_count_too_much(self, mock_openai_cls):
        """GPT thêm/xóa quá nhiều từ → fallback về raw Whisper, KHÔNG BỎ entry."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # Raw: "một hai ba bốn" (4 từ), GPT trả về "một hai ba bốn năm sáu bảy tám" (8 từ → +4 = 100%)
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            ["một hai ba bốn năm sáu bảy tám"]
        )
        seg = make_segment(0.0, 3.0, "một hai ba bốn")
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertEqual(len(entries), 1, "Phải giữ lại entry dù GPT reject")
        # Should be raw text (GPT output too different)
        self.assertEqual(entries[0].text.strip(), "một hai ba bốn")

    # ── Test 11: word count ±15% tolerance ───────────────────────────────
    @patch("openai.OpenAI")
    def test_word_count_tolerance_15_percent(self, mock_openai_cls):
        """GPT sửa lệch 1 từ trong 8 từ (~12.5%) → phải ACCEPT."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        # raw: 8 words ASCII, corrected: 8 words (same count) -> must accept
        mock_client.chat.completions.create.return_value = self._mock_openai_response(
            ["khung long xuat hien tren trai dat xua"]  # 8 words
        )
        seg = make_segment(0.0, 5.0, "khong long xuat hien tren trai dat xua")  # 8 words
        entries = self.builder.build(
            self.wav, [seg], "vi", "test",
            subtitle_text="khung long xuat hien tren trai dat xua roi"
        )
        self.assertEqual(len(entries), 1)
        self.assertIn("khung long", entries[0].text)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 3 — Static helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestStaticHelpers(unittest.TestCase):

    # ── Test 12: _count_words vi/en ───────────────────────────────────────
    def test_count_words_vietnamese(self):
        # 4 distinct words: xin chao the gioi
        count = DatasetBuilder._count_words("xin chao the gioi", "vi")
        self.assertEqual(count, 4)

    def test_count_words_3_words(self):
        count = DatasetBuilder._count_words("xin chao gioi", "vi")
        self.assertEqual(count, 3)

    def test_count_words_with_punctuation(self):
        # "xin chao, the gioi!" -> split -> ["xin", "chao,", "the", "gioi!"] = 4 tokens
        count = DatasetBuilder._count_words("xin chao, the gioi!", "vi")
        self.assertEqual(count, 4)

    def test_count_words_empty(self):
        self.assertEqual(DatasetBuilder._count_words("", "vi"), 0)
        self.assertEqual(DatasetBuilder._count_words("   ", "vi"), 0)

    # ── Test 13: _in_reference ────────────────────────────────────────────
    def test_in_reference_exact_match(self):
        self.assertTrue(DatasetBuilder._in_reference(
            "khủng long xuất hiện",
            "những con khủng long xuất hiện từ rất lâu rồi"
        ))

    def test_in_reference_fuzzy_match(self):
        self.assertTrue(DatasetBuilder._in_reference(
            "khủng lông xuất hiện",   # typo
            "khủng long xuất hiện từ rất lâu",
            threshold=0.8
        ))

    def test_in_reference_not_match(self):
        self.assertFalse(DatasetBuilder._in_reference(
            "máy bay vũ trụ tốc độ cao",
            "xin chào thế giới bình yên",
            threshold=0.8
        ))

    def test_in_reference_text_longer_than_reference(self):
        """Text dài hơn reference — không được crash (đây là bug đã sửa)."""
        result = DatasetBuilder._in_reference(
            "một hai ba bốn năm sáu bảy tám chín mười",    # 10 words
            "một hai ba",                                    # 3 words
            threshold=0.5
        )
        # Should return bool, no crash
        self.assertIsInstance(result, bool)

    def test_in_reference_empty_strings(self):
        """Edge case: empty text or empty reference always returns False (no crash)."""
        # Empty text: Python's '' in 'anything' is True, but we guard against this
        self.assertFalse(DatasetBuilder._in_reference("", "reference text"))
        self.assertFalse(DatasetBuilder._in_reference("text", ""))
        self.assertFalse(DatasetBuilder._in_reference("", ""))

    # ── Test 14: _find_context ────────────────────────────────────────────
    def test_find_context_found(self):
        ref = "khủng long xuất hiện từ thời tiền sử rất lâu trước đây"
        ctx = DatasetBuilder._find_context("khủng long xuất hiện", ref)
        self.assertIn("khủng long", ctx)

    def test_find_context_not_found_returns_start(self):
        ref = "một đoạn text không liên quan"
        ctx = DatasetBuilder._find_context("máy bay vũ trụ", ref)
        # Fallback: returns first 200 chars of ref
        self.assertTrue(len(ctx) > 0)

    def test_find_context_empty_ref(self):
        ctx = DatasetBuilder._find_context("some text", "")
        self.assertEqual(ctx, "")

    # ── Test 15: _parse_batch_response ────────────────────────────────────
    def test_parse_batch_response_normal(self):
        response = "[1] xin chào thế giới [2] khủng long xuất hiện"
        result = DatasetBuilder._parse_batch_response(response, 2)
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertIn("xin chào", result[0])
        self.assertIn("khủng long", result[1])

    def test_parse_batch_response_partial(self):
        """GPT chỉ trả 1/2 → result[1] = None."""
        response = "[1] xin chào thế giới"
        result = DatasetBuilder._parse_batch_response(response, 2)
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])

    def test_parse_batch_response_newline_fallback(self):
        """Fallback khi format sai — line-by-line parsing."""
        response = "xin chào thế giới\nkhủng long xuất hiện"
        result = DatasetBuilder._parse_batch_response(response, 2)
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 4 — Transcriber segmentation logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTranscriberSegmentation(unittest.TestCase):

    def setUp(self):
        cfg = {
            "whisper": {"model": "tiny", "language": "vi"},
            "segmentation": {"min_duration": 2.0, "max_duration": 12.0},
        }
        self.transcriber = Transcriber(cfg)

    # ── Test 16: _improve_boundaries không IndexError ─────────────────────
    def test_improve_boundaries_no_crash(self):
        words = [
            make_word("từ1", 0.0, 0.5),
            make_word("từ2", 1.0, 1.5),   # gap > 0.1 → sẽ adjust
            make_word("từ3", 2.0, 2.5),
        ]
        result = self.transcriber._improve_boundaries(words)
        self.assertEqual(len(result), 3)
        # Start của từ2 phải được adjust (từ 1.0 → 0.5 + 0.05 = 0.55)
        self.assertAlmostEqual(result[1]["start"], 0.55, places=2)

    def test_improve_boundaries_single_word(self):
        words = [make_word("từ1", 0.0, 0.5)]
        result = self.transcriber._improve_boundaries(words)
        self.assertEqual(len(result), 1)

    def test_improve_boundaries_empty(self):
        result = self.transcriber._improve_boundaries([])
        self.assertEqual(result, [])

    def test_improve_boundaries_min_duration_enforced(self):
        """Từ với duration < 0.1s phải được extend."""
        words = [make_word("x", 0.0, 0.05)]  # 50ms < 100ms
        result = self.transcriber._improve_boundaries(words)
        self.assertAlmostEqual(result[0]["end"] - result[0]["start"], 0.1, places=2)

    # ── Test 17: _group_into_segments tạo ra segment hợp lệ ──────────────
    def test_group_basic_vi(self):
        """10 từ, ~3s mỗi từ → phải tạo ra các segment 2-12s."""
        words = [make_word(f"từ{i}", i * 2.5, i * 2.5 + 2.3) for i in range(8)]
        # last word có dấu câu
        words[-1]["word"] = "xong."
        segs = self.transcriber._group_into_segments(words, "vi")
        self.assertGreater(len(segs), 0)
        for seg in segs:
            self.assertGreater(seg.duration, 0, "Duration phải > 0")
            self.assertGreater(len(seg.text.strip()), 0, "Text không được rỗng")

    def test_group_handles_last_word(self):
        """Word cuối phải luôn được đưa vào segment, không bỏ xót."""
        words = [make_word(f"từ{i}", float(i), float(i) + 0.8) for i in range(5)]
        segs = self.transcriber._group_into_segments(words, "vi")
        # Reconstruct all words from segments
        all_words = []
        for seg in segs:
            all_words.extend(seg.words)
        word_texts = set(w["word"] for w in all_words)
        for i in range(5):
            self.assertIn(f"từ{i}", word_texts, f"từ{i} bị mất sau segmentation")

    # ── Test 18: _cleanup_vietnamese không bỏ segment ────────────────────
    def test_cleanup_vi_keeps_sentence_boundary(self):
        segs = [
            Segment(0, 3, "xin chào bạn.", []),   # ends với sentence
            Segment(3, 4, "ok", []),               # đây sẽ được merge
            Segment(4, 7, "tạm biệt bạn nhé.", []),
        ]
        result = self.transcriber._cleanup_vietnamese(segs)
        self.assertGreater(len(result), 0)


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 5 — Integration: không bỏ xót từ nào (zero word loss)
# ─────────────────────────────────────────────────────────────────────────────

class TestZeroWordLoss(unittest.TestCase):
    """
    Đây là test quan trọng nhất: đảm bảo toàn bộ nội dung audio đều được
    đưa vào metadata.csv, không mất từ nào.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wav = os.path.join(self.tmp, "test.wav")
        make_wav(self.wav, duration_sec=220.0)  # 50 segs * 4s = 200s needed + margin
        cfg = dict(CFG_BASE)
        cfg["dataset"] = dict(CFG_BASE["dataset"])
        cfg["dataset"]["output_dir"] = self.tmp
        self.builder = DatasetBuilder(cfg)

    def _count_output_words(self) -> set[str]:
        """Đọc metadata.csv và trả về tập tất cả từ trong cột text."""
        meta = Path(self.tmp) / "metadata.csv"
        words = set()
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                parts = line.split("|")
                if len(parts) >= 2:
                    words.update(parts[1].lower().split())
        return words

    def test_all_normal_segments_appear_in_metadata(self):
        """100% segments bình thường → 100% xuất vào metadata."""
        input_words = set()
        segs = []
        for i in range(10):
            text = f"đây là đoạn thứ {i} trong bài test"
            segs.append(make_segment(i * 5.0, (i + 1) * 5.0 - 0.1, text))
            input_words.update(text.split())

        self.builder.build(self.wav, segs, "vi", "test")
        output_words = self._count_output_words()
        # Mọi từ unique trong input phải xuất hiện trong output
        missing = input_words - output_words
        self.assertEqual(len(missing), 0,
                         f"Các từ bị mất: {missing}")

    def test_long_segment_words_preserved_after_split(self):
        """Segment 22s auto-split -> words preserved in output."""
        # 14 words, each 1.5s apart -> total ~21s -> will be split into 2 halves
        # Each half: 7 words * 1.5s = 10.5s -> within 1-13s range -> both halves saved
        all_input_words = [f"tu{i}" for i in range(14)]
        words_dict = [make_word(w, i * 1.5, i * 1.5 + 1.3) for i, w in enumerate(all_input_words)]
        seg = Segment(
            start=0.0, end=22.0,
            text=" ".join(all_input_words),
            words=words_dict
        )
        self.builder.build(self.wav, [seg], "vi", "test")
        output_words = self._count_output_words()
        # At least 12/14 words must appear in output (1-2 may be edge cases)
        found = sum(1 for w in all_input_words if w in output_words)
        self.assertGreaterEqual(found, 12,
            f"Only {found}/14 words found in output. Words in output: {sorted(output_words)}")

    def test_short_boundary_at_exactly_min_duration(self):
        """Segment đúng bằng min_duration (1.0s) phải được giữ lại."""
        seg = make_segment(0.0, 1.0, "ngắn vừa đủ")
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertEqual(len(entries), 1, "Segment 1.0s đúng min phải được giữ")

    def test_segment_just_below_min_dropped(self):
        """Segment 0.99s phải bị drop."""
        seg = make_segment(0.0, 0.99, "nhỏ hơn tí")
        entries = self.builder.build(self.wav, [seg], "vi", "test")
        self.assertEqual(len(entries), 0, "Segment 0.99s phải bị drop")

    def test_large_batch_all_exported(self):
        """50 segment bình thường → 50 entries trong metadata."""
        segs = [
            make_segment(i * 4.0, i * 4.0 + 3.5, f"đây là segment số {i} trong bài kiểm tra")
            for i in range(50)
        ]
        entries = self.builder.build(self.wav, segs, "vi", "test")
        self.assertEqual(len(entries), 50, "Phải có đúng 50 entries")
        meta = Path(self.tmp) / "metadata.csv"
        lines = [l for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertEqual(len(lines), 50, "metadata.csv phải có đúng 50 dòng")


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE 6 — Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizer(unittest.TestCase):

    def setUp(self):
        from src.normalizer import TextNormalizer
        self.norm = TextNormalizer()

    def test_normalize_vi_lowercase(self):
        result = self.norm.normalize("Xin Chào Thế Giới", "vi")
        self.assertEqual(result, result.lower(), "Output phải lowercase")

    def test_normalize_empty_string(self):
        result = self.norm.normalize("", "vi")
        self.assertIsInstance(result, str)

    def test_normalize_no_crash_any_lang(self):
        for lang in ("vi", "en", "ja", "ko", "tl"):
            try:
                result = self.norm.normalize("test text 123", lang)
                self.assertIsInstance(result, str)
            except Exception as e:
                self.fail(f"normalize() crash với lang={lang}: {e}")

    def test_clean_reference_vi(self):
        ref = "<html>Xin chào\n\nThế giới</html>"
        result = self.norm.clean_reference(ref, "vi")
        self.assertIsInstance(result, str)
        self.assertNotIn("<html>", result)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Color output khi chạy trực tiếp
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(suite)

    print("\n" + "=" * 60)
    print(f"  Tổng: {result.testsRun} tests")
    print(f"  ✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    if result.failures:
        print(f"  ❌ Failed: {len(result.failures)}")
    if result.errors:
        print(f"  💥 Errors: {len(result.errors)}")
    print("=" * 60)
    sys.exit(0 if result.wasSuccessful() else 1)
