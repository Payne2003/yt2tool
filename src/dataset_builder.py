"""
src/dataset_builder.py — Extract audio segments + write metadata.csv (LJSpeech format)

Fully ported from tts_project/build_dataset_app_final.py:
- Word-level timestamp priority for accurate slicing
- Duration safety check (< min_dur or > max_dur → skip or split)
- 16-bit / 22050Hz / mono export
- OpenAI batch correction with word-count + reference validation
- Latin-only ASCII filenames (no Vietnamese diacritics)
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydub import AudioSegment as PydubAudio

from .logger import get_logger
from .normalizer import TextNormalizer
from .transcriber import Segment

log = get_logger("dataset_builder")


@dataclass
class DatasetEntry:
    wav_rel: str       # relative path, e.g. "wavs/title_0001_2.5s.wav"
    text: str          # original / corrected transcription
    normalized: str    # normalized text for TTS training


class DatasetBuilder:
    """
    Given a list of Segments (with timestamps) and the source WAV,
    extract individual audio chunks and write metadata.csv.
    """

    def __init__(self, cfg: dict):
        ds_cfg = cfg.get("dataset", {})
        seg_cfg = cfg.get("segmentation", {})
        self._output_dir = Path(ds_cfg.get("output_dir", "output"))
        self._wavs_dir = self._output_dir / "wavs"
        self._metadata_path = self._output_dir / "metadata.csv"
        self._append = ds_cfg.get("append_mode", True)
        self._sample_rate = ds_cfg.get("sample_rate", 22050)
        self._min_dur: float = seg_cfg.get("min_duration", 1.0)
        self._max_dur: float = seg_cfg.get("max_duration", 13.0)
        self._wavs_dir.mkdir(parents=True, exist_ok=True)
        self._normalizer = TextNormalizer()
        self._openai_key: str = cfg.get("openai", {}).get("api_key", "")
        openai_cfg = cfg.get("openai", {})
        # Configurable correction thresholds
        self._wc_tolerance: float  = openai_cfg.get("word_count_tolerance", 0.20)  # ±20%
        self._ref_threshold: float = openai_cfg.get("reference_threshold",  0.70)  # fuzzy

    # ── Public ────────────────────────────────────────────────────────────────

    def build(
        self,
        audio_path: str,
        segments: list[Segment],
        language: str,
        prefix: str,
        subtitle_text: str = "",
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[DatasetEntry]:
        """
        Extract audio segments and create dataset entries.

        Upgrades:
          - Long segments (> max_dur) auto-split or saved as-is if 2+ words
          - OpenAI rejects → fallback to raw Whisper (zero data loss)
          - original_texts/ and corrected_texts/ saved per-segment
          - Word-count tolerance ±20% min=2 (configurable in config.yaml)
          - Reference threshold 0.70 (configurable in config.yaml)
        """
        cb = progress_cb or (lambda m: log.debug(m))

        cb(f"Loading audio: {audio_path}")
        audio = PydubAudio.from_wav(audio_path)
        audio = (audio
                 .set_frame_rate(self._sample_rate)
                 .set_channels(1)
                 .set_sample_width(2))   # 16-bit

        start_index = self._count_existing()
        cb(f"Existing entries: {start_index}")

        # ── Create text output dirs (like build_dataset_app_final.py) ─────────
        orig_dir = self._output_dir / "original_texts"
        corr_dir = self._output_dir / "corrected_texts"
        orig_dir.mkdir(parents=True, exist_ok=True)
        corr_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: audio/ dir is NOT created — wav files go directly to wavs/

        # ── Clean reference text ───────────────────────────────────────────────
        cleaned_ref = self._normalizer.clean_reference(subtitle_text, language) if subtitle_text else ""

        # ── Step 1: Extract audio slices ──────────────────────────────────────
        segment_paths: list[str] = []
        transcriptions: list[str] = []

        def _save_and_register(hs: float, he: float, ht: str) -> None:
            """Cut audio, save wav + original txt, register in lists."""
            hd = he - hs
            global_idx = start_index + len(segment_paths)  # index BEFORE append
            # Use Latin-only slug for filename (no Vietnamese diacritics)
            slug = DatasetBuilder._slugify(prefix)
            fname = f"{slug}_{global_idx:05d}_{hd:.2f}s.wav"
            wav_out = self._wavs_dir / fname
            audio[int(hs * 1000): int(he * 1000)].export(str(wav_out), format="wav")
            segment_paths.append(str(wav_out))             # append AFTER computing idx
            transcriptions.append(ht)
            (orig_dir / fname.replace(".wav", ".txt")).write_text(ht, encoding="utf-8")
            return global_idx, fname  # return for logging

        for i, seg in enumerate(segments):
            # Priority: use word-level timestamps (more accurate)
            if seg.words:
                t_start = seg.words[0]["start"]
                t_end   = seg.words[-1]["end"]
            else:
                t_start = seg.start
                t_end   = seg.end

            actual_dur = t_end - t_start
            txt = seg.text.strip()

            # ── Auto-split or save overlong segments ──────────────────────
            if actual_dur > self._max_dur:
                words = seg.words if seg.words else []
                if len(words) >= 4:
                    mid = len(words) // 2
                    parts = [
                        (words[0]["start"], words[mid - 1]["end"],
                         " ".join(w["word"].strip() for w in words[:mid])),
                        (words[mid]["start"], words[-1]["end"],
                         " ".join(w["word"].strip() for w in words[mid:])),
                    ]
                    for hs, he, ht in parts:
                        hd = he - hs
                        if self._min_dur <= hd <= self._max_dur:
                            gidx, fname = _save_and_register(hs, he, ht)
                            cb(f"  ✂ [{gidx:05d}] split {hd:.2f}s | {ht[:50]}")
                        else:
                            cb(f"  ⚠ Split half out of range {hd:.2f}s — skipped")
                elif len(words) >= 2:
                    # Too few words to split but has some content: save as-is
                    gidx, fname = _save_and_register(t_start, t_end, txt)
                    cb(f"  ✅ [{gidx:05d}] {actual_dur:.2f}s (long, saved as-is) | {txt[:50]}")
                else:
                    cb(f"  ⛔ Skip seg {i+1}: {actual_dur:.2f}s too long, only {len(words)} word(s)")
                continue

            # Skip too-short segments
            if actual_dur < self._min_dur:
                cb(f"⚠ Skipping segment {i+1}: {actual_dur:.2f}s < min {self._min_dur}s")
                continue

            gidx, fname = _save_and_register(t_start, t_end, txt)
            cb(f"  ✅ [{gidx:05d}] {actual_dur:.2f}s | {txt[:50]}")

        if not segment_paths:
            cb("⚠ No segments extracted.")
            return []

        # ── Step 2: OpenAI correction (optional) ──────────────────────────────
        # Returns dict {index: corrected_text} for accepted segments only
        corrected_map: dict[int, str] = {}

        if self._openai_key:
            cb(f"🔄 OpenAI correction for {len(transcriptions)} segments…")
            try:
                corrected_map = self._correct_with_openai(
                    transcriptions=transcriptions,
                    reference_text=cleaned_ref,
                    language=language,
                    cb=cb,
                )
            except Exception as exc:
                cb(f"❌ OpenAI correction failed: {exc} — using raw transcriptions")

        # ── Step 3: Normalize & write metadata ───────────────────────────────
        # NO data loss: if OpenAI rejected, use raw Whisper text
        entries: list[DatasetEntry] = []
        for idx, (wav_path, raw_text) in enumerate(zip(segment_paths, transcriptions)):
            final_text = corrected_map.get(idx, raw_text)  # fallback → raw
            wav_rel = "wavs/" + Path(wav_path).name
            normalized = self._normalizer.normalize(final_text, language)
            entry = DatasetEntry(wav_rel=wav_rel, text=final_text, normalized=normalized)
            entries.append(entry)
            # Save corrected text file
            txt_name = Path(wav_path).name.replace(".wav", ".txt")
            (corr_dir / txt_name).write_text(final_text, encoding="utf-8")

        self._write_metadata(entries)
        gpt_ok   = len(corrected_map)
        fallback = len(entries) - gpt_ok
        pct      = gpt_ok * 100 // len(entries) if entries else 0
        cb(f"✓ Wrote {len(entries)} entries → {self._metadata_path}")
        cb(f"  📊 GPT accepted: {gpt_ok} ({pct}%) | raw fallback: {fallback} | "
           f"total: {len(entries)}")
        return entries

    # ── OpenAI Batch Correction ───────────────────────────────────────────────

    def _correct_with_openai(
        self,
        transcriptions: list[str],
        reference_text: str,
        language: str,
        cb: Callable[[str], None],
    ) -> dict[int, str]:
        """
        Batch-correct Whisper transcriptions using OpenAI (gpt-4.1-nano).
        Ported from build_dataset_app_final.py:correct_with_openai_and_reference.

        Returns:
            dict {index: corrected_text} — only accepted segments.
            Rejected segments are NOT included → caller uses raw fallback.
        """
        import openai
        client = openai.OpenAI(api_key=self._openai_key)

        corrected_map: dict[int, str] = {}

        batch_size = 10
        total = len(transcriptions)
        total_batches = (total + batch_size - 1) // batch_size
        word_count_failures = 0
        ref_failures = 0
        api_failures = 0

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = transcriptions[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            batch_num = batch_start // batch_size + 1

            cb(f"🔄 Batch {batch_num}/{total_batches} (segs {batch_start+1}-{batch_end})")

            try:
                # Build batch context
                batch_data = []
                for idx, tr in zip(batch_indices, batch):
                    batch_data.append({
                        "transcription": tr,
                        "context": self._find_context(tr, reference_text),
                        "word_count": self._count_words(tr, language),
                        "original_index": idx,
                    })

                prompt = self._build_batch_prompt(batch_data, language)
                sys_msg, examples = self._get_system_and_examples(language)

                messages = [{"role": "system", "content": sys_msg}]
                messages.extend(examples)
                messages.append({"role": "user", "content": prompt})

                resp = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages,
                    max_tokens=1200,
                    temperature=0.1,
                )
                batch_corrected = self._parse_batch_response(
                    resp.choices[0].message.content, len(batch)
                )

                # Validate each segment
                for i, (orig, corrected) in enumerate(zip(batch, batch_corrected)):
                    seg_global = batch_indices[i]
                    if corrected is None:
                        api_failures += 1
                        cb(f"  ❌ Seg {seg_global+1}: parse error")
                        continue

                    orig_wc = batch_data[i]["word_count"]
                    corr_wc = self._count_words(corrected, language)

                    # Check 1: word count ±20% with min tolerance=2
                    tolerance = max(2, round(orig_wc * self._wc_tolerance))
                    if abs(corr_wc - orig_wc) > tolerance:
                        word_count_failures += 1
                        cb(f"  ↩ Seg {seg_global+1}: GPT count {orig_wc}→{corr_wc} "
                           f"(±{tolerance}) → raw fallback")
                        continue

                    # Check 2: reference match (skip for ja/ko)
                    if language not in ("ja", "ko") and reference_text.strip():
                        if not self._in_reference(
                            corrected, reference_text, threshold=self._ref_threshold
                        ):
                            ref_failures += 1
                            cb(f"  ↩ Seg {seg_global+1}: GPT not in reference → raw fallback")
                            continue

                    corrected_map[seg_global] = corrected
                    cb(f"  ✅ Seg {seg_global+1}: '{corrected[:40]}'")

                cb(f"  ✅ Batch {batch_num} done")

            except Exception as exc:
                api_failures += len(batch)
                cb(f"  ❌ Batch {batch_num} API error: {exc}")
                continue

        valid_count = len(corrected_map)
        pct_ok = valid_count * 100 // total if total else 0
        cb(f"📊 GPT correction: {valid_count}/{total} accepted ({pct_ok}%) | "
           f"wc_fail={word_count_failures} ref_fail={ref_failures} api_fail={api_failures} "
           f"→ all fallback to raw Whisper")
        return corrected_map

    # ── OpenAI helpers ────────────────────────────────────────────────────────

    def _build_batch_prompt(self, batch_data: list[dict], language: str) -> str:
        """Create batch correction prompt (ported from build_dataset_app_final.py)."""
        if language == "vi":
            prompt = "Hãy sửa lỗi cho các phiên âm Whisper sau đây. Trả về kết quả theo định dạng: [1] text_đã_sửa [2] text_đã_sửa [3] text_đã_sửa...\n\n"
            for i, d in enumerate(batch_data, 1):
                prompt += f"[{i}] Phiên âm: \"{d['transcription']}\"\n"
                if d["context"]:
                    prompt += f"    Text gốc liên quan: \"{d['context'][:150]}...\"\n"
                prompt += f"    Số từ phải giữ nguyên: {d['word_count']}\n\n"
        elif language == "ja":
            prompt = "以下のWhisper書き起こしを日本語として校正してください。出力は1行で、形式は次のとおりです: [1] 修正テキスト [2] 修正テキスト [3] 修正テキスト...\n"
            for i, d in enumerate(batch_data, 1):
                prompt += f"[{i}] 書き起こし: \"{d['transcription']}\"\n"
                if d["context"]:
                    prompt += f"    関連する原文/文脈: \"{d['context'][:150]}...\"\n"
                prompt += f"    維持すべき文字数（スペース・句読点を除く）: {d['word_count']}\n\n"
        elif language == "tl":
            prompt = ("Pakisuri at ayusin ang mga sumusunod na transcription mula sa Whisper bilang tamang Filipino/Tagalog. "
                      "Ibalik ang sagot sa isang linya, sa format na: [1] Naayos na Teksto [2] Naayos na Teksto [3] Naayos na Teksto...\n")
            for i, d in enumerate(batch_data, 1):
                prompt += f"[{i}] Transcription: \"{d['transcription']}\"\n"
                if d["context"]:
                    prompt += f"    Kaugnay na orihinal/konteksto: \"{d['context'][:150]}...\"\n"
                prompt += f"    Dapat panatilihin ang dami ng salita (hiwalay ayon sa whitespace): {d['word_count']}\n\n"
        elif language == "ko":
            prompt = "다음 Whisper 전사를 한국어 원문에 맞게 교정해주세요. 출력은 한 줄로, 형식은: [1] 교정된 텍스트 [2] 교정된 텍스트 [3] 교정된 텍스트...\n\n"
            for i, d in enumerate(batch_data, 1):
                prompt += f"[{i}] 전사: \"{d['transcription']}\"\n"
                if d["context"]:
                    prompt += f"    관련 원문/문맥: \"{d['context'][:150]}...\"\n"
                prompt += f"    유지해야 할 단어 수 (공백 기준): {d['word_count']}\n\n"
        else:  # English
            prompt = "Please correct the following Whisper transcriptions. Return results in format: [1] corrected_text [2] corrected_text [3] corrected_text...\n\n"
            for i, d in enumerate(batch_data, 1):
                prompt += f"[{i}] Transcription: \"{d['transcription']}\"\n"
                if d["context"]:
                    prompt += f"    Related original text: \"{d['context'][:150]}...\"\n"
                prompt += f"    Must maintain word count: {d['word_count']}\n\n"
        return prompt

    def _get_system_and_examples(self, language: str) -> tuple[str, list[dict]]:
        """Return system message + few-shot examples (ported from tts_project)."""
        if language == "vi":
            sys_msg = ("Bạn là chuyên gia tìm text gốc dựa trên phiên âm Whisper tiếng Việt. "
                       "Xử lý từng segment riêng biệt, giữ nguyên số từ cho mỗi segment.")
            examples = [
                {"role": "user", "content": (
                    "Hãy sửa lỗi cho các phiên âm Whisper sau đây. "
                    "Trả về kết quả theo định dạng: [1] text_đã_sửa [2] text_đã_sửa\n\n"
                    "[1] Phiên âm: \"luc này nhà họ nam cũng có không ít người\"\n"
                    "    Text gốc liên quan: \"Lúc này nhà họ Nam cũng có không ít người đến...\"\n"
                    "    Số từ phải giữ nguyên: 10\n\n"
                    "[2] Phiên âm: \"ô đen trang phuc thống nhất\"\n"
                    "    Text gốc liên quan: \"Đêm mưa, ô đen, trang phục thống nhất...\"\n"
                    "    Số từ phải giữ nguyên: 6"
                )},
                {"role": "assistant",
                 "content": "[1] Lúc này nhà họ Nam cũng có không ít người [2] ô đen, trang phục thống nhất"},
            ]
        elif language == "ja":
            sys_msg = ("ベトナム語のウィスパー音訳に基づいて原文を見つけるエキスパートです。"
                       "各セグメントを個別に処理し、各セグメントの単語数を一定に保ちます。")
            examples = [
                {"role": "user", "content": (
                    "以下のWhisper書き起こしを日本語として校正してください。出力は1行で、形式は: [1] 修正テキスト [2] 修正テキスト\n\n"
                    "[1] 書き起こし: \"かいしゃ は らいげつ しんしょうひん を はつばい します\"\n"
                    "    関連する原文/文脈: \"同社は来月、新商品を発売します。\"\n"
                    "    維持すべき語数: 7\n\n"
                    "[2] 書き起こし: \"にほん の けいざい は じょうしょう して います\"\n"
                    "    関連する原文/文脈: \"日本の経済は上昇しています。\"\n"
                    "    維持すべき語数: 7"
                )},
                {"role": "assistant",
                 "content": "[1] 会社 は 来月 新商品 を 発売 します [2] 日本 の 経済 は 上昇 して います"},
            ]
        elif language == "tl":
            sys_msg = ("Ikaw ay isang eksperto sa paghahanap ng orihinal na teksto batay sa "
                       "transcription ng Whisper sa wikang Filipino. Proseso ang bawat segment "
                       "nang hiwalay at panatilihin ang parehong bilang ng mga salita.")
            examples = [
                {"role": "user", "content": (
                    "Pakisuri at ayusin ang mga Whisper transcription. "
                    "Sagot sa format: [1] text [2] text\n\n"
                    "[1] Transcription: \"ang kompanya ay mag lulunsad sa sunod na buan\"\n"
                    "    Kaugnay na teksto: \"Ang kompanya ay maglulunsad sa susunod na buwan...\"\n"
                    "    Bilang ng salita: 8\n\n"
                    "[2] Transcription: \"ang artipicial inteligensiya ay may potesyal\"\n"
                    "    Kaugnay na teksto: \"Ang artipisyal na intelihensiya ay may potensyal...\"\n"
                    "    Bilang ng salita: 6"
                )},
                {"role": "assistant",
                 "content": "[1] ang kompanya ay maglulunsad sa susunod na buwan [2] ang artipisyal intelihensiya ay may potensyal"},
            ]
        elif language == "ko":
            sys_msg = ("당신은 한국어 Whisper 음성 전사를 기반으로 원문을 찾는 전문가입니다. "
                       "각 세그먼트를 개별적으로 처리하고 각 세그먼트의 단어 수를 동일하게 유지하세요.")
            examples = [
                {"role": "user", "content": (
                    "다음 Whisper 전사를 한국어 원문에 맞게 교정해주세요. "
                    "출력은 한 줄로, 형식은: [1] 교정된_텍스트 [2] 교정된_텍스트\n\n"
                    "[1] 전사: \"회 사 는 다 음 달 새 제 품 을 출 시 합 니 다\"\n"
                    "    관련 원문: \"회사는 다음 달 새 제품을 출시합니다.\"\n"
                    "    유지해야 할 단어 수: 8\n\n"
                    "[2] 전사: \"한 국 경 제 가 성 장 하 고 있 습 니 다\"\n"
                    "    관련 원문: \"한국 경제가 성장하고 있습니다.\"\n"
                    "    유지해야 할 단어 수: 7"
                )},
                {"role": "assistant",
                 "content": "[1] 회사 는 다음 달 새 제품 을 출시 합니다 [2] 한국 경제 가 성장 하고 있습니다"},
            ]
        else:  # English
            sys_msg = ("You are an expert at finding original text based on Whisper transcription. "
                       "Process each segment separately, maintain word count for each segment.")
            examples = [
                {"role": "user", "content": (
                    "Please correct the following Whisper transcriptions. "
                    "Return results in format: [1] corrected_text [2] corrected_text\n\n"
                    "[1] Transcription: \"the compeny is planing to launch\"\n"
                    "    Related original text: \"The company is planning to launch a new product...\"\n"
                    "    Must maintain word count: 6\n\n"
                    "[2] Transcription: \"artifical inteligence has the potencial\"\n"
                    "    Related original text: \"Artificial intelligence has the potential...\"\n"
                    "    Must maintain word count: 5"
                )},
                {"role": "assistant",
                 "content": "[1] The company is planning to launch [2] Artificial intelligence has the potential"},
            ]
        return sys_msg, examples

    @staticmethod
    def _parse_batch_response(response_text: str, expected_count: int) -> list[str | None]:
        """Parse [1] text [2] text... response from OpenAI."""
        pattern = r'\[(\d+)\]\s*([^\[]*?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, response_text, re.DOTALL)
        result: list[str | None] = [None] * expected_count
        for idx_str, text in matches:
            idx = int(idx_str) - 1
            text = text.strip().strip('"')
            if 0 <= idx < expected_count:
                result[idx] = text
        # Fallback: line-by-line
        if all(t is None for t in result):
            lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
            for j, line in enumerate(lines[:expected_count]):
                line = re.sub(r'^\[?\d+\]?\s*', '', line).strip().strip('"')
                if line:
                    result[j] = line
        return result

    @staticmethod
    def _count_words(text: str, language: str) -> int:
        """Count words/chars depending on language."""
        if language == "ja":
            return len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
        elif language == "ko":
            hangul = len(re.findall(r'[\uAC00-\uD7AF]', text))
            others = re.findall(r'[^\uAC00-\uD7AF\s]+', text)
            return hangul + len(others)
        else:
            return len([w for w in re.sub(r'\s+', ' ', text.strip()).split() if w])

    @staticmethod
    def _find_context(transcription: str, reference_text: str, size: int = 200) -> str:
        """
        Find relevant context from reference text (ported from tts_project.find_best_matching_context).
        Uses exact phrase match first, then fuzzy char-level sliding window as fallback.
        """
        if not reference_text.strip() or not transcription.strip():
            return ""

        words = transcription.strip().split()
        if not words:
            return ""

        ref_lower = reference_text.lower()
        best_match_pos = -1

        # Try decreasing phrase lengths (exact match)
        for n in (min(5, len(words)), 3, 2):
            phrase = " ".join(words[:n]).lower()
            pos = ref_lower.find(phrase)
            if pos != -1:
                best_match_pos = pos
                break

        # Fuzzy char-level sliding window (step=10 for efficiency, threshold=0.6)
        if best_match_pos == -1:
            search_key = " ".join(words[:min(5, len(words))]).lower()
            key_len = len(search_key)
            best_ratio = 0.0
            for i in range(0, max(1, len(ref_lower) - key_len + 1), 10):
                window = ref_lower[i:i + key_len]
                ratio = difflib.SequenceMatcher(None, search_key, window).ratio()
                if ratio > best_ratio and ratio > 0.6:
                    best_ratio = ratio
                    best_match_pos = i

        # Extract context window around best match
        if best_match_pos != -1:
            start = max(0, best_match_pos - size // 2)
            end   = min(len(reference_text), best_match_pos + size)
            ctx = reference_text[start:end]
            # Try to end at sentence boundary (keep text tidier)
            for punct in (".", "!", "?", "\n"):
                last_p = ctx.rfind(punct)
                if last_p > len(ctx) * 0.7:
                    ctx = ctx[:last_p + 1]
                    break
            return ctx.strip()

        # Fallback: first `size` chars of reference
        return reference_text[:size].strip()

    @staticmethod
    def _in_reference(text: str, reference: str, threshold: float = 0.70) -> bool:
        """Fuzzy check whether text is present in reference (sliding window)."""
        clean = lambda s: re.sub(r'[^\w\s]', '', s.lower().strip())
        ct = clean(text)
        cr = clean(reference)
        # Guard: empty text or reference always False
        if not ct or not cr:
            return False
        if ct in cr:
            return True
        tw = ct.split()
        rw = cr.split()
        if not tw or not rw:
            return False
        n = len(tw)
        # If text is longer than reference, use reversed comparison
        if n > len(rw):
            return difflib.SequenceMatcher(None, ct, cr).ratio() >= threshold
        for i in range(len(rw) - n + 1):
            window = ' '.join(rw[i:i + n])
            if difflib.SequenceMatcher(None, ct, window).ratio() >= threshold:
                return True
        return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _slugify(text: str, max_len: int = 30) -> str:
        """
        Convert any text (including Vietnamese with diacritics) to a short,
        Latin-only ASCII slug suitable for filenames.

        Steps:
          1. Unicode normalize (NFKD) → separate base chars from diacritics
          2. Encode as ASCII, ignore non-ASCII → removes all diacritics
          3. Replace spaces/special chars with '_'
          4. Collapse repeated '_' and strip
          5. Truncate to max_len

        Examples:
          "Tất tần tật VŨ TRỤ PHIM KHỦNG LONG" → "Tat_tan_tat_VU_TRU_PHIM_KHUNG_LONG"
          "hello world!" → "hello_world"
        """
        # Normalize unicode: NFKD separates diacritics from base chars
        nfkd = unicodedata.normalize("NFKD", text)
        # Keep only ASCII-compatible chars
        ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
        # Replace anything that's not alphanumeric with underscore
        slug = re.sub(r"[^\w]+", "_", ascii_str)
        # Collapse multiple underscores and strip
        slug = re.sub(r"_+", "_", slug).strip("_")
        # Truncate
        return slug[:max_len] if slug else "seg"

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _count_existing(self) -> int:
        if not self._append or not self._metadata_path.exists():
            return 0
        try:
            with open(self._metadata_path, encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def _write_metadata(self, entries: list[DatasetEntry]) -> None:
        """Write in LJSpeech format matching tts_project's create_metadata:
        segment_name|original_text|normalized_text  (no extension, no path prefix)"""
        mode = "a" if self._append and self._metadata_path.exists() else "w"
        with open(self._metadata_path, mode, encoding="utf-8", newline="") as f:
            for e in entries:
                # tts_project writes: segment_name|text|normalized  (basename without ext)
                segment_name = Path(e.wav_rel).stem  # e.g. "title_segment_00001_2.50s"
                f.write(f"{segment_name}|{e.text}|{e.normalized}\n")

    def print_summary(self, entries: list[DatasetEntry]) -> None:
        if not entries:
            log.info("No entries created.")
            return
        durs = []
        for e in entries:
            m = re.search(r"_([\d.]+)s\.wav$", e.wav_rel)
            if m:
                durs.append(float(m.group(1)))
        if durs:
            log.info(
                "Dataset: %d segs | avg %.1fs | min %.1fs | max %.1fs | total %.1f min",
                len(entries), sum(durs)/len(durs), min(durs), max(durs), sum(durs)/60
            )
        log.info("Metadata → %s", self._metadata_path)
