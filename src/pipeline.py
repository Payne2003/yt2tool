"""
src/pipeline.py — Orchestrates the full YouTube → TTS Dataset pipeline
"""
from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config_loader import load_config
from .dataset_builder import DatasetBuilder, DatasetEntry
from .downloader import DownloadResult, YouTubeDownloader
from .logger import get_logger, setup_logger
from .normalizer import TextNormalizer
from .transcriber import Transcriber

console = Console()
log = get_logger("pipeline")


class YT2DatasetPipeline:
    """
    Full pipeline:
      1. Download audio + subtitle from YouTube URL
      2. Transcribe audio with Whisper (word-level timestamps)
      3. Segment into 2–12 second chunks
      4. Normalize text (language-specific)
      5. Export WAV segments + metadata.csv
    """

    def __init__(
        self,
        cfg: dict | None = None,
        config_path: str | None = None,
        language: str | None = None,
        whisper_model: str | None = None,
        output_dir: str | None = None,
        openai_key: str | None = None,
    ):
        self._cfg = cfg or load_config(config_path)

        # CLI overrides
        if language:
            self._cfg.setdefault("whisper", {})["language"] = language
        if whisper_model:
            self._cfg.setdefault("whisper", {})["model"] = whisper_model
        if output_dir:
            self._cfg.setdefault("dataset", {})["output_dir"] = output_dir
        if openai_key:
            self._cfg.setdefault("openai", {})["api_key"] = openai_key

        self._downloader = YouTubeDownloader(self._cfg)
        self._transcriber = Transcriber(self._cfg, progress_cb=self._log_cb)
        self._builder = DatasetBuilder(self._cfg)
        self._normalizer = TextNormalizer()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, url: str) -> list[DatasetEntry]:
        """Run the full pipeline for a single YouTube URL."""
        console.rule(f"[bold cyan]YT2Dataset[/bold cyan]")
        console.print(f"  URL: [link={url}]{url}[/link]")

        # ── Step 1: Download ──────────────────────────────────────────────────
        with self._spinner("Downloading audio…"):
            dl: DownloadResult = self._downloader.download(url)

        if not dl.success:
            console.print(f"[bold red]✗ Download failed:[/bold red] {dl.error}")
            return []

        console.print(f"  [green]✓[/green] Video: {dl.title}")
        console.print(f"  [green]✓[/green] Audio: {dl.audio_path}")
        if dl.subtitle_path:
            console.print(f"  [green]✓[/green] Subtitle: {dl.subtitle_path}")
        else:
            console.print("  [yellow]⚠[/yellow] No subtitle — Whisper only")

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        lang = self._cfg.get("whisper", {}).get("language", "auto")
        console.print(f"\n[bold]Transcribing[/bold] (model={self._cfg.get('whisper',{}).get('model','medium')}, lang={lang})…")

        segments = self._transcriber.transcribe(dl.audio_path, language=lang if lang != "auto" else None)

        if not segments:
            console.print("[bold red]✗ No segments produced. Aborting.[/bold red]")
            return []

        console.print(f"  [green]✓[/green] {len(segments)} segments created")

        # ── Step 3: Build dataset ─────────────────────────────────────────────
        prefix = self._sanitize(dl.title)
        subtitle_text = ""
        if dl.subtitle_path:
            subtitle_text = Path(dl.subtitle_path).read_text(encoding="utf-8", errors="replace")

        console.print(f"\n[bold]Building dataset[/bold] (prefix={prefix})…")
        entries = self._builder.build(
            audio_path=dl.audio_path,
            segments=segments,
            language=lang if lang != "auto" else "en",
            prefix=prefix,
            subtitle_text=subtitle_text,
            progress_cb=self._log_cb,
        )

        # ── Summary ───────────────────────────────────────────────────────────
        self._print_summary(dl, entries)
        return entries

    def run_batch(self, urls: list[str]) -> list[DatasetEntry]:
        """Run pipeline for multiple URLs, appending to same dataset."""
        all_entries: list[DatasetEntry] = []
        for i, url in enumerate(urls, 1):
            console.print(f"\n[bold cyan]── URL {i}/{len(urls)} ──[/bold cyan]")
            entries = self.run(url)
            all_entries.extend(entries)
        console.print(f"\n[bold green]✔ Total: {len(all_entries)} segments from {len(urls)} videos[/bold green]")
        return all_entries

    # ── Private ───────────────────────────────────────────────────────────────

    def _spinner(self, msg: str):
        return Progress(
            SpinnerColumn(),
            TextColumn(msg),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        )

    def _log_cb(self, msg: str) -> None:
        log.debug(msg)
        # Also show in console at lower verbosity
        if any(kw in msg for kw in ("✓", "⚠", "Error", "Detected", "Transcrib")):
            console.print(f"  {msg}")

    def _print_summary(self, dl: DownloadResult, entries: list[DatasetEntry]) -> None:
        out_dir = Path(self._cfg.get("dataset", {}).get("output_dir", "output"))
        table = Table(title="Summary", show_lines=True)
        table.add_column("Item", style="cyan")
        table.add_column("Value")
        table.add_row("Video", dl.title)
        table.add_row("Segments", str(len(entries)))
        table.add_row("Dataset dir", str(out_dir))
        table.add_row("Metadata", str(out_dir / "metadata.csv"))
        if entries:
            import re as _re
            durs = []
            for e in entries:
                m = _re.search(r"_([\d.]+)s\.wav$", e.wav_rel)
                if m:
                    durs.append(float(m.group(1)))
            if durs:
                table.add_row("Avg duration", f"{sum(durs)/len(durs):.1f}s")
                table.add_row("Total audio", f"{sum(durs)/60:.1f} min")
        console.print(table)

    @staticmethod
    def _sanitize(name: str, max_len: int = 50) -> str:
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]', "_", name)
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe[:max_len]
