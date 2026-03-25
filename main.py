"""
main.py — YT2Dataset CLI entry-point

Usage:
  # Single URL
  python main.py "https://www.youtube.com/watch?v=xxx"

  # With language
  python main.py "https://youtu.be/xxx" --lang vi

  # Multiple URLs
  python main.py url1 url2 url3 --lang vi

  # From file
  python main.py --file urls.txt --lang vi

  # Custom output dir
  python main.py "https://youtu.be/xxx" --output D:/my_dataset --lang vi

  # Large Whisper model
  python main.py "https://youtu.be/xxx" --whisper-model large-v3 --lang vi
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import load_config
from src.logger import setup_logger
from src.pipeline import YT2DatasetPipeline

console = Console()


@click.command()
@click.argument("urls", nargs=-1, required=False)
@click.option("--file", "-f", "url_file", default=None,
              type=click.Path(exists=True), help="Text file with one URL per line")
@click.option("--lang", "-l", default=None,
              help="Language code: vi / en / ja / ko / tl / auto (default: từ config)")
@click.option("--whisper-model", "-m", default=None,
              help="Whisper model: tiny/base/small/medium/large/large-v3")
@click.option("--output", "-o", default=None,
              type=click.Path(), help="Output directory for dataset")
@click.option("--openai-key", default=None,
              envvar="OPENAI_API_KEY", help="OpenAI API key for transcription correction")
@click.option("--config", "-c", default=None,
              type=click.Path(exists=True), help="Path to config.yaml")
def main(
    urls: tuple[str, ...],
    url_file: str | None,
    lang: str | None,
    whisper_model: str | None,
    output: str | None,
    openai_key: str | None,
    config: str | None,
):
    """YT2Dataset — YouTube → TTS Dataset Pipeline.

    Download YouTube audio, transcribe with Whisper, normalize text,
    and export a LJSpeech-format dataset (wavs/ + metadata.csv).
    """
    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(config)
    log_cfg = cfg.get("logging", {})
    setup_logger(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "logs/yt2dataset.log"),
        console=log_cfg.get("console", True),
    )

    # ── Collect URLs ──────────────────────────────────────────────────────────
    raw_urls: list[str] = list(urls)
    if url_file:
        lines = Path(url_file).read_text(encoding="utf-8").splitlines()
        raw_urls.extend(l.strip() for l in lines if l.strip() and not l.startswith("#"))

    if not raw_urls:
        console.print("[bold yellow]No URLs provided.[/bold yellow]")
        console.print("Usage: python main.py \"<youtube_url>\" [--lang vi]")
        raise SystemExit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    pipeline = YT2DatasetPipeline(
        cfg=cfg,
        language=lang,
        whisper_model=whisper_model,
        output_dir=output,
        openai_key=openai_key,
    )

    if len(raw_urls) == 1:
        pipeline.run(raw_urls[0])
    else:
        pipeline.run_batch(raw_urls)


if __name__ == "__main__":
    main()
