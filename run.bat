@echo off
REM ── YT2Dataset launcher ──────────────────────────────────────────────────────
REM Usage:
REM   run.bat            → mở GUI (mặc định)
REM   run.bat gui        → mở GUI
REM   run.bat cli "https://youtu.be/xxx" --lang vi   → chạy CLI

cd /d "%~dp0"
title YT2Dataset

REM Activate venv nếu có
call "%~dp0..\.venv\Scripts\activate.bat" 2>nul || call "%~dp0.venv\Scripts\activate.bat" 2>nul

IF "%1"=="cli" (
    shift
    python main.py %*
) ELSE (
    python app.py
)
