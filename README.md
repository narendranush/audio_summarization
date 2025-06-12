# 🗣️ Financial YouTube Video Audio Summarization

This Streamlit app lets you **transcribe and summarize speech** from financial YouTube videos or uploaded `.wav` audio files. It leverages **Whisper-based ASR** for transcription and **transformer models** fine-tuned for summarization in the finance domain.

---

## 🔧 Features

- 🎙️ **Input Types**: Upload `.wav` audio or provide a YouTube link.
- 🔊 **Audio Preprocessing**: Mono conversion & resampling to 16kHz.
- 🧠 **Transcription**: Whisper-based model for accurate speech recognition.
- 📚 **Summarization**: Finance-specific summarization via BRIO model.
- 📹 **YouTube Support**: Downloads & extracts audio with `yt-dlp` + `ffmpeg`.

---

## 🚀 Quick Start

### 1. Install `uv`

> `uv` is a fast Python package/dependency manager and an alternative to pip + venv.

If you don’t already have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with Homebrew:

```bash
brew install astral-sh/uv/uv
```

### 2. Install dependencies using `uv`

```bash
uv venv       # Create virtual environment
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

uv pip install -r pyproject.toml
```

> Or, if your `pyproject.toml` is already configured with `[project]` and `[dependencies]`, just run:

```bash
uv pip install .
```

### 3. Run the App

```bash
streamlit run main.py
```

---

## 📦 Requirements Overview

Here’s a breakdown of key libraries used:

| Package        | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `streamlit`    | UI interface                                    |
| `transformers` | Hugging Face pipelines for ASR/NLP              |
| `torchaudio`   | Audio handling & preprocessing                  |
| `yt-dlp`       | Downloading audio from YouTube                  |
| `soundfile`    | WAV file reading/writing                        |
| `ffmpeg`       | Required backend for audio extraction (YouTube) |

---

## ⚠️ ffmpeg is Required

The app uses `yt-dlp` which depends on **`ffmpeg`** for extracting and converting audio.

### How to Install ffmpeg:

- **Linux (Ubuntu/Debian)**:

  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **macOS (with Homebrew)**:

  ```bash
  brew install ffmpeg
  ```

- **Windows**:

  - Download from: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
  - Add `ffmpeg/bin` to your system PATH.

---

## 📂 Project Structure

```
app.py               # Main application
pyproject.toml       # Dependencies and config for uv
README.md            # Documentation
```

---

## ✨ Example Use Case

1. Upload a `.wav` file or paste a YouTube link in the sidebar.
2. Click **"🚀 Generate Result !!"**.
3. View the generated transcription and summarization.

---
