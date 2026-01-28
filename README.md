# Video Summarization (Whisper + Qwen)

A **production-grade video summarization system** that converts YouTube videos or uploaded video files into concise, factual summaries using **Whisper (ASR)** and **Qwen 2.5 (LLM, GGUF)**.

The repository contains:

* A **local Streamlit application** (full CPU/GPU support, production-style architecture)
* A **Google Colab notebook** that demonstrates the **GPU execution path and timing behavior**

---

## Key Features

* YouTube URL or local video upload
* Audio extraction + validation (FFmpeg, 16 kHz mono PCM WAV)
* Fast transcription with Whisper (GPU or CPU)
* Adaptive summarization strategies:

  * **Direct** (short videos)
  * **Hierarchical** (medium videos)
  * **Map-Reduce** (long videos)
* GPU acceleration (CUDA) with automatic CPU fallback
* Model caching & duplicate download detection
* Exportable artifacts (transcript, summary, metadata)
* Clean, modular, production-ready codebase

---

## Architecture Overview

```
User Input (URL / Upload)
        ↓
YouTube Downloader / File Handler
        ↓
FFmpeg Audio Validation + Conversion
        ↓
Whisper Transcription (GPU / CPU)
        ↓
Adaptive Summarization (Qwen GGUF)
        ↓
Streamlit UI Output
```

Core modules:

* `app.py` - Streamlit UI and request orchestration
* `youtube_downloader.py` - YouTube audio + metadata handling
* `audio_transcriber.py` - Whisper wrapper and audio preprocessing
* `text_summarizer.py` - Direct / Hierarchical / Map-Reduce logic
* `video_processor.py` - End-to-end pipeline controller
* `config.py` - Paths, model choices, defaults

---

## Quickstart (Local - Recommended)

### Requirements

* Python **3.10+**
* **FFmpeg** installed
* 8+ CPU cores recommended
* CUDA-capable GPU optional (for acceleration)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/video-summarization.git
cd video-summarization
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `openai-whisper` - Speech-to-text transcription
- `llama-cpp-python` - GGUF model inference
- `huggingface-hub` - Model downloading from HuggingFace
- `yt-dlp` - YouTube video downloading
- `torch` - GPU/CPU support
- `streamlit` - Web interface

**Note:** Models (~1GB) are downloaded automatically on first run from HuggingFace.

### 4. Install FFmpeg

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 5. Run Application

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## Colab Notebook (GPU Demo)

The notebook in `notebooks/colab_demo.ipynb` demonstrates:

* Whisper GPU transcription
* Qwen GGUF GPU inference
* Extractive compression + hierarchical summarization
* Timing breakdown and artifact export

**Important**:

* The Colab notebook showcases the **GPU execution path only**
* CPU benchmarks were collected separately on local hardware and are discussed in documentation

---

## Summarization Strategy Logic

| Content Length | Strategy     | Description                     |
| -------------- | ------------ | ------------------------------- |
| < ~10 min      | Direct       | Single summarization pass       |
| 10-30 min      | Hierarchical | Section summaries -> consolidate |
| > 30 min       | Map-Reduce   | Chunk -> summarize -> reduce      |

Design goals:

* Avoid context overflows
* Reduce LLM token load
* Preserve factual accuracy
* Scale to long-form content

---

## Performance Expectations

Example reference timings (varies by hardware):

* **Colab GPU (T4)**

  * ~6-min video: **~50s end-to-end**
  * Whisper ~13s
  * Summarization ~37s

* **Local CPU (8 cores)**

  * ~10-min video: **~2 minutes**

Primary bottleneck is **LLM decoding**, not transcription.

---

## Project Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── colab_demo.ipynb
├── scripts/
│   └── download_models.py
├── src/
│   ├── app.py
│   ├── youtube_downloader.py
│   ├── audio_transcriber.py
│   ├── text_summarizer.py
│   ├── video_processor.py
│   └── config.py
├── data/
│   ├── temp/        # downloaded audio
│   └── uploads/     # user uploads
└── model_cpp/
    └── models/      # Qwen GGUF models
```

`.gitkeep` files are used to track empty directories.

---

## Configuration

**Models are downloaded automatically from HuggingFace on first run.**

Environment variables (optional but recommended):

```bash
# Recommended: Set HuggingFace token to avoid rate limits
export HF_TOKEN=your_huggingface_token

# Optional: Custom model directory
export MODEL_DIR=./model_cpp/models

# Optional: Custom data directory
export DATA_DIR=./data

# Optional: Whisper model size
export WHISPER_MODEL=tiny.en
```

**Get your HuggingFace token:**
1. Visit https://huggingface.co/settings/tokens
2. Create a read token
3. Set it in your environment or `.env` file

**Pre-download models (optional):**
```bash
# Download models before first run
HF_TOKEN=your_token python scripts/download_models.py
```

---

## Troubleshooting

* **FFmpeg missing** - install via system package manager
* **GPU not detected** - verify CUDA + PyTorch installation
* **Model download slow** - set `HF_TOKEN`
* **Long videos OOM** - Map-Reduce strategy is applied automatically

---

## Contributing

Contributions are welcome.

Suggested workflow:

1. Fork repository
2. Create feature branch
3. Add tests and documentation
4. Submit pull request

---

## License

MIT License. See `LICENSE` file for details.

---

## Disclaimer

Users are responsible for ensuring they have the right to download and process video content. This project is for educational and research purposes.
