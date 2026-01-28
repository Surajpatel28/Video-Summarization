"""Configuration settings for the video summarization application."""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
TEMP_DIR = DATA_DIR / 'temp'
UPLOADS_DIR = DATA_DIR / 'uploads'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Create directories if they don't exist
for directory in [DATA_DIR, TEMP_DIR, UPLOADS_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings (CPU Optimized)
DEFAULT_WHISPER_MODEL = 'tiny.en'  # Fastest for English (~18s for 8min video)
DEFAULT_QWEN_MODEL_PATH = 'model_cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'

# Audio settings
AUDIO_FORMAT = 'm4a'
AUDIO_QUALITY = '192'

# Summarization settings
DEFAULT_MAX_SUMMARY_LENGTH = 150
DEFAULT_MIN_SUMMARY_LENGTH = 50
MAX_CHUNK_LENGTH = 1024

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.webm']
SUPPORTED_AUDIO_FORMATS = ['.m4a', '.mp3', '.opus', '.wav']
