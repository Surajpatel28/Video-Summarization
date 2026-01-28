#!/usr/bin/env python3
"""
Standalone script to pre-download models from HuggingFace.

This script allows users to download the Qwen GGUF model before running the app.
Useful for:
- Pre-caching models in containerized environments
- Offline deployment preparation
- Avoiding first-run download delays

Usage:
    python scripts/download_models.py
    
    # With HuggingFace token (recommended to avoid rate limits):
    HF_TOKEN=your_token python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from huggingface_hub import hf_hub_download


def download_qwen_model():
    """Download Qwen 2.5 1.5B GGUF model from HuggingFace."""
    repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    cache_dir = Path(__file__).parent.parent / "model_cpp" / "models"
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    expected_path = cache_dir / filename
    if expected_path.exists():
        print(f"OK: Model already exists at: {expected_path}")
        print(f"  Size: {expected_path.stat().st_size / (1024**3):.2f} GB")
        return str(expected_path)
    
    print("=" * 70)
    print("Downloading Qwen 2.5 1.5B GGUF Model")
    print("=" * 70)
    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    print(f"Size: ~1 GB")
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("OK: Using HF_TOKEN for authentication")
    else:
        print("WARNING: No HF_TOKEN found - may encounter rate limits")
        print("  Get your token at: https://huggingface.co/settings/tokens")
    print()
    
    try:
        print("Starting download...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
            token=hf_token,
            resume_download=True
        )
        
        print()
        print("=" * 70)
        print("OK: Download Complete!")
        print("=" * 70)
        print(f"Model saved to: {downloaded_path}")
        
        # Check file size
        file_size_gb = Path(downloaded_path).stat().st_size / (1024**3)
        print(f"File size: {file_size_gb:.2f} GB")
        print()
        print("You can now run the application with:")
        print("  streamlit run app.py")
        
        return downloaded_path
        
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR: Download Failed")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        
        if "rate limit" in str(e).lower():
            print("TIP: Rate limit exceeded. Solutions:")
            print("  1. Set HF_TOKEN environment variable")
            print("  2. Wait and try again later")
            print("  3. Download manually from:")
            print(f"     https://huggingface.co/{repo_id}/tree/main")
        
        sys.exit(1)


def check_whisper_model():
    """Check if Whisper models are available."""
    print("\n" + "=" * 70)
    print("Whisper Models")
    print("=" * 70)
    print("Whisper models are downloaded automatically by openai-whisper")
    print("on first use. No action needed here.")
    print()
    print("Default model: tiny.en (~39 MB)")
    print("Cache location: ~/.cache/whisper/")


if __name__ == "__main__":
    print()
    print("Video Summarization - Model Downloader")
    print()
    
    # Download Qwen model
    download_qwen_model()
    
    # Info about Whisper
    check_whisper_model()
    
    print("=" * 70)
    print("OK: Setup Complete!")
    print("=" * 70)
    print()
