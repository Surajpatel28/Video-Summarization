"""Main video processing pipeline."""
import os
from pathlib import Path
from .youtube_downloader import YouTubeDownloader
from .audio_transcriber import AudioTranscriber
from .text_summarizer import TextSummarizer


class VideoProcessor:
    """Complete pipeline for video summarization."""
    
    def __init__(self, whisper_model='tiny.en', model_path='model_cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'):
        """
        Initialize the video processor with CPU-optimized models.
        
        Args:
            whisper_model (str): faster-whisper model size (default: 'tiny.en')
            model_path (str): Path to Qwen GGUF model file
        """
        self.downloader = YouTubeDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.summarizer = TextSummarizer(model_path=model_path)
    
    def process_youtube_video(self, url):
        """
        Process a YouTube video: download audio, transcribe, and summarize.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            dict: Dictionary containing video info, transcription, and summary
        """
        # Get video info
        print("Fetching video information...")
        info = self.downloader.get_video_info(url)
        
        # Download audio
        print("Downloading audio...")
        audio_path = self.downloader.download_audio(url)
        
        # Transcribe audio
        print("Transcribing audio to text...")
        transcription = self.transcriber.transcribe(audio_path)
        
        # Summarize text
        print("Generating summary...")
        summary = self.summarizer.summarize(transcription)
        
        print("Processing complete!")
        
        return {
            'title': info['title'],
            'uploader': info['uploader'],
            'duration': info['duration'],
            'transcription': transcription,
            'summary': summary,
            'audio_path': audio_path
        }
    
    def process_local_video(self, video_path):
        """
        Process a local video file: extract audio, transcribe, and summarize.
        
        Args:
            video_path (str): Path to local video file
            
        Returns:
            dict: Dictionary containing transcription and summary
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # For local videos, we'll use the video directly as audio
        # (Whisper can handle video files)
        print("🎤 Transcribing audio from video...")
        transcription = self.transcriber.transcribe(video_path)
        
        # Summarize text
        print("📝 Generating summary...")
        summary = self.summarizer.summarize(transcription)
        
        print("✅ Processing complete!")
        
        return {
            'title': Path(video_path).stem,
            'transcription': transcription,
            'summary': summary
        }
