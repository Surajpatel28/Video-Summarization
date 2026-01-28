"""Module for downloading YouTube videos and audio."""
import yt_dlp
import os
from pathlib import Path
import time


class YouTubeDownloader:
    """Download YouTube videos and audio with duplicate detection."""
    
    def __init__(self, output_dir='data/temp'):
        """
        Initialize the YouTube downloader with output directory.
        
        Args:
            output_dir (str): Directory to save downloaded files
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_video_id(self, url):
        """Extract video ID from YouTube URL."""
        try:
            ydl_opts = {'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('id')
        except Exception as e:
            raise Exception(f"Error extracting video ID: {str(e)}")
    
    def _check_file_exists(self, video_id, extensions=None):
        """Check if file with given ID already exists in output directory."""
        if extensions is None:
            extensions = ('.wav', '.mp4', '.mkv', '.webm', '.m4a', '.mp3', '.opus')
        
        for file in os.listdir(self.output_dir):
            if video_id in file and file.endswith(extensions) and not file.endswith('.part'):
                existing_path = os.path.join(self.output_dir, file)
                print(f"✓ Using existing file: {existing_path}")
                return existing_path
        return None
    
    def download_audio(self, url, filename=None, force_download=False, audio_format='m4a'):
        """
        Download AUDIO ONLY from YouTube video (RECOMMENDED for summarization).
        
        Args:
            url (str): YouTube video URL
            filename (str): Optional custom filename
            force_download (bool): If True, re-download even if exists
            audio_format (str): Output audio format - 'm4a' (default), 'mp3', 'opus', 'wav'
            
        Returns:
            str: Path to downloaded audio file
        """
        # Get video ID first
        video_id = self._get_video_id(url)
        
        # Check if audio already exists (unless force download)
        if not force_download:
            existing_file = self._check_file_exists(video_id)
            if existing_file:
                return existing_file
        
        # Configure download options
        if filename:
            output_template = os.path.join(self.output_dir, f'{filename}_[{video_id}].%(ext)s')
        else:
            output_template = os.path.join(self.output_dir, '%(title)s_[%(id)s].%(ext)s')
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': False,  # Enable output to see what's happening
            'no_warnings': False,  # Show warnings
            'extractor_args': {'youtube': {'player_client': ['android']}},
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',  # Use WAV instead of M4A for reliability
                'preferredquality': '0',  # Best quality (irrelevant for WAV)
            }],
            'postprocessor_args': [
                '-ar', '16000',  # 16kHz sample rate (Whisper native)
                '-ac', '1',      # Mono audio
                '-acodec', 'pcm_s16le',  # 16-bit PCM
            ],
            'keepvideo': False,  # Don't keep the video file
            'writethumbnail': False,
        }
        
        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_title = info.get('title', 'audio')
            
            # Look for WAV file (we changed to WAV format)
            if filename:
                audio_file = os.path.join(self.output_dir, f'{filename}_[{video_id}].wav')
            else:
                audio_file = os.path.join(self.output_dir, f'{video_title}_[{video_id}].wav')
            
            # Wait for file to be completely written
            time.sleep(2)
            
            # Try to find the file
            if os.path.exists(audio_file):
                print(f"✓ Downloaded: {audio_file}")
                return audio_file
            else:
                # Fallback: search for any audio file with video_id
                print(f"Expected file not found, searching for: {video_id}")
                for file in os.listdir(self.output_dir):
                    if video_id in file and not file.endswith('.part'):
                        found_path = os.path.join(self.output_dir, file)
                        print(f"✓ Found: {found_path}")
                        return found_path
                raise Exception("Downloaded audio file not found")
    
    def get_video_info(self, url):
        """Get video information without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extractor_args': {'youtube': {'player_client': ['android']}}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'id': info.get('id'),
                'title': info.get('title'),
                'duration': info.get('duration'),
                'uploader': info.get('uploader'),
                'view_count': info.get('view_count'),
                'description': info.get('description')
            }
