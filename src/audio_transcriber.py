"""Module for transcribing audio to text using Whisper (GPU/CPU compatible)."""
import whisper
import os
import time
import subprocess
import tempfile
from pathlib import Path

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"


class AudioTranscriber:
    """Transcribe audio files to text using OpenAI Whisper (GPU/CPU)."""
    
    def __init__(self, model_size='tiny.en'):
        """
        Initialize the audio transcriber.
        
        Args:
            model_size (str): Whisper model size - 'tiny.en', 'tiny', 'base', 'small', 'medium'
        """
        self.model_size = model_size
        self.model = None
        self.device = DEVICE
    
    def load_model(self):
        """Load the Whisper model (uses GPU if available)."""
        if self.model is None:
            print(f"Loading Whisper {self.model_size} model on {self.device.upper()}...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"Model loaded on {self.device.upper()}!")
    
    def _validate_audio_file(self, audio_path):
        """Validate that the audio file has proper audio streams."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type,codec_name,duration,sample_rate',
                '-of', 'default=noprint_wrappers=1',
                str(audio_path)
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                print(f"Audio file info:\n{result.stdout}")
                # Check if it has codec_type=audio
                if 'codec_type=audio' in result.stdout:
                    return True
                else:
                    print("WARNING: No audio stream found in file!")
                    return False
            else:
                print(f"WARNING: Could not validate audio file: {result.stderr[:200]}")
                return True  # Assume valid if can't check
                
        except Exception as e:
            print(f"WARNING: Audio validation failed: {e}")
            return True  # Assume valid if can't check
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio with FFmpeg to ensure compatibility with Whisper."""
        print(f"Starting audio preprocessing for: {audio_path}")
        
        # Use absolute paths to avoid issues
        audio_path = os.path.abspath(audio_path)
        
        # Check if already in correct format (16kHz mono WAV)
        if audio_path.lower().endswith('.wav'):
            try:
                cmd = [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=codec_name,sample_rate,channels',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(audio_path)
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                if result.returncode == 0:
                    values = result.stdout.strip().split('\n')
                    if len(values) >= 3:
                        codec, sample_rate, channels = values[0], values[1], values[2]
                        if codec == 'pcm_s16le' and sample_rate == '16000' and channels == '1':
                            print(f"OK: Audio already in correct format (16kHz mono WAV), using directly")
                            return audio_path
            except:
                pass  # Fall through to conversion
        
        temp_dir = Path(audio_path).parent
        base_name = Path(audio_path).stem
        temp_wav = temp_dir / f"{base_name}_whisper_temp.wav"
        
        # Remove existing temp file if present
        if temp_wav.exists():
            try:
                os.remove(temp_wav)
                print("Removed existing temp file")
            except:
                pass
        
        try:
            # Method 1: Direct FFmpeg conversion (most reliable)
            print(f"Converting to WAV: {temp_wav}")
            cmd = [
                'ffmpeg',
                '-hide_banner',
                '-loglevel', 'error',
                '-i', str(audio_path),
                '-vn',           # No video
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-c:a', 'pcm_s16le',  # 16-bit PCM
                '-f', 'wav',     # Force WAV format
                '-y',            # Overwrite if exists
                str(temp_wav)
            ]
            
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            # Check if conversion was successful
            if result.returncode == 0 and temp_wav.exists() and temp_wav.stat().st_size > 1000:  # At least 1KB
                print(f"OK: Audio preprocessed successfully: {temp_wav.stat().st_size / 1024 / 1024:.2f} MB")
                return str(temp_wav)
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"ERROR: FFmpeg conversion failed (code {result.returncode})")
                print(f"Error: {error_msg[:500]}")
                
        except subprocess.TimeoutExpired:
            print("ERROR: FFmpeg conversion timed out (>60s)")
        except FileNotFoundError:
            print("ERROR: FFmpeg not found in PATH - install FFmpeg first")
        except Exception as e:
            print(f"ERROR: Preprocessing exception: {type(e).__name__}: {str(e)}")
        
        # If all methods fail, raise an error instead of silently failing
        raise RuntimeError(
            f"Failed to preprocess audio file. The downloaded video appears to have no audio content or is restricted. "
            f"This specific video cannot be processed. Try a different video."
        )
    
    def transcribe(self, audio_path, language='en'):
        """
        Transcribe audio file to text.
        
        Args:
            audio_path (str): Path to audio file
            language (str): Language code (default: 'en')
            
        Returns:
            str: Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate audio file first
        print("\n" + "=" * 60)
        print("AUDIO VALIDATION STAGE")
        print("=" * 60)
        if not self._validate_audio_file(audio_path):
            raise ValueError("Audio file does not contain valid audio stream")
        
        self.load_model()
        
        processed_audio = None
        temp_file_created = False
        
        try:
            # Preprocess audio to ensure compatibility
            print("\n" + "=" * 60)
            print("AUDIO PREPROCESSING STAGE")
            print("=" * 60)
            processed_audio = self._preprocess_audio(audio_path)
            temp_file_created = True
            print(f"Using preprocessed audio: {processed_audio}")
            
            print("\n" + "=" * 60)
            print("TRANSCRIPTION STAGE")
            print("=" * 60)
            print(f"Transcribing audio on {self.device.upper()}...")
            start = time.time()
            
            # Use fp16 for GPU, fp32 for CPU
            result = self.model.transcribe(
                processed_audio,
                language=language,
                fp16=(self.device == 'cuda'),
                verbose=False
            )
            
            transcription = result['text']
            
            elapsed = time.time() - start
            print(f"OK: Transcription completed in {elapsed:.1f}s on {self.device.upper()}")
            print(f"OK: Transcribed {len(transcription)} characters")
            print("=" * 60 + "\n")
            
            return transcription
            
        except Exception as e:
            print(f"\nERROR: TRANSCRIPTION FAILED")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("=" * 60 + "\n")
            raise
            
        finally:
            # Clean up temporary file if created
            if temp_file_created and processed_audio and os.path.exists(processed_audio):
                try:
                    os.remove(processed_audio)
                    print(f"Cleaned up temp file: {processed_audio}")
                except Exception as cleanup_error:
                    print(f"Warning: Failed to cleanup temp file: {cleanup_error}")
