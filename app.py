"""Streamlit application for video summarization."""
import streamlit as st
import os
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.video_processor import VideoProcessor

# Check for GPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    DEVICE = "cpu"
    GPU_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Video Summarization App",
    page_icon="VS",
    layout="wide"
)


@st.cache_resource
def load_transcription_model(model_size):
    """Load and cache the transcription model."""
    from src.audio_transcriber import AudioTranscriber
    print(f"Loading transcription model: {model_size}...")
    transcriber = AudioTranscriber(model_size=model_size)
    transcriber.load_model()
    print("Transcription model loaded!")
    return transcriber


@st.cache_resource
def load_summarization_model():
    """Load and cache the summarization model."""
    from src.text_summarizer import TextSummarizer
    print("Loading summarization model...")
    summarizer = TextSummarizer()
    summarizer.load_model()
    print("Summarization model loaded!")
    return summarizer


# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'result' not in st.session_state:
    st.session_state.result = None


def initialize_processor(transcriber, summarizer):
    """Initialize the video processor with pre-loaded models."""
    from src.youtube_downloader import YouTubeDownloader
    processor = VideoProcessor.__new__(VideoProcessor)
    processor.downloader = YouTubeDownloader()
    processor.transcriber = transcriber
    processor.summarizer = summarizer
    return processor


def main():
    # Header
    st.title("Video Summarization App")
    st.markdown("Convert YouTube videos or uploaded videos into concise summaries!")
    st.caption("CPU-Optimized: Videos under 10 minutes process in ~2 minutes. Longer videos may take additional time.")
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    
    # Show device status
    if GPU_AVAILABLE:
        st.sidebar.success(f"Running on: **GPU ({DEVICE.upper()})** - Fast mode!")
    else:
        st.sidebar.info("Running on: **CPU** - Videos <10min process in ~2 minutes")
    
    whisper_model = st.sidebar.selectbox(
        "Whisper Model Size",
        options=['tiny.en', 'tiny', 'base', 'small'],
        index=0,
        help="'tiny.en' is fastest for English. Use 'base' for better accuracy or other languages."
    )
    
    # Pre-load models at startup
    with st.spinner("Loading models..."):
        transcriber = load_transcription_model(whisper_model)
        summarizer = load_summarization_model()
    
    # Main content tabs
    tab1, tab2 = st.tabs(["YouTube Video", "Upload Video"])
    
    # Tab 1: YouTube Video
    with tab1:
        st.header("YouTube Video")
        youtube_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=..."
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            process_youtube = st.button("Process", key="youtube_button", type="primary")
        
        if process_youtube and youtube_url:
            try:
                # Start timer
                start_time = time.time()
                
                # Initialize processor with pre-loaded models
                processor = initialize_processor(transcriber, summarizer)
                
                # Create placeholders for intermediate results
                info_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                # Step 1: Get video info
                with progress_placeholder.status("Processing...", expanded=False):
                    info = processor.downloader.get_video_info(youtube_url)
                    
                # Display video info immediately
                with info_placeholder.container():
                    st.divider()
                    st.subheader("Video Information")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        # Show thumbnail (YouTube size)
                        thumbnail_url = f"https://img.youtube.com/vi/{info['id']}/maxresdefault.jpg"
                        st.image(thumbnail_url, width=320)
                    with col2:
                        st.write(f"**Title:** {info['title']}")
                        st.write(f"**Uploader:** {info['uploader']}")
                        minutes = info['duration'] // 60
                        seconds = info['duration'] % 60
                        st.write(f"**Duration:** {minutes}m {seconds}s")
                
                # Step 2: Download audio
                with progress_placeholder.status("Processing...", expanded=False):
                    audio_path = processor.downloader.download_audio(youtube_url)
                
                # Step 3: Transcribe
                with progress_placeholder.status("Processing...", expanded=False):
                    transcription = processor.transcriber.transcribe(audio_path)
                
                # Step 4: Summarize
                with progress_placeholder.status("Processing...", expanded=False):
                    summary = processor.summarizer.summarize(transcription)
                
                # Store and display final results
                result = {
                    'title': info['title'],
                    'uploader': info['uploader'],
                    'duration': info['duration'],
                    'transcription': transcription,
                    'summary': summary,
                    'audio_path': audio_path
                }
                st.session_state.result = result
                
                # Clear only progress placeholder, keep video info visible
                progress_placeholder.empty()
                
                # Calculate and print total time
                total_time = time.time() - start_time
                print(f"\n{'='*60}")
                print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                print(f"{'='*60}\n")
                
                display_results(result)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        elif process_youtube and not youtube_url:
            st.warning("Please enter a YouTube URL")
    
    # Tab 2: Upload Video
    with tab2:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mkv', 'avi', 'mov', 'webm']
        )
        
        if uploaded_file:
            st.video(uploaded_file)
            
            col1, col2 = st.columns([1, 5])
            with col1:
                process_upload = st.button("Process", key="upload_button", type="primary")
            
            if process_upload:
                try:
                    # Start timer
                    start_time = time.time()
                    
                    # Save uploaded file temporarily
                    temp_path = Path("data/uploads") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Initialize processor with pre-loaded models
                    processor = initialize_processor(transcriber, summarizer)
                    
                    # Create placeholders
                    progress_placeholder = st.empty()
                    
                    # Step 1: Transcribe
                    with progress_placeholder.status("Processing...", expanded=False):
                        transcription = processor.transcriber.transcribe(str(temp_path))
                    
                    # Step 2: Summarize
                    with progress_placeholder.status("Processing...", expanded=False):
                        summary = processor.summarizer.summarize(transcription)
                    
                    # Store and display results
                    result = {
                        'title': Path(uploaded_file.name).stem,
                        'transcription': transcription,
                        'summary': summary
                    }
                    st.session_state.result = result
                    
                    # Clear placeholder and show results
                    progress_placeholder.empty()
                    
                    # Calculate and print total time
                    total_time = time.time() - start_time
                    print(f"\n{'='*60}")
                    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                    print(f"{'='*60}\n")
                    
                    display_results(result)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Display previous result if exists
    if st.session_state.result and not (process_youtube or (uploaded_file and process_upload)):
        st.divider()
        st.subheader("Previous Result:")
        display_results(st.session_state.result)


def display_results(result):
    """Display the processing results."""
    # Summary
    st.subheader("Summary")
    st.info(result['summary'])


if __name__ == "__main__":
    main()
