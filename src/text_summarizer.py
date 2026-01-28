"""Module for summarizing text using Qwen GGUF (GPU/CPU compatible)."""
from llama_cpp import Llama
import re
import multiprocessing
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False


class TextSummarizer:
    """Summarize text using Qwen GGUF model (GPU/CPU compatible)."""
    
    def __init__(self, model_path='model_cpp/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'):
        """
        Initialize the text summarizer with GGUF model.
        
        Args:
            model_path (str): Path to GGUF model file (used as cache location)
        """
        self.model_path = model_path
        self.repo_id = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
        self.filename = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
        self.model = None
        self.max_context = 32768
        self.gpu_available = GPU_AVAILABLE
    
    def _download_model(self):
        """Download model from HuggingFace if not present locally."""
        model_path = Path(self.model_path)
        
        # Check if model exists at the specified path
        if model_path.exists():
            print(f"Model already exists at {self.model_path}")
            return str(model_path)
        
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Qwen model from HuggingFace...")
        print(f"This is a one-time download (~1GB)...")
        print(f"Note: Set HF_TOKEN environment variable to avoid rate limits")
        
        try:
            # Use huggingface_hub for proper caching and token support
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                cache_dir=str(model_path.parent),
                token=os.environ.get("HF_TOKEN"),  # Supports HF_TOKEN for authentication
                resume_download=True  # Resume if interrupted
            )
            
            print(f"OK: Model downloaded successfully!")
            print(f"  Cached at: {downloaded_path}")
            
            # Update model_path to the actual downloaded location
            self.model_path = downloaded_path
            return downloaded_path
            
        except Exception as e:
            print(f"ERROR: Error downloading model: {e}")
            if "rate limit" in str(e).lower():
                print("TIP: Set HF_TOKEN environment variable to increase rate limits")
                print("   Get your token at: https://huggingface.co/settings/tokens")
            raise RuntimeError(f"Failed to download model from HuggingFace: {e}")
    
    def load_model(self):
        """Load the GGUF model (uses GPU if available)."""
        # Download model if not present
        self._download_model()
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            device = "GPU" if self.gpu_available else "CPU"
            print(f"Loading Qwen GGUF model on {device}...")
            
            # Configure for GPU or CPU
            if self.gpu_available:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=32768,
                    n_gpu_layers=35,  # Offload all layers to GPU
                    n_batch=512,
                    verbose=False
                )
            else:
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=32768,
                    n_threads=multiprocessing.cpu_count(),
                    n_batch=512,
                    verbose=False
                )
            print(f"Model loaded on {device}!")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.model.tokenize(text.encode()))
    
    def chunk_text(self, text: str, chunk_size: int = 3000, overlap: int = 300) -> list:
        """Smart chunking that respects sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                overlap_sentences = current_chunk[-3:] if len(current_chunk) > 3 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks
    
    def summarize(self, text, max_length=256, use_hierarchical=True):
        """
        Summarize the given text using the best strategy based on length.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum tokens for summary
            use_hierarchical (bool): Use hierarchical (section-based) approach for long content
            
        Returns:
            str: Summarized text
        """
        self.load_model()
        
        token_count = self.count_tokens(text)
        
        # Direct summarization for shorter content (< ~10 minutes of transcript)
        if token_count < 15000:
            return self._summarize_direct(text, max_length)
        # Hierarchical summarization for medium-length content (10-30 minutes)
        elif token_count < 25000 and use_hierarchical:
            return self._summarize_hierarchical(text, max_length)
        # Map-reduce for very long content
        else:
            return self._summarize_map_reduce(text)
    
    def _summarize_direct(self, text: str, max_length: int = 256, section_name: str = None) -> str:
        """Direct summarization with enhanced prompting."""
        section_context = f"Section: {section_name}\n" if section_name else ""
        
        prompt = f"""<|im_start|>system
You are an expert at summarizing video content. Your summaries are accurate, well-organized, and capture the essential information.<|im_end|>
<|im_start|>user
Instruction: Create a comprehensive yet concise summary of the video transcript below.

Guidelines:
- Identify and highlight the main topics discussed
- Capture key points, arguments, and conclusions
- Preserve important details, examples, or data mentioned
- Structure the summary logically with clear flow
- Use clear, professional language

{section_context}Transcript:
{text}

Provide a well-structured summary:<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = self.model(
                prompt,
                max_tokens=max_length,
                temperature=0.3,
                top_p=0.85,
                stop=["<|im_end|>"],
                echo=False
            )
            
            print(f"Model output type: {type(output)}")
            print(f"Model output keys: {output.keys() if isinstance(output, dict) else 'Not a dict'}")
            
            if output and 'choices' in output and len(output['choices']) > 0:
                if 'text' in output['choices'][0]:
                    return output['choices'][0]['text'].strip()
                else:
                    print(f"No 'text' in choices[0], keys: {output['choices'][0].keys()}")
                    return "Unable to generate summary - no text in response."
            else:
                print(f"Invalid output structure: {output}")
                return "Unable to generate summary - invalid response structure."
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _summarize_hierarchical(self, text: str, max_length: int = 256) -> str:
        """Hierarchical summarization: sections → section summaries → final summary."""
        print("Using hierarchical summarization...")
        
        # Divide into logical sections (approximately 5-7 minute segments)
        word_count = len(text.split())
        words_per_section = 2500  # ~5 minutes of speech
        num_sections = max(2, min(5, word_count // words_per_section))
        
        sections = self._divide_into_sections(text, num_sections)
        print(f"Processing {len(sections)} sections...")
        
        # Summarize each section
        section_summaries = []
        for i, section in enumerate(sections, 1):
            print(f"Section {i}/{len(sections)}...", end="\r")
            section_summary = self._summarize_direct(
                section, 
                max_length=150, 
                section_name=f"Part {i} of {len(sections)}"
            )
            section_summaries.append(f"Part {i}: {section_summary}")
        
        # Create final consolidated summary
        combined = "\n\n".join(section_summaries)
        print("\nGenerating final consolidated summary...")
        
        final_prompt = f"""<|im_start|>system
You are an expert at creating unified summaries from multiple parts.<|im_end|>
<|im_start|>user
Instruction: Consolidate these section summaries into one cohesive, comprehensive summary.

Section Summaries:
{combined}

Create a unified summary that flows naturally:<|im_end|>
<|im_start|>assistant
"""
        
        try:
            output = self.model(
                final_prompt,
                max_tokens=max_length,
                temperature=0.3,
                top_p=0.85,
                stop=["<|im_end|>"],
                echo=False
            )
            
            if output and 'choices' in output and len(output['choices']) > 0:
                if 'text' in output['choices'][0]:
                    return output['choices'][0]['text'].strip()
            return "Unable to generate consolidated summary."
        except Exception as e:
            print(f"Error in final consolidation: {str(e)}")
            return "\n\n".join(section_summaries)
    
    def _divide_into_sections(self, text: str, num_sections: int) -> list:
        """Divide text into logical sections based on sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences_per_section = len(sentences) // num_sections
        
        sections = []
        for i in range(num_sections):
            start_idx = i * sentences_per_section
            end_idx = start_idx + sentences_per_section if i < num_sections - 1 else len(sentences)
            section = ' '.join(sentences[start_idx:end_idx])
            sections.append(section)
        
        return sections
    
    def _summarize_map_reduce(self, text: str) -> str:
        """Map-Reduce summarization for very long content (>30 min)."""
        print("Text is very long, using map-reduce...")
        chunks = self.chunk_text(text, chunk_size=3000, overlap=300)
        print(f"Processing {len(chunks)} chunks...")

        chunk_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}/{len(chunks)}...", end="\r")
            summary = self._summarize_direct(chunk, max_length=200)
            chunk_summaries.append(summary)
        
        combined = "\n\n".join(chunk_summaries)
        print("\nGenerating final summary...")
        return self._summarize_direct(combined, max_length=300)
