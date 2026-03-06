"""
Configuration file for VideoRAG system.
All configurable parameters are centralized here with explanations.
"""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

# Base directory - automatically detected
BASE_DIR = Path(__file__).parent.resolve()

# Detect Streamlit Cloud (Linux container with specific env vars)
IS_STREAMLIT_CLOUD = (
    os.environ.get("SF_PARTNER") == "streamlit"
    or os.environ.get("HOSTNAME") == "streamlit"
    or os.path.exists("/mount/src")
)

# Load .env from parent directory (D:\langchain_models\.env)
# This allows sharing the same .env across multiple projects
env_path = BASE_DIR.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from: {env_path}")
else:
    # Fallback: try loading from current directory
    load_dotenv()
    print("⚠ Warning: .env file not found in parent directory, using system environment variables")

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

# Data directories
_REPO_CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "assets"

# On Streamlit Cloud, chromadb's Rust engine needs full read/write access
# (WAL journal, shared-memory files, segment locks).  The repo mount may not
# guarantee this, so we copy the pre-built DB to /tmp/ on every cold start.
if IS_STREAMLIT_CLOUD:
    _CLOUD_CHROMA_DIR = Path("/tmp/chroma_db")
    if not (_CLOUD_CHROMA_DIR / "chroma.sqlite3").exists():
        print(f"[config] Copying chroma_db to {_CLOUD_CHROMA_DIR} for Cloud ...")
        if _CLOUD_CHROMA_DIR.exists():
            shutil.rmtree(_CLOUD_CHROMA_DIR)
        shutil.copytree(str(_REPO_CHROMA_DB_DIR), str(_CLOUD_CHROMA_DIR))
        print(f"[config] Copy complete – {sum(1 for _ in _CLOUD_CHROMA_DIR.rglob('*'))} files")
    else:
        print(f"[config] Cloud chroma_db already at {_CLOUD_CHROMA_DIR}")
    CHROMA_DB_DIR = _CLOUD_CHROMA_DIR
else:
    CHROMA_DB_DIR = _REPO_CHROMA_DB_DIR

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)

# Log files
APP_LOG_FILE = LOGS_DIR / "app.log"
RAG_EVALUATION_LOG_FILE = LOGS_DIR / "rag_evaluation_log.csv"

# Data files
DATA_DIR = BASE_DIR / "architecture" / "data"
VIDEOS_CSV = DATA_DIR / "videos.csv"
TRANSCRIPT_CSV = DATA_DIR / "video_with_meta_data_and_transcript.csv"
URLS_FILE = DATA_DIR / "urls.txt"


# ============================================================================
# TEXT-TO-SPEECH (TTS) CONFIGURATION
# ============================================================================

# TTS paths - use environment variables with fallback defaults
# Set these in .env file or system environment variables
PIPER_EXE = os.getenv("PIPER_EXE", r"D:\piper\piper.exe")
PIPER_VOICE = os.getenv("PIPER_VOICE", r"D:\piper\en_US-lessac-medium.onnx")
FFMPEG_EXE = os.getenv("FFMPEG_EXE", r"D:\Downloads_D_drive\ffmpeg-7.1.1-full_build\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe")

# TTS output settings
TTS_AUDIO_BITRATE = "192k"  # MP3 audio quality
TTS_TEMP_WAV = "temp_voice.wav"  # Temporary WAV file for Piper
TTS_OUTPUT_FILE = "answer.mp3"  # Final audio output


# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

class RetrievalConfig:
    """
    Hybrid retrieval parameters.
    These values are tuned based on experimentation with the dataset.
    """
    
    # MMR (Maximal Marginal Relevance) Retriever
    # - Balances relevance and diversity to avoid redundant results
    MMR_K = 6  # Number of final documents to return
    MMR_FETCH_K = 20  # Number of candidates to fetch before MMR reranking
    # Rationale: fetch_k=20 gives MMR enough candidates to select diverse results
    
    # MultiQuery Retriever
    # - Generates multiple query variations to handle ambiguous questions
    MULTIQUERY_K = 6  # Documents per generated query
    # Rationale: Same as MMR_K for consistency across retrievers
    
    # BM25 Retriever
    # - Keyword-based lexical search for exact term matching
    BM25_K = 6  # Number of documents to retrieve
    # Rationale: Consistent with other retrievers for balanced hybrid retrieval
    
    # Cross-Encoder Reranking
    # - Final reranking step after combining all retriever results
    RERANK_TOP_N = 10  # Number of documents to keep after reranking
    # Rationale: 10 provides enough context without overwhelming the LLM
    
    # Cross-encoder model
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Rationale: Lightweight but effective, good balance of speed and quality

    # Minimum CrossEncoder score required to proceed with answering.
    # ms-marco-MiniLM-L-6-v2 outputs raw logits (not probabilities).
    # In-context relevant chunks typically score > 2.0.
    # Tangential / out-of-context chunks typically score < 0.
    # Calibrated against corpus: lowest in-context = 1.804, highest OOC = -3.911.
    # Threshold 0.5 gives a safety buffer of ~1.3 units below the weakest in-context score.
    RERANK_MIN_SCORE = 0.5


# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================

class EmbeddingConfig:
    """Embedding model configuration."""
    
    # HuggingFace embedding model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # Rationale: Fast, lightweight, good quality for general text
    # 384 dimensions, suitable for ChromaDB
    
    # Embedding batch size for vector store creation
    BATCH_SIZE = 100
    # Rationale: Balance between memory usage and speed


# ============================================================================
# LLM CONFIGURATION
# ============================================================================

class LLMConfig:
    """LLM model configuration for answer generation and evaluation."""
    
    # Answer Generation LLM
    ANSWER_MODEL = "llama-3.3-70b-versatile"  # Groq model
    ANSWER_TEMPERATURE = 0  # Deterministic outputs for consistency
    # Rationale: temperature=0 ensures reproducible answers for same question
    
    # MultiQuery Generation LLM
    MULTIQUERY_MODEL = "gemini-2.5-flash"  # Google Gemini
    # Rationale: Fast and cost-effective for query expansion
    
    # Evaluation LLM (RAGAS judge)
    EVAL_MODEL = "openai/gpt-oss-120b"  # Groq model for RAGAS
    EVAL_TEMPERATURE = 0  # Deterministic evaluation
    # Rationale: Consistent evaluation scores across runs

    # Ground Truth LLM (reference answer generation)
    # MUST be a different model from EVAL_MODEL to avoid circular evaluation:
    # qwen/qwen3-32b (Alibaba/Qwen family) is chosen because:
    # - Completely different family from both the answer model (Meta/Llama)
    #   and the RAGAS judge (OpenAI GPT-OSS) — genuine independence
    # - 32B params with strong synthesis ability; does NOT default to
    #   "I don't know" on questions that require reasoning from context,
    #   which was the failure mode of llama-3.3-70b-versatile
    GROUND_TRUTH_MODEL = "qwen/qwen3-32b"  # Groq preview model
    GROUND_TRUTH_TEMPERATURE = 0


# ============================================================================
# RAG EVALUATION CONFIGURATION
# ============================================================================

class EvaluationConfig:
    """RAGAS evaluation and confidence scoring configuration."""
    
    # Confidence score weights
    # These weights determine how much each metric contributes to final confidence
    WEIGHT_ANSWER_RELEVANCY = 0.40  # Did the model answer the question? (Increased for real-world UX)
    WEIGHT_FAITHFULNESS = 0.40      # Did the answer stick to context? (HIGHEST)
    WEIGHT_CONTEXT_RECALL = 0.10    # Did retrieval cover needed info? (Lowered, internal metric)
    WEIGHT_CONTEXT_PRECISION = 0.10 # How noisy was the retrieved context? (Lowered, internal metric)
    
    # Rationale for weights:
    # - Faithfulness (40%): Most important - prevents hallucinations
    # - Answer Relevancy (40%): Ensures question is actually answered, which users care about most
    # - Context Recall (10%): Good retrieval is important, but secondary to the final answer quality
    # - Context Precision (10%): Noise is less critical if the LLM can filter it out successfully
    
    # Confidence threshold for retry suggestion
    CONFIDENCE_THRESHOLD = 0.4
    # Rationale: Below 0.4 indicates low-quality answer worth retrying
    
    # Auto-logging threshold
    AUTO_LOG_THRESHOLD = 0.6
    # Rationale: Log answers below 0.6 for quality monitoring
    
    # RAGAS prompt configuration
    RAGAS_DEFAULT_N = 1  # Number of samples for RAGAS prompts
    # Rationale: n=1 reduces latency while maintaining quality


# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

class PromptConfig:
    """Answer generation prompt configuration."""
    
    # Number of chunks to use in final answer
    TOP_CHUNKS = 5
    # Rationale: 5 chunks provide comprehensive coverage without overwhelming, giving the LLM more context to find the real answer
    
    # Minimum sentences per answer
    MIN_SENTENCES = 2
    # Rationale: Ensures explanatory answers, but allows concise answers if that's all that's needed
    
    # Target sentences per answer
    TARGET_SENTENCES_MIN = 3
    TARGET_SENTENCES_MAX = 6
    # Rationale: 3-6 sentences balance detail and conciseness without forcing the LLM to hallucinate "fluff"


# ============================================================================
# STREAMLIT UI CONFIGURATION
# ============================================================================

class UIConfig:
    """Streamlit frontend configuration."""
    
    # Page configuration
    PAGE_TITLE = "VideoRAG"
    PAGE_ICON = "🎥"
    LAYOUT = "wide"
    
    # Evaluation progress
    MAX_EVAL_WAIT_SECONDS = 180  # 3 minutes timeout for evaluation
    # Rationale: RAGAS evaluation can take 2-3 minutes on cloud
    
    # Audio generation progress
    MAX_AUDIO_WAIT_SECONDS = 20  # 20 seconds timeout for audio
    # Rationale: TTS generation is typically fast
    
    # Display delays
    AUDIO_READY_DISPLAY_SECONDS = 1.0  # Show "Audio ready!" message duration
    # Rationale: Give user time to see success message


# ============================================================================
# VECTOR STORE CONFIGURATION
# ============================================================================

class VectorStoreConfig:
    """ChromaDB vector store configuration."""
    
    COLLECTION_NAME = "lexi_transcripts"
    PERSIST_DIRECTORY = str(CHROMA_DB_DIR)
    
    # Distance metric
    DISTANCE_METRIC = "cosine"  # or "l2", "ip"
    # Rationale: Cosine similarity is standard for semantic search


# ============================================================================
# API KEYS (from environment variables)
# ============================================================================

# These should be set in .env file or Streamlit Cloud Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_confidence_weights() -> dict:
    """Return confidence weights as a dictionary."""
    return {
        "answer_relevancy": EvaluationConfig.WEIGHT_ANSWER_RELEVANCY,
        "faithfulness": EvaluationConfig.WEIGHT_FAITHFULNESS,
        "context_recall": EvaluationConfig.WEIGHT_CONTEXT_RECALL,
        "context_precision": EvaluationConfig.WEIGHT_CONTEXT_PRECISION,
    }


def validate_config() -> bool:
    """
    Validate that all required configuration is present.
    Returns True if valid, raises ValueError otherwise.
    """
    errors = []
    
    # Check API keys
    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY not set in environment")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set in environment")
    
    # Check TTS paths (only if running locally)
    if os.getenv("SF_PARTNER") != "streamlit":
        if not Path(PIPER_EXE).exists():
            errors.append(f"Piper executable not found at: {PIPER_EXE}")
        if not Path(PIPER_VOICE).exists():
            errors.append(f"Piper voice model not found at: {PIPER_VOICE}")
        if not Path(FFMPEG_EXE).exists():
            errors.append(f"FFmpeg executable not found at: {FFMPEG_EXE}")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


# ============================================================================
# EXPORT ALL CONFIGS
# ============================================================================

__all__ = [
    "BASE_DIR",
    "IS_STREAMLIT_CLOUD",
    "DATA_DIR",
    "CHROMA_DB_DIR",
    "LOGS_DIR",
    "APP_LOG_FILE",
    "RAG_EVALUATION_LOG_FILE",
    "PIPER_EXE",
    "PIPER_VOICE",
    "FFMPEG_EXE",
    "RetrievalConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "EvaluationConfig",
    "PromptConfig",
    "UIConfig",
    "VectorStoreConfig",
    "get_confidence_weights",
    "validate_config",
]
