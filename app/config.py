"""
Application Configuration
Centralizes environment variables, model identifiers, scoring weights,
and threshold constants for the AI Career Copilot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Gemini Models with graceful fallback order
# Gemini Models
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODELS = []

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ATS Scoring Weights & Thresholds
DEFAULT_SEMANTIC_WEIGHT = 0.70
DEFAULT_SKILL_WEIGHT = 0.30
SKILL_SIMILARITY_THRESHOLD = 65.0  # Cosine similarity % to consider a skill matched
SKILL_PARTIAL_THRESHOLD = 45.0     # Cosine similarity % for related/adjacent skills

# Document Processing Constraints
MAX_FILE_SIZE_MB = 10
MAX_TEXT_LENGTH_CHARS = 100000
CHUNK_WORD_SIZE = 160
CHUNK_OVERLAP_WORDS = 30

# UI Theme Config
APP_TITLE = "AI Career Copilot"
APP_SUBTITLE = "Enterprise ATS Intelligence • Resume Optimizer • AI Career Strategist"
APP_VERSION = "2.0.0"

