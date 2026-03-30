"""
Configuration for the Course Planning Assistant.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # e:\LMS System
DATASET_COURSES = BASE_DIR / "Dataset"
DATASET_PROGRAM_GUIDE = BASE_DIR / "Dataset Program Guide"
DATASET_COLLEGE_POLICY = BASE_DIR / "Dataset College Policy"
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "vector_store"

# ── Chunking ───────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ── Retrieval ──────────────────────────────────────────
TOP_K = 7

# ── Evidence Keywords ──────────────────────────────────
EVIDENCE_KEYWORDS = [
    "prerequisite", "pre-requisite", "prerequisites",
    "requirement", "required", "must complete",
    "eligibility", "prior knowledge", "co-requisite",
]

# ── Embedding Model ───────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── LLM Config ────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Intent Labels ─────────────────────────────────────
INTENT_LABELS = ["eligibility", "planning", "information"]
