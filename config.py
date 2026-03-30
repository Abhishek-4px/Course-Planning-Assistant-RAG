import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent  
DATASET_COURSES = BASE_DIR / "Dataset"
DATASET_PROGRAM_GUIDE = BASE_DIR / "Dataset Program Guide"
DATASET_COLLEGE_POLICY = BASE_DIR / "Dataset College Policy"
VECTOR_STORE_DIR = Path(__file__).resolve().parent / "vector_store"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 7
EVIDENCE_KEYWORDS = [
    "prerequisite", "pre-requisite", "prerequisites",
    "requirement", "required", "must complete",
    "eligibility", "prior knowledge", "co-requisite",
]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.3-70b-versatile"

INTENT_LABELS = ["eligibility", "planning", "information"]
