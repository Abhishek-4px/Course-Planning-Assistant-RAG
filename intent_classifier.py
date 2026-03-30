"""
Intent Classification Module
- Uses LLM (not rules) to classify user queries into:
    "eligibility" | "planning" | "information"
"""
from llm_client import call_llm
from config import INTENT_LABELS


INTENT_PROMPT = """You are an intent classifier for a university Course Planning Assistant.

Classify the following student query into EXACTLY ONE of these categories:
- "eligibility"  → student asks if they CAN take a specific course (prerequisite check)
- "planning"     → student asks what courses they SHOULD take next, or asks for recommendations
- "information"  → student asks for general info about a course, policy, syllabus, or schedule

Rules:
- Return ONLY the single word label, nothing else.
- If unsure, default to "information".

Student Query: {query}

Label:"""


def classify_intent(query: str) -> str:
    """
    Classify a user query into one of the three intent labels.
    Returns one of: "eligibility", "planning", "information"
    """
    prompt = INTENT_PROMPT.format(query=query)
    response = call_llm(prompt).strip().lower().strip('"').strip("'")

    # Validate the response
    for label in INTENT_LABELS:
        if label in response:
            return label

    # Default fallback
    return "information"
