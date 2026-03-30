
import os
from config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def _call_groq(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def call_llm(prompt: str) -> str:
    if GEMINI_API_KEY:
        try:
            return _call_gemini(prompt)
        except Exception as e:
            print(f"Warning: Gemini failed: {e}. Falling back to Groq ...")

    if GROQ_API_KEY:
        try:
            return _call_groq(prompt)
        except Exception as e:
            print(f"Error: Groq also failed: {e}")
            raise

    raise RuntimeError("No LLM API key configured. Set GEMINI_API_KEY or GROQ_API_KEY.")
