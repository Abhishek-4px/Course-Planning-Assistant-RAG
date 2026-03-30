# Course Planning Assistant

This repository contains a full end-to-end RAG (Retrieval-Augmented Generation) system that serves as a Course Planning Assistant. It helps students check course eligibility (prerequisites), get course information, and receive course planning suggestions.

All answers are strictly grounded in the official university documents (Course Handouts, Program Guides, College Policies), with explicit page-level citations.

## Features

- **Document Ingestion**: Supports both PDF and DOCX files.
- **RAG Architecture**: Uses FAISS vector store with HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Evidence Filtering**: Prioritizes chunks containing prerequisite/eligibility keywords.
- **Intent Classifier**: Uses an LLM to distinguish between `eligibility`, `planning`, and `information` queries.
- **Strict Answer Generation**: Instructs the LLM to only use provided context, default to "I don't know" when missing, and guarantee page-level citations.
- **Multi-LLM Support**: Primary Gemini LLM with Groq fallback.
- **Gradio UI**: Modern, accessible chat-like interface.

## Setup Instructions

### 1. Prerequisites
- **Python 3.12+**
- An API Key from Google (Gemini) or Groq.

### 2. Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GEMINI_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

### 3. Installation
Create a virtual environment and install the dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

*(Note: Langchain text splitters package `langchain-text-splitters` is additionally requested).*

### 4. Running the Application

Before answering queries, the system must process the documents into a FAISS vector store.

**Step 1: Ingest Documents**
```bash
python ingest.py
```
This script reads all PDFs and DOCX files from the dataset directories, splits them into 500-character chunks with overlap, and generates the `vector_store/` FAISS index.

**Step 2: Start the UI Server**
```bash
python app.py
```
This loads the Gradio web interface at `http://127.0.0.1:7860/`. A public sharing link is also generated and printed to the terminal.

## Example Queries to Try

- **Information:** "What topics does Data Structures cover?"
- **Eligibility:** "Can I take Operating Systems?" (Include completed courses)
- **Planning:** "What should I take next semester?"

## Evaluation Results

- **Citation Coverage Rate:** 100% (Enforced by prompt + post-processing fallback)
- **Eligibility Correctness:** System correctly correlates course names in prerequisites context and successfully infers eligibility rules like "None" or specific chained dependencies.
- **Abstention Accuracy:** The system correctly outputs "I don't know" when asked about out-of-scope policies (e.g., "What is quantum computing?") if no context exists, but properly answers using context if provided (e.g., QMQC course handout).
