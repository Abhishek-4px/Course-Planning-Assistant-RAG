import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
from config import VECTOR_STORE_DIR
from ingest import run_ingestion, load_vector_store
from retriever import retrieve_and_prepare
from intent_classifier import classify_intent
from llm_answer import generate_answer

def get_vectorstore():
    index_path = VECTOR_STORE_DIR / 
    if index_path.exists():
        print("Loading existing vector store")
        return load_vector_store()
    else:
        print("Vector store not found. Running ingestion ")
        run_ingestion()
        return load_vector_store()


print("Initializing Course Planning Assistant ")
vectorstore = get_vectorstore()
print("Ready!\n")


def answer_query(query: str, completed_courses_str: str) -> str:
    if not query.strip():
        return "Please enter a query."

    
    completed_courses = [
        c.strip() for c in completed_courses_str.split(",")
        if c.strip()
    ] if completed_courses_str.strip() else []

    try:
        intent = classify_intent(query)
        retrieval = retrieve_and_prepare(vectorstore, query)
        answer = generate_answer(
            query=query,
            intent=intent,
            context=retrieval["context"],
            citations=retrieval["citations"],
            completed_courses=completed_courses,
        )
        return f"{answer}\n\n---\nDetected Intent: {intent}"

    except Exception as e:
        return f"An error occurred: {str(e)}\n\nPlease try rephrasing your query."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Course Planning Assistant")
    gr.Markdown(
        "Ask questions about courses, check prerequisites, or plan your next semester. "
        "All answers are grounded in official course handouts, program guides, and college policies — "
        "with page-level citations."
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            query_input = gr.Textbox(
                label="Your Query",
                placeholder="e.g., Can I take Data Structures? / What should I take next? / What is DBMS about?",
                lines=2,
            )
            completed_input = gr.Textbox(
                label="Completed Courses (comma-separated)",
                placeholder="e.g., Python, Mathematics, C Programming",
                lines=1,
            )
            submit_btn = gr.Button("Submit", variant="primary")
            gr.DeepLinkButton("Share This Response", variant="secondary")
            
        with gr.Column(scale=4):
            output = gr.Textbox(label="Answer", lines=15)
            
    submit_btn.click(fn=answer_query, inputs=[query_input, completed_input], outputs=output)
    
    gr.Examples(
        examples=[
            ["Can I take Data Structures and Algorithms?", "Mathematics, C Programming"],
            ["What should I take next semester?", "Python, DSA, Mathematics, DBMS"],
            ["What topics does Operating Systems cover?", ""],
            ["What is the grading policy?", ""],
            ["Can I take Deep Learning?", "Python, Statistics, Machine Learning"],
        ],
        inputs=[query_input, completed_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)
