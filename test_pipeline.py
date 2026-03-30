"""Quick test of the full pipeline."""
import sys, os, warnings
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
warnings.filterwarnings('ignore')

from ingest import load_vector_store
from retriever import retrieve_and_prepare
from intent_classifier import classify_intent
from llm_answer import generate_answer

vs = load_vector_store()
output = []

# Test 1: Information query
query = "What are the prerequisites for Data Structures?"
intent = classify_intent(query)
result = retrieve_and_prepare(vs, query)
answer = generate_answer(query, intent, result['context'], result['citations'], [])
output.append(f"=== TEST 1: {query} ===")
output.append(f"Intent: {intent}")
output.append(f"Answer:\n{answer}\n")

# Test 2: Eligibility query
query = "Can I take Operating Systems?"
completed = ["C Programming", "Data Structures", "Mathematics"]
intent = classify_intent(query)
result = retrieve_and_prepare(vs, query)
answer = generate_answer(query, intent, result['context'], result['citations'], completed)
output.append(f"=== TEST 2: {query} ===")
output.append(f"Completed: {completed}")
output.append(f"Intent: {intent}")
output.append(f"Answer:\n{answer}\n")

# Test 3: Out-of-scope (should say "I don't know")
query = "What is quantum computing?"
intent = classify_intent(query)
result = retrieve_and_prepare(vs, query)
answer = generate_answer(query, intent, result['context'], result['citations'], [])
output.append(f"=== TEST 3: {query} ===")
output.append(f"Intent: {intent}")
output.append(f"Answer:\n{answer}\n")

text = "\n".join(output)
print(text)
with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(text)
print("\n[Saved to test_output.txt]")
