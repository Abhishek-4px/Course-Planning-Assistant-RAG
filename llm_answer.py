"""
LLM Answer Generation Module
- Strict prompt: answer ONLY from context
- Always includes citations
- Says "I don't know" when info is missing
"""
from typing import List, Dict, Any

from llm_client import call_llm
from reasoning import check_eligibility, suggest_next_courses


# ── Prompt Templates ──────────────────────────────────


INFORMATION_PROMPT = """You are a university Course Planning Assistant.

STRICT RULES:
1. Answer the student's question using ONLY the context provided below.
2. If the information is NOT in the context, say exactly: "I don't have enough information to answer that based on the available documents."
3. Do NOT use any external knowledge. Do NOT guess or hallucinate.
4. Always end your response with a "Sources:" section listing the document name and page number for each piece of information you used.
5. Use the citation format: [Document Name - Page X]

CONTEXT:
{context}

STUDENT QUERY: {query}

Provide a clear, helpful answer with citations:"""


ELIGIBILITY_PROMPT = """You are a university Course Planning Assistant checking course eligibility.

STRICT RULES:
1. Use ONLY the context and eligibility analysis provided below.
2. Do NOT use external knowledge. Do NOT guess.
3. If information is missing, say: "I don't have enough information to determine eligibility based on the available documents."
4. Always include citations in format: [Document Name - Page X]

CONTEXT:
{context}

ELIGIBILITY ANALYSIS:
{eligibility_info}

STUDENT QUERY: {query}
COMPLETED COURSES: {completed}

Provide a clear eligibility assessment with citations:"""


PLANNING_PROMPT = """You are a university Course Planning Assistant helping with course planning.

STRICT RULES:
1. Use ONLY the context and planning analysis provided below.
2. Do NOT use external knowledge. Do NOT guess.
3. If information is missing, say: "I don't have enough information to make recommendations based on the available documents."
4. Always include citations in format: [Document Name - Page X]

CONTEXT:
{context}

PLANNING ANALYSIS:
{planning_info}

STUDENT QUERY: {query}
COMPLETED COURSES: {completed}

Provide clear course recommendations with citations:"""


# ── Answer Generators ─────────────────────────────────


def answer_information(query: str, context: str, citations: List[str]) -> str:
    """Generate an information-type answer."""
    prompt = INFORMATION_PROMPT.format(context=context, query=query)
    answer = call_llm(prompt)

    # Append citations if the LLM didn't include them
    if "Sources:" not in answer and "sources:" not in answer.lower():
        cite_block = "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        answer += cite_block

    return answer


def answer_eligibility(
    query: str,
    context: str,
    citations: List[str],
    completed_courses: List[str],
) -> str:
    """Generate an eligibility-type answer."""
    # Run prerequisite reasoning
    # Extract course name from query (rough heuristic)
    course_name = query  # the LLM will parse it from context anyway
    elig = check_eligibility(course_name, completed_courses, context)

    eligibility_info = []
    if elig["required"]:
        eligibility_info.append(f"Required prerequisites: {', '.join(elig['required'])}")
    else:
        eligibility_info.append("No explicit prerequisites found in the retrieved documents.")

    if elig["missing"]:
        eligibility_info.append(f"Missing prerequisites: {', '.join(elig['missing'])}")
    elif elig["required"]:
        eligibility_info.append("All prerequisites are satisfied.")

    if elig["completed_match"]:
        eligibility_info.append(f"Already completed: {', '.join(elig['completed_match'])}")

    eligibility_info.append(f"Eligible: {'Yes' if elig['eligible'] else 'No'}")

    elig_text = "\n".join(eligibility_info)
    completed_text = ", ".join(completed_courses) if completed_courses else "None specified"

    prompt = ELIGIBILITY_PROMPT.format(
        context=context,
        eligibility_info=elig_text,
        query=query,
        completed=completed_text,
    )
    answer = call_llm(prompt)

    if "Sources:" not in answer and "sources:" not in answer.lower():
        cite_block = "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        answer += cite_block

    return answer


def answer_planning(
    query: str,
    context: str,
    citations: List[str],
    completed_courses: List[str],
) -> str:
    """Generate a planning-type answer."""
    suggestions = suggest_next_courses(completed_courses, context)

    if suggestions:
        planning_info = "Courses you may be eligible for:\n"
        for s in suggestions:
            prereqs = ", ".join(s["prerequisites"]) if s["prerequisites"] else "None"
            planning_info += f"- {s['course']} (Prerequisites: {prereqs}) — {s['status']}\n"
    else:
        planning_info = "No specific course suggestions could be derived from the retrieved documents."

    completed_text = ", ".join(completed_courses) if completed_courses else "None specified"

    prompt = PLANNING_PROMPT.format(
        context=context,
        planning_info=planning_info,
        query=query,
        completed=completed_text,
    )
    answer = call_llm(prompt)

    if "Sources:" not in answer and "sources:" not in answer.lower():
        cite_block = "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)
        answer += cite_block

    return answer


def generate_answer(
    query: str,
    intent: str,
    context: str,
    citations: List[str],
    completed_courses: List[str],
) -> str:
    """
    Route to the correct answer generator based on intent.
    """
    if intent == "eligibility":
        return answer_eligibility(query, context, citations, completed_courses)
    elif intent == "planning":
        return answer_planning(query, context, citations, completed_courses)
    else:
        return answer_information(query, context, citations)
