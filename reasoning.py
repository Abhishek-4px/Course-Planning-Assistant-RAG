"""
Prerequisite Reasoning Module
- Extracts prerequisite relationships from context
- Checks eligibility given completed courses
- Suggests next courses for planning
"""
import re
from typing import List, Dict, Set, Tuple


def extract_prerequisites(context: str) -> Dict[str, List[str]]:
    """
    Parse retrieved context to extract prerequisite mappings.
    Looks for patterns like:
        Prerequisite(s): Course A, Course B
        Pre-requisite: if any ...
        Prerequisites: None

    Returns dict: { "Course Name": ["Prereq1", "Prereq2"] }
    """
    prereq_map: Dict[str, List[str]] = {}

    # Try to find course name and its prerequisites from the context
    lines = context.split("\n")
    current_source = ""

    for line in lines:
        line_stripped = line.strip()

        # Track current document source
        if line_stripped.startswith("[") and " - Page" in line_stripped:
            current_source = line_stripped.strip("[]").split(" - Page")[0].strip()
            continue

        # Look for prerequisite patterns
        prereq_match = re.search(
            r"[Pp]re[\-\s]?[Rr]equisite[s]?\s*[:]\s*(?:if any\s*)?(.+)",
            line_stripped,
        )
        if prereq_match:
            prereq_text = prereq_match.group(1).strip()

            # Skip empty / "None" / "if any" only
            if not prereq_text or prereq_text.lower() in ("none", "nil", "na", "n/a", "-", "if any"):
                prereq_map[current_source] = []
                continue

            # Split by comma, "and", or newline
            prereqs = re.split(r"[,;\n]|\band\b", prereq_text)
            prereqs = [p.strip().rstrip(".") for p in prereqs if p.strip() and len(p.strip()) > 1]
            prereq_map[current_source] = prereqs

    return prereq_map


def check_eligibility(
    course_name: str,
    completed_courses: List[str],
    context: str,
) -> Dict:
    """
    Check if a student can take a given course, based on extracted prerequisites.

    Returns dict with:
        - eligible: bool
        - required: list of required prerequisites
        - missing: list of prerequisites not yet completed
        - completed_match: list of prerequisites already completed
    """
    prereq_map = extract_prerequisites(context)

    # Normalize completed courses for comparison
    completed_lower = {c.strip().lower() for c in completed_courses if c.strip()}

    # Find prerequisites for the target course (fuzzy match on source name)
    required: List[str] = []
    for source, prereqs in prereq_map.items():
        if (
            course_name.lower() in source.lower()
            or source.lower() in course_name.lower()
        ):
            required = prereqs
            break

    if not required and prereq_map:
        # If exact match fails, use the first entry
        required = list(prereq_map.values())[0]

    # Compare
    missing = []
    completed_match = []
    for req in required:
        req_lower = req.lower()
        # Check if any completed course matches (partial match)
        matched = any(
            req_lower in comp or comp in req_lower
            for comp in completed_lower
        )
        if matched:
            completed_match.append(req)
        else:
            missing.append(req)

    return {
        "eligible": len(missing) == 0,
        "required": required,
        "missing": missing,
        "completed_match": completed_match,
    }


def suggest_next_courses(
    completed_courses: List[str],
    context: str,
) -> List[Dict]:
    """
    Based on the context, suggest courses whose prerequisites are
    satisfied by the student's completed courses.

    Returns list of dicts with course info.
    """
    prereq_map = extract_prerequisites(context)
    completed_lower = {c.strip().lower() for c in completed_courses if c.strip()}

    suggestions = []
    for source, prereqs in prereq_map.items():
        if not prereqs:
            # No prerequisites — always eligible
            suggestions.append({
                "course": source,
                "prerequisites": [],
                "status": "No prerequisites required",
            })
            continue

        # Check if all prerequisites are met
        all_met = True
        for req in prereqs:
            req_lower = req.lower()
            if not any(req_lower in comp or comp in req_lower for comp in completed_lower):
                all_met = False
                break

        if all_met:
            suggestions.append({
                "course": source,
                "prerequisites": prereqs,
                "status": "All prerequisites satisfied",
            })

    return suggestions
