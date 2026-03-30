
import re
from typing import List, Dict, Set, Tuple


def extract_prerequisites(context: str) -> Dict[str, List[str]]:
    prereq_map: Dict[str, List[str]] = {}
    lines = context.split("\n")
    current_source = ""

    for line in lines:
        line_stripped = line.strip()

       
        if line_stripped.startswith("[") and " - Page" in line_stripped:
            current_source = line_stripped.strip("[]").split(" - Page")[0].strip()
            continue

        prereq_match = re.search(
            r"[Pp]re[\-\s]?[Rr]equisite[s]?\s*[:]\s*(?:if any\s*)?(.+)",
            line_stripped,
        )
        if prereq_match:
            prereq_text = prereq_match.group(1).strip()

            if not prereq_text or prereq_text.lower() in ("none", "nil", "na", "n/a", "-", "if any"):
                prereq_map[current_source] = []
                continue

            prereqs = re.split(r"[,;\n]|\band\b", prereq_text)
            prereqs = [p.strip().rstrip(".") for p in prereqs if p.strip() and len(p.strip()) > 1]
            prereq_map[current_source] = prereqs

    return prereq_map


def check_eligibility(
    course_name: str,
    completed_courses: List[str],
    context: str,
) -> Dict:
    
    prereq_map = extract_prerequisites(context)

    completed_lower = {c.strip().lower() for c in completed_courses if c.strip()}

    required: List[str] = []
    for source, prereqs in prereq_map.items():
        if (
            course_name.lower() in source.lower()
            or source.lower() in course_name.lower()
        ):
            required = prereqs
            break

    if not required and prereq_map:

        required = list(prereq_map.values())[0]

    
    missing = []
    completed_match = []
    for req in required:
        req_lower = req.lower()
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
    
    prereq_map = extract_prerequisites(context)
    completed_lower = {c.strip().lower() for c in completed_courses if c.strip()}

    suggestions = []
    for source, prereqs in prereq_map.items():
        if not prereqs:
            
            suggestions.append({
                "course": source,
                "prerequisites": [],
                "status": "No prerequisites required",
            })
            continue

        
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
