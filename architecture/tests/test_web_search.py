"""
Evaluation script for web_search_answer() across diverse query types.
Run from the 15.LexiChat directory or the repo root.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from architecture.web_search.web_search import web_search_answer

# ── Test queries ──────────────────────────────────────────────────────────────
QUERIES = [
    # (label, question)

    # 1. Simple factual
    ("Factual — person",         "Who is MS Dhoni?"),

    # 2. Ambiguous abbreviation (the original bug trigger)
    ("Ambiguous abbreviation",   "What is ML?"),

    # 3. Complex / multi-part
    ("Complex multi-part",       "What are the differences between deep learning, machine learning, and AI, and which industries use each the most?"),

    # 4. Current events
    ("Current event",            "What happened in the 2025 ICC Cricket World Cup final?"),

    # 5. Hybrid — mix of named entity + concept
    ("Hybrid entity+concept",    "How did Elon Musk influence the electric vehicle market and what is Tesla's current market share?"),

    # 6. Numeric / statistical
    ("Numeric / statistical",    "What is the current population of India and its growth rate?"),

    # 7. How-to / procedural
    ("How-to procedural",        "How do I set up a RAG pipeline using LangChain and ChromaDB?"),

    # 8. Comparison
    ("Comparison",               "Compare Python and Rust for systems programming"),

    # 9. Very short / vague
    ("Vague short query",        "best cricket player"),

    # 10. Obscure / niche
    ("Obscure niche",            "What is the Havana Syndrome and what causes it?"),
]

# ── Runner ────────────────────────────────────────────────────────────────────
SEP = "=" * 80

def truncate(text: str, n: int = 400) -> str:
    return text[:n] + " …[truncated]" if len(text) > n else text

def evaluate(label: str, question: str) -> None:
    print(f"\n{SEP}")
    print(f"[{label}]")
    print(f"Q: {question}")
    print(SEP)

    result = web_search_answer(question)

    if result["error"] and not result["answer"]:
        print(f"  ❌  ERROR: {result['error']}")
        return

    answer = result["answer"] or "(no answer)"
    sources = result["sources"]

    print(f"  ANSWER:\n{truncate(answer)}\n")
    print(f"  SOURCES ({len(sources)}):")
    for i, url in enumerate(sources[:5], 1):
        print(f"    {i}. {url}")

    # Basic quality checks
    flags = []
    lower = answer.lower()
    if "i don't know" in lower or "cannot answer" in lower:
        flags.append("⚠️  model refused")
    if "multiple sclerosis" in lower and "dhoni" in question.lower():
        flags.append("❌  wrong topic (MS ≠ Dhoni)")
    if len(answer) < 80:
        flags.append("⚠️  very short answer")
    if not sources:
        flags.append("⚠️  no sources returned")

    if flags:
        print(f"\n  FLAGS: {' | '.join(flags)}")
    else:
        print("  ✅  Passed basic checks")


if __name__ == "__main__":
    print(f"\nRunning web_search evaluation on {len(QUERIES)} queries...\n")
    for label, question in QUERIES:
        evaluate(label, question)
    print(f"\n{SEP}\nDone.\n")
