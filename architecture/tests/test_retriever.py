"""
Test script for retriever_pipeline2.py — runs ask() on diverse query types
outside of Streamlit by patching st.cache_resource before import.

Categories tested:
  IN-CONTEXT   — topics that SHOULD be in the video transcripts
  OUT-OF-CONTEXT — topics that should NOT be there → expect "I don't know"
  AMBIGUOUS    — short/vague queries
  COMPLEX      — multi-part, compound questions
"""

import sys, os

# ── Patch Streamlit BEFORE importing the pipeline ────────────────────────────
# st.cache_resource is a decorator; outside Streamlit it must be a no-op.
from unittest.mock import MagicMock
import unittest.mock as mock

streamlit_mock = MagicMock()
streamlit_mock.cache_resource = lambda **kw: (lambda fn: fn)   # passthrough
streamlit_mock.cache_resource.side_effect = None
sys.modules["streamlit"] = streamlit_mock

# ── sys.path: point to 15.LexiChat root ──────────────────────────────────────
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from architecture.retrieval.retriever_pipeline2 import ask   # noqa: E402


# ── Test queries ──────────────────────────────────────────────────────────────
QUERIES = [
    # (category, label, question)

    # ── IN-CONTEXT (expects real answers with timestamps) ──────────────────
    ("IN-CONTEXT",    "AI basics",            "What is artificial intelligence?"),
    ("IN-CONTEXT",    "Geopolitics",          "What is geopolitics?"),
    ("IN-CONTEXT",    "Habits/growth",        "What habits are emphasized for sustained improvement?"),
    ("IN-CONTEXT",    "Problem solving",      "How is problem-solving framed in the context of complex systems?"),

    # ── OUT-OF-CONTEXT (expects "I don't know") ────────────────────────────
    ("OUT-OF-CONTEXT","Person not in corpus", "Who is Vinod Khosla?"),
    ("OUT-OF-CONTEXT","Unrelated topic",      "What is quantum computing?"),
    ("OUT-OF-CONTEXT","Sports not in corpus", "Who won the 2025 IPL?"),

    # ── AMBIGUOUS ──────────────────────────────────────────────────────────
    ("AMBIGUOUS",     "Single word",          "leadership"),
    ("AMBIGUOUS",     "Pronoun reference",    "What did he say about the future?"),
    ("AMBIGUOUS",     "Broad open-ended",     "What is the main idea?"),

    # ── COMPLEX / MULTI-PART ───────────────────────────────────────────────
    ("COMPLEX",       "Multi-part",           "What is AI and how does it relate to geopolitics and decision-making?"),
    ("COMPLEX",       "Compare/contrast",     "What are the differences between different ways of thinking discussed in the videos?"),
    ("COMPLEX",       "Causal chain",         "How do cultural values influence political decisions according to the transcripts?"),
]


# ── Evaluation helpers ────────────────────────────────────────────────────────
SEP  = "=" * 80
SEP2 = "-" * 80

DONT_KNOW_PHRASES = [
    "i don't know",
    "transcripts do not contain",
    "do not contain information",
]

def is_dont_know(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in DONT_KNOW_PHRASES)

def has_timestamp_links(answer: str) -> bool:
    import re
    return bool(re.search(r'\[\[\d{2}:\d{2}\]\]\(https?://', answer))

def truncate(text: str, n: int = 500) -> str:
    return text[:n] + "\n  …[truncated]" if len(text) > n else text

def evaluate(category: str, label: str, question: str) -> dict:
    print(f"\n{SEP}")
    print(f"[{category}] {label}")
    print(f"Q: {question}")
    print(SEP2)

    result = ask(question)
    answer = result.get("answer", "")
    n_ctx  = len(result.get("retrieved_contexts", []))

    print(f"Retrieved chunks : {n_ctx}")
    print(f"Answer:\n  {truncate(answer.strip())}\n")

    # ── Quality checks ────────────────────────────────────────────────────
    flags   = []
    passed  = []

    dont_know = is_dont_know(answer)
    has_ts    = has_timestamp_links(answer)

    if category == "OUT-OF-CONTEXT":
        if dont_know:
            passed.append("✅ Correctly refused (I don't know)")
        else:
            flags.append("❌ HALLUCINATION — should have said I don't know")

    elif category in ("IN-CONTEXT", "AMBIGUOUS", "COMPLEX"):
        if dont_know:
            # Might be legitimate if topic not in DB — flag as warning, not error
            flags.append("⚠️  Returned 'I don't know' — topic may not be in corpus")
        else:
            passed.append("✅ Returned an answer")
            if has_ts:
                passed.append("✅ Contains timestamp links")
            else:
                flags.append("⚠️  No [[MM:SS]](URL) timestamp links found")

            lower = answer.lower()
            meta_phrases = ["the chunk mentions", "the discussion is", "the context mentions",
                            "it is mentioned that", "the transcripts mention",
                            "found at https", "discussion about"]
            meta_hits = [p for p in meta_phrases if p in lower]
            if meta_hits:
                flags.append(f"⚠️  Meta-commentary detected: {meta_hits}")
            else:
                passed.append("✅ No meta-commentary")

            spec_phrases = ["could be related", "might be", "possibly", "more research needed",
                            "i cannot confirm", "without more information"]
            spec_hits = [p for p in spec_phrases if p in lower]
            if spec_hits:
                flags.append(f"❌ Speculation detected: {spec_hits}")
            else:
                passed.append("✅ No speculation")

    for p in passed:
        print(f"  {p}")
    for f in flags:
        print(f"  {f}")

    return {
        "category": category,
        "label": label,
        "dont_know": dont_know,
        "has_timestamps": has_ts,
        "flags": flags,
        "passed": passed,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nRunning retriever pipeline tests on {len(QUERIES)} queries...\n")

    results = []
    for i, (category, label, question) in enumerate(QUERIES):
        if i > 0:
            time.sleep(15)   # Gemini free tier: 5 RPM — wait between queries
        r = evaluate(category, label, question)
        results.append(r)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    total   = len(results)
    errors  = sum(1 for r in results if any("❌" in f for f in r["flags"]))
    warnings= sum(1 for r in results if any("⚠️" in f for f in r["flags"]))
    clean   = total - errors - warnings

    print(f"  Total  : {total}")
    print(f"  ✅ Clean  : {clean}")
    print(f"  ⚠️  Warnings: {warnings}")
    print(f"  ❌ Errors   : {errors}")

    if errors:
        print(f"\nFailed queries:")
        for r in results:
            if any("❌" in f for f in r["flags"]):
                print(f"  [{r['category']}] {r['label']}")
                for f in r["flags"]:
                    if "❌" in f:
                        print(f"    {f}")
    print(f"\n{SEP}\nDone.\n")
