"""
Centralised Langfuse observability for LexiChat.

Provides:
  - A singleton Langfuse client (initialised lazily, no-op when keys are absent)
  - Helper wrappers that create traces / spans / generations with consistent
    naming so every RAG request flows through a single trace tree:
        trace (one per user question)
         ├─ span  "retrieval"
         │   ├─ span "mmr_retrieve"
         │   ├─ span "multiquery_retrieve"
         │   ├─ span "bm25_retrieve"
         │   ├─ span "combine_dedup"
         │   └─ span "cross_encoder_rerank"
         ├─ generation "answer_llm"
         ├─ span  "evaluation"      (optional)
         └─ span  "web_search"      (optional)
  - Cost tracking utilities for Groq / Gemini free-tier usage
  - Score helpers that push RAGAS metrics back to the trace
  - Graceful degradation: every public function is a no-op when Langfuse
    is not configured, so the app runs identically without keys.
"""

import os
import time
import logging
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_langfuse_client = None
_initialised = False


def _get_client():
    """Return the Langfuse client, or None if not configured."""
    global _langfuse_client, _initialised
    if _initialised:
        return _langfuse_client
    _initialised = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info("[langfuse] Keys not set — observability disabled")
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            enabled=True,
        )
        logger.info("[langfuse] Client initialised (%s)", host)
    except Exception as exc:
        logger.warning("[langfuse] Failed to init: %s", exc)
        _langfuse_client = None

    return _langfuse_client


def flush():
    """Flush any pending events (call at the end of a request cycle)."""
    client = _get_client()
    if client:
        try:
            client.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def create_trace(*, name: str, user_id: str = "anonymous",
                 session_id: Optional[str] = None,
                 metadata: Optional[dict] = None,
                 tags: Optional[list] = None):
    """Create a top-level trace for one user question. Returns trace or None."""
    client = _get_client()
    if not client:
        return None
    try:
        return client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
            tags=tags or [],
        )
    except Exception as exc:
        logger.debug("[langfuse] trace error: %s", exc)
        return None


def start_span(trace, *, name: str, input: Any = None,
               metadata: Optional[dict] = None):
    """Start a child span on a trace or parent span. Returns span or None."""
    if trace is None:
        return None
    try:
        return trace.span(name=name, input=input, metadata=metadata or {})
    except Exception:
        return None


def end_span(span, *, output: Any = None, metadata: Optional[dict] = None,
             level: str = "DEFAULT"):
    """End a span with output data."""
    if span is None:
        return
    try:
        span.end(output=output, metadata=metadata or {}, level=level)
    except Exception:
        pass


def log_generation(trace, *, name: str, model: str,
                   input: Any = None, output: Any = None,
                   usage: Optional[dict] = None,
                   metadata: Optional[dict] = None,
                   model_parameters: Optional[dict] = None):
    """Log an LLM generation (prompt + response + token counts)."""
    if trace is None:
        return None
    try:
        return trace.generation(
            name=name,
            model=model,
            input=input,
            output=output,
            usage=usage or {},
            metadata=metadata or {},
            model_parameters=model_parameters or {},
        )
    except Exception:
        return None


def score_trace(trace, *, name: str, value: float,
                comment: Optional[str] = None):
    """Attach a numeric score to a trace (e.g. RAGAS metrics, confidence)."""
    if trace is None:
        return
    try:
        trace.score(name=name, value=value, comment=comment or "")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Timer context manager (measures wall-clock ms)
# ---------------------------------------------------------------------------

@contextmanager
def timed_span(trace, *, name: str, input: Any = None,
               metadata: Optional[dict] = None):
    """
    Context manager that creates a span, yields (span, timing_dict), and
    auto-ends the span with duration metadata.

    Usage:
        with timed_span(trace, name="rerank") as (span, timing):
            ...do work...
        # timing["duration_ms"] is available after the block
    """
    timing: dict[str, float] = {}
    span = start_span(trace, name=name, input=input, metadata=metadata)
    t0 = time.perf_counter()
    try:
        yield span, timing
    finally:
        dur = (time.perf_counter() - t0) * 1000
        timing["duration_ms"] = dur
        end_span(span, metadata={"duration_ms": round(dur, 1)})


# ---------------------------------------------------------------------------
# Token-cost estimation (Groq free-tier is $0, but we track for visibility)
# ---------------------------------------------------------------------------

# Approximate cost per 1M tokens (USD). Groq free-tier = $0 but we record
# the theoretical cost so the dashboard shows what it *would* cost at scale.
_COST_PER_1M = {
    "llama-3.3-70b-versatile":  {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":     {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b":      {"input": 0.00, "output": 0.00},  # Groq free
    "qwen/qwen3-32b":           {"input": 0.00, "output": 0.00},  # Groq free
    "gemini-2.5-flash":         {"input": 0.00, "output": 0.00},  # Google free
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for a generation."""
    rates = _COST_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
