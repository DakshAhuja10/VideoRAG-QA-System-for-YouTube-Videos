"""
Web search fallback for LexiChat.

Used when the RAG system detects it cannot answer from the transcript
knowledge base ("I don't know" response).  Searches DuckDuckGo, scrapes
the top results, and passes the cleaned text to Groq to produce an answer.

No extra dependencies beyond what LexiChat already requires:
  - ddgs        (DuckDuckGo search)
  - httpx       (HTTP client)
  - beautifulsoup4  (HTML parsing)
  - langchain-groq  (LLM answer generation)
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_groq import ChatGroq

from config import LLMConfig

# Observability
from architecture.observability.langfuse_tracing import (
    start_span, end_span, log_generation, timed_span,
    flush as langfuse_flush, estimate_cost,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Characters to keep per source page (avoids flooding the LLM context)
_MAX_CHARS_PER_PAGE = 6000

# Number of URLs to fetch (more sources = richer, more detailed answer)
_MAX_URLS = 5

_ANSWER_PROMPT = """You are a knowledgeable assistant answering a question using web search results.

Write a **detailed, well-structured answer** using ONLY the information from the search results below.

Guidelines:
- Cover all key facts, background context, and notable details found across the sources.
- Use multiple paragraphs or a short list when the topic has several distinct aspects.
- Aim for at least 4-6 sentences; more if the topic warrants it.
- Do NOT include source URLs or citations inside the answer — they will be shown separately.
- If the results contain conflicting information, note the discrepancy.
- IMPORTANT: If the search results are NOT about the same person/topic as the question, say "The search did not return relevant results for this question." and stop. Do not answer using off-topic results.

Search Results:
{context}

Question: {question}

Detailed Answer:"""

_QUERY_REWRITE_PROMPT = """Convert the following question into an optimal web search query.
- Make it specific and unambiguous (e.g. expand abbreviations if context suggests a person or place)
- Remove filler words like "who is", "what is", "tell me about"
- Output ONLY the search query, nothing else.

Question: {question}
Search query:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _search_ddg(query: str, max_results: int = _MAX_URLS) -> list[dict]:
    """
    Return top unique results from DuckDuckGo as dicts with 'href' and 'body'.
    The 'body' field is DDG's own text snippet — always present even when
    scraping the page later fails.  Retries once after 4 s on rate-limit.
    """
    _MAX_ATTEMPTS = 2
    _RETRY_WAIT   = 4  # seconds

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            raw = DDGS().text(query, max_results=max_results)
            seen: set = set()
            results = []
            for r in raw:
                href = r.get("href", "")
                if href and href not in seen:
                    seen.add(href)
                    results.append({"href": href, "body": r.get("body", "")})
            return results
        except Exception:
            if attempt < _MAX_ATTEMPTS:
                time.sleep(_RETRY_WAIT)
            else:
                return []


def _fetch_page_text(url: str, timeout: float = 12.0) -> str:
    """
    Fetch a URL and extract readable text using BeautifulSoup.
    Strips scripts, styles, navbars, headers, and footers.
    Returns empty string on failure.
    """
    try:
        with httpx.Client(
            headers=_HEADERS,
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html = response.text

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        # Collapse multiple spaces/newlines
        import re
        text = re.sub(r"\s{2,}", " ", text)
        return text[:_MAX_CHARS_PER_PAGE]

    except Exception:
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def web_search_answer(question: str, trace=None) -> dict:
    """
    Search DuckDuckGo, scrape top results, and generate an answer via Groq.

    Scraping strategy (two-tier):
    - Primary: full page scrape via httpx + BeautifulSoup (rich content)
    - Fallback: DDG's own 'body' snippet (always available, no scraping needed)

    Returns a dict with:
      - "answer"  : str | None  — the generated answer
      - "sources" : list[str]   — URLs that were used
      - "error"   : str | None  — error message if something failed
    """
    ws_span = start_span(trace, name="web_search", input={"question": question})

    # 1. Rewrite the question into a better search query
    llm = None
    try:
        llm = ChatGroq(model=LLMConfig.ANSWER_MODEL, temperature=0)
        with timed_span(ws_span, name="query_rewrite") as (qr_span, _):
            search_query = llm.invoke(
                _QUERY_REWRITE_PROMPT.format(question=question)
            ).content.strip().strip('"').strip("'")
            end_span(qr_span, output={"search_query": search_query})
    except Exception:
        search_query = question  # fall back to raw question

    # 2. Search
    with timed_span(ws_span, name="ddg_search", input={"query": search_query}) as (ddg_span, _):
        results = _search_ddg(search_query)
        end_span(ddg_span, output={"result_count": len(results)})
    if not results:
        return {
            "answer": None,
            "sources": [],
            "error": "DuckDuckGo returned no results. Please try again in a moment.",
        }

    # 3. Build source blocks — full scrape first, DDG snippet as fallback
    source_blocks = []
    scraped_urls = []
    with timed_span(ws_span, name="scrape_pages", input={"url_count": len(results)}) as (scrape_span, _):
        for r in results:
            url  = r["href"]
            body = r["body"]   # DDG snippet — always non-empty

            # Try full page scrape for richer content
            page_text = _fetch_page_text(url)
            text = page_text if page_text else body  # fallback to DDG snippet

            if text:
                source_blocks.append(f"Source: {url}\n\n{text}")
                scraped_urls.append(url)
        end_span(scrape_span, output={"scraped_count": len(scraped_urls)})

    if not source_blocks:
        end_span(ws_span, output={"error": "no content scraped"}, level="WARNING")
        return {
            "answer": None,
            "sources": [r["href"] for r in results],
            "error": "Could not retrieve content from the search results.",
        }

    # 4. Answer with Groq
    context = "\n\n---\n\n".join(source_blocks)
    prompt = _ANSWER_PROMPT.format(context=context, question=question)

    try:
        # Reuse the same llm instance (or create a fresh one if rewrite step failed)
        if llm is None:
            llm = ChatGroq(model=LLMConfig.ANSWER_MODEL, temperature=0)
        raw_answer = llm.invoke(prompt).content

        log_generation(
            ws_span, name="web_answer_llm", model=LLMConfig.ANSWER_MODEL,
            input=prompt[:500], output=raw_answer[:500],
            metadata={"sources": scraped_urls},
        )

        # Strip any trailing "Sources:" / "Source URLs:" block the LLM appends
        import re
        answer = re.split(
            r"\n\s*\*{0,2}source(?:s| urls?)?\*{0,2}\s*:?",
            raw_answer,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip()

        end_span(ws_span, output={"answer_len": len(answer), "sources": scraped_urls})
        langfuse_flush()
        return {"answer": answer, "sources": scraped_urls, "error": None}
    except Exception as e:
        end_span(ws_span, output={"error": str(e)}, level="ERROR")
        langfuse_flush()
        return {"answer": None, "sources": scraped_urls, "error": str(e)}
