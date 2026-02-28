# 🎥 LexiChat — YouTube Video RAG System

LexiChat is an **end-to-end Retrieval-Augmented Generation (RAG) system** for **grounded, timestamped question answering over YouTube video transcripts**.

It ingests YouTube playlists, extracts transcripts with timestamps, builds a persistent vector database, and retrieves relevant context using a **hybrid retrieval pipeline** (MMR + MultiQuery + BM25) with **cross-encoder reranking**. Answers are strictly grounded to transcript content and come with **clickable YouTube timestamps**.

When the system cannot answer from its transcript knowledge base, an integrated **DuckDuckGo web-search fallback** lets users fetch an answer from the open web in one click.

---

## 🔗 Live Demo

👉 **[Try LexiChat on Streamlit Cloud](https://videorag-app-system-for-youtube-videos-3vpsxvnappai2vxtpjgon3.streamlit.app/)**

> ⚠️ Deployed on Streamlit Cloud (shared infrastructure).
> **Answer generation** is fast (~2–3 seconds).
> **RAG evaluation** takes ~2–2.5 minutes due to multiple sequential LLM calls on shared, rate-limited infrastructure.

---

## 📐 System Architecture

### High-Level Data Flow

`
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE (offline)                     │
│                                                                         │
│  YouTube URL / Playlist                                                 │
│        │                                                                │
│        ▼                                                                │
│  youtube_meta_data.py ──► yt-dlp: titles, channels, durations           │
│        │                  └──► architecture/data/videos.csv              │
│        ▼                                                                │
│  transcript_generate.py ──► youtube-transcript-api (yt-dlp fallback)    │
│        │                    └──► video_with_meta_data_and_transcript.csv │
│        ▼                                                                │
│  doc_loader.py ──► CSV → LangChain Documents + SHA-256 hash per chunk   │
│        │                                                                │
│        ▼                                                                │
│  vector_store_chroma.py ──► all-MiniLM-L6-v2 embeddings                 │
│                             Hash-based dedup → persist to chroma_db/    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE (runtime)                       │
│                                                                         │
│  User Question                                                          │
│        │                                                                │
│        ├──► Compound query? ──► _decompose_query() splits into          │
│        │                        sub-questions via Groq LLM              │
│        ▼                                                                │
│  ┌─────────────────────────────────────────────┐                        │
│  │  Per sub-question, 3 parallel retrievers:   │                        │
│  │                                             │                        │
│  │  ┌──────────┐ ┌───────────┐ ┌───────────┐  │                        │
│  │  │ MMR      │ │MultiQuery │ │  BM25     │  │                        │
│  │  │ k=6      │ │Gemini 2.5 │ │  k=6      │  │                        │
│  │  │fetch_k=20│ │Flash      │ │  keyword  │  │                        │
│  │  │relevance │ │query      │ │  matching  │  │                        │
│  │  │+diversity│ │expansion  │ │           │  │                        │
│  │  └────┬─────┘ └─────┬─────┘ └─────┬─────┘  │                        │
│  │       └──────────────┼────────────┘         │                        │
│  │                      ▼                       │                        │
│  │        combine_results() — deduplicate       │                        │
│  │                      ▼                       │                        │
│  │    CrossEncoder Reranker (ms-marco-MiniLM)   │                        │
│  │    Score each chunk against its sub-question  │                        │
│  │                      ▼                       │                        │
│  │            Top-N most relevant chunks         │                        │
│  └─────────────────────────────────────────────┘                        │
│        │                                                                │
│        ▼                                                                │
│  Relevance Gate (rerank_score < 0.5?)                                   │
│        │                                                                │
│   YES (relevant)              NO (irrelevant)                           │
│        │                            │                                   │
│        ▼                            ▼                                   │
│  Groq LLM (llama-3.1-8b)    "I don't know. The transcripts             │
│  Strict grounded prompt       do not contain information                │
│  3 numbered answers            about this topic."                       │
│  + clickable timestamps              │                                  │
│        │                            ▼                                   │
│        │                  DuckDuckGo web-search                         │
│        │                  fallback button shown                         │
│        ▼                                                                │
│  Answer streamed to UI                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE (on-demand)                       │
│                                                                         │
│  Ground Truth Generator ──► Groq qwen3-32b (independent model family)   │
│        │                                                                │
│        ▼                                                                │
│  RAGAS Evaluation (Judge: Groq openai/gpt-oss-120b)                     │
│  ├── Context Precision  (10% weight) — signal-to-noise in context       │
│  ├── Context Recall     (10% weight) — was needed info retrieved?       │
│  ├── Faithfulness       (40% weight) — grounded to context?             │
│  └── Answer Relevancy   (40% weight) — did it address the question?     │
│        │                                                                │
│        ▼                                                                │
│  Confidence Score = weighted sum (0.0 – 1.0)                            │
│  ├── < 0.4 → retry suggestion shown                                    │
│  ├── < 0.6 → auto-logged to rag_evaluation_log.csv                     │
│  └── Retry triggers hybrid_retrieve_broad() with wider k               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Streamlit)                             │
│                                                                         │
│  ├── Streamed answer display with clickable timestamps                  │
│  ├── TTS audio playback (Piper local / gTTS cloud — auto-selected)     │
│  ├── RAGAS evaluation panel with confidence visualization               │
│  ├── Retry-on-low-confidence with side-by-side comparison               │
│  ├── Evaluation history dashboard with trend charts                     │
│  └── Dynamic video ingestion (local only — disabled on Cloud)           │
└─────────────────────────────────────────────────────────────────────────┘
`

### Component Interaction Map

`
                    ┌─────────────┐
                    │   app.py    │  ◄── Streamlit entry point
                    └──────┬──────┘
                           │ imports
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      ┌───────────┐ ┌──────────┐ ┌──────────┐
      │ retriever  │ │evaluation│ │web_search│
      │ pipeline   │ │ pipeline │ │          │
      └──────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            │
      ┌───────────┐ ┌──────────┐      │
      │ config.py │ │  RAGAS   │      │
      │ (central) │ │  library │      │
      └──────┬─────┘ └──────────┘      │
             │                         │
             ▼                         ▼
      ┌───────────┐            ┌──────────────┐
      │ chroma_db │            │ DuckDuckGo + │
      │ (ChromaDB)│            │ httpx + Groq │
      └───────────┘            └──────────────┘
`

---

## ⚠️ Content Scope — Read Before Asking Questions

This is a **content-grounded system**, not a general-purpose chatbot. It can only answer questions about the **19 indexed YouTube videos**. Questions outside this scope return:

> **"I don't know. The transcripts do not contain information about this topic."**

This is **by design** — the system refuses to answer from its own parametric knowledge to prevent hallucination. When this happens, a **DuckDuckGo search** button appears to let you fetch an open-web answer.

### 🎥 Indexed Videos

**🧠 Lex Fridman — Personal Development & Mindset**
- [Do Something Difficult Every Day | AMA #1](https://www.youtube.com/watch?v=wKw1tpN7NVE)
- [How to Learn and Master a New Skill](https://www.youtube.com/watch?v=_ySbzVXiwzQ)
- [Make Disadvantage Your Superpower | AMA #6](https://www.youtube.com/watch?v=pXEl0R1BX-g)
- [Who is Hedgy? A Story of Minimalism | AMA #5](https://www.youtube.com/watch?v=KceRmxCnXDA)
- [Impostor Syndrome — Pave Your Own Path | AMA #4](https://www.youtube.com/watch?v=l5Uw8qG7vZU)
- [Dealing with Negative Comments | AMA #3](https://www.youtube.com/watch?v=GFB0o1QQyLw)
- [Sleep and Burnout | AMA #2](https://www.youtube.com/watch?v=wuJa1lnv_TQ)

**🤖 Deep Learning & AI**
- [Deep Learning State of the Art (2020)](https://www.youtube.com/watch?v=0VH1Lim8gL8)
- [Deep Learning Basics: Introduction and Overview](https://www.youtube.com/watch?v=O5xeyoRL95U)
- [Sam Altman Shows Me GPT-5... And What's Next](https://www.youtube.com/watch?v=hmtuvNfytjM)
- [Are Alien Civilizations Powered by Nuclear Fusion?](https://www.youtube.com/watch?v=cGskUxUgzY8)

**🌍 Geopolitics**
- [The Art of Geopolitics, Part 1: Introduction](https://www.youtube.com/watch?v=EUowNpYL120)
- [The Art of Geopolitics, Part 2: Human Dimension](https://www.youtube.com/watch?v=tXgtV_P87ZE)
- [Art of Geopolitics Part 3: The Great Game](https://www.youtube.com/watch?v=w7mTuL1hfzA)
- [Geopolitical Momentum: Is EU-India Deal a Challenge to Great Powers?](https://www.youtube.com/watch?v=Zgx8lcf_qqg)
- [Greenland's Secret SUPERPOWER | Geopolitical Case Study](https://www.youtube.com/watch?v=FdjnOdLDAbg)
- [How Board Games Explain the New World Order](https://www.youtube.com/watch?v=YwcXLZE-EHo)

**🇮🇳 Republic Day 2026**
- [Republic Day 2026: Four-Legged Warriors to Debut at R-Day Parade](https://www.youtube.com/watch?v=lMmPM6TS1e8)
- [Republic Day 2026: Indian Army To Showcase Elite Animal Force](https://www.youtube.com/watch?v=M3xg4v8Kv54)

### 💡 Recommended Questions

**Deep Learning & AI**
- What possibilities does the speaker say deep learning can bring to the world?
- What does the speaker say about learning representations from data?
- What does Sam Altman say about GPT-5 and what comes next?

**Geopolitics**
- What is geopolitics?
- What makes Greenland geopolitically significant?
- How do board games explain the structure of the new world order?
- What is the Great Game and how does it relate to modern geopolitics?

**Mindset & Philosophy**
- How does the speaker suggest dealing with impostor syndrome?
- What advice is given for mastering a difficult technical skill?
- How should one deal with sleep deprivation and burnout?

**Republic Day 2026**
- What animal units are being showcased at the Republic Day 2026 parade?
- How does the Indian Army use animals in its operations?

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | MMR (relevance + diversity) + MultiQuery (LLM query expansion) + BM25 (keyword matching), combined and deduplicated before reranking |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` rescores candidates per sub-question for precision |
| **Compound Query Decomposition** | Multi-topic questions are split into sub-questions, each retrieved and reranked independently |
| **Hallucination-Resistant Prompting** | Two-stage prompt: STEP 1 is a hard relevance gate ("I don't know" and STOP); formatting rules only activate when context is relevant |
| **DuckDuckGo Web Search Fallback** | When RAG returns "I don't know", one click invokes DDG search + page scraping + Groq summarisation |
| **RAGAS Evaluation** | On-demand 4-metric evaluation with weighted confidence scoring, retry-on-low-confidence, and evaluation history dashboard |
| **Dual TTS Backend** | Piper (offline, high quality) for local; gTTS (cloud-safe) for Streamlit Cloud — auto-selected at runtime |
| **SHA-256 Deduplication** | Transcript chunks are hashed; only genuinely new content gets embedded |
| **Committed Vector Store** | `chroma_db/` is committed to git so the deployed app requires no rebuild step |

---

## 📁 Project Structure

`
LexiChat/
│
├── config.py                    ← Central config: paths, model names, thresholds, toggles
├── app.py                       ← Streamlit UI — the application entry point
├── requirements.txt             ← Python dependencies
├── .env.example                 ← Template for required API keys
├── .gitattributes               ← Marks chroma_db files as binary (prevents corruption)
│
├── architecture/                ← All core logic, layered by concern
│   │
│   ├── data/                    ← Source data files
│   │   ├── videos.csv                           ← Video metadata (title, URL, channel)
│   │   ├── video_with_meta_data_and_transcript.csv  ← Per-chunk transcript with timestamps
│   │   └── urls.txt                             ← Raw YouTube URLs for ingestion
│   │
│   ├── ingestion/               ← Data pipeline: metadata → transcript → embed
│   │   ├── youtube_meta_data.py     ← yt-dlp: extract metadata + dump playlist URLs
│   │   ├── transcript_generate.py   ← youtube-transcript-api (yt-dlp fallback for cloud IPs)
│   │   ├── doc_loader.py            ← CSV → LangChain Documents + SHA-256 dedup
│   │   ├── ingestion_pipeline.py    ← Streamlit-facing orchestrator (ingest single video)
│   │   └── vector_store_chroma.py   ← Build / incrementally update ChromaDB collection
│   │
│   ├── retrieval/               ← Hybrid retrieval + query decomposition + answer generation
│   │   └── retriever_pipeline.py    ← MMR + MultiQuery + BM25 + CrossEncoder + Groq LLM
│   │
│   ├── evaluation/              ← RAG quality measurement
│   │   ├── evaluation_pipeline.py   ← RAGAS runner, confidence scoring, CSV logger
│   │   └── metrics_tracker.py       ← Latency and error rate instrumentation
│   │
│   └── web_search/              ← Open-web fallback when RAG cannot answer
│       └── web_search.py            ← DDG search + httpx page scraping + Groq synthesis
│
├── chroma_db/                   ← Persistent ChromaDB vector store (committed to git)
├── logs/                        ← Runtime logs + evaluation CSV (gitignored)
├── assets/                      ← UI screenshots
├── .streamlit/config.toml       ← Streamlit theme (dark mode)
└── .devcontainer/               ← GitHub Codespaces support
`

---

## 🛠️ Tech Stack

| Category | Technology | Role |
|---|---|---|
| Framework | Python 3.10+, Streamlit | App runtime and UI |
| LLM — Answers | Groq `llama-3.1-8b-instant` | Fast, grounded answer generation |
| LLM — MultiQuery | Google `gemini-2.5-flash` | Query expansion for ambiguous questions |
| LLM — RAGAS evaluation | Groq `openai/gpt-oss-120b` | Independent judge for RAG quality |
| LLM — Ground truth | Groq `qwen/qwen3-32b` | Reference answers from a different model family |
| LLM — Query decomposition | Groq `llama-3.1-8b-instant` (cached) | Split compound questions |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) | Free, no API key, runs on CPU |
| Vector Store | ChromaDB (disk-persisted) | Persistent local vector database |
| Retrieval | LangChain MMR, MultiQuery, BM25 | Hybrid retrieval for coverage |
| Reranking | `ms-marco-MiniLM-L-6-v2` (CrossEncoder) | Precision scoring per sub-question |
| Evaluation | RAGAS | Industry-standard RAG evaluation framework |
| Web Search | `ddgs`, `httpx`, `beautifulsoup4` | Fallback when RAG can't answer |
| TTS (local) | Piper + FFmpeg | Offline, high quality |
| TTS (cloud) | gTTS | Cloud-safe, no binaries needed |

### Why These Models?

All models were chosen to be **100% free** — no paid API required:

- **Groq** provides free-tier access to llama, qwen, and openai-compatible models with generous rate limits
- **Google AI Studio** provides free Gemini API keys
- **HuggingFace sentence-transformers** run locally on CPU — zero API cost for embeddings
- **CrossEncoder** runs locally — zero API cost for reranking
- **ChromaDB** is a local vector store — no hosted DB fees

No fine-tuning is used because this is a **general-purpose video RAG system** — users can ingest *any* YouTube video on *any* topic. The system doesn't know in advance what content will be indexed, so fine-tuning a model on specific domain knowledge would actually hurt generalization. Instead, the system relies on **retrieval quality** (hybrid retrieval + reranking) and **prompt engineering** (strict grounding rules) to produce accurate answers from whatever transcripts are in the database.

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10+
- A **Google API key** ([Get one here](https://aistudio.google.com/app/apikey)) — for MultiQuery LLM
- A **Groq API key** ([Get one here](https://console.groq.com/keys)) — for answer generation + evaluation

### Step 1: Clone the Repository

`ash
git clone https://github.com/DakshAhuja10/VideoRAG-QA-System-for-YouTube-Videos.git
cd VideoRAG-QA-System-for-YouTube-Videos
`

> ✅ `chroma_db/` is committed to the repository. The **19 pre-indexed videos are ready to query** — no ingestion needed.

### Step 2: Create Virtual Environment and Install Dependencies

`ash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
`

### Step 3: Configure Environment Variables

`ash
copy .env.example .env         # Windows
# cp .env.example .env         # macOS / Linux
`

Open `.env` and fill in your API keys:

`env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
`

### Step 4: Run the Application

`ash
streamlit run app.py
`

Opens at **http://localhost:8501**. You can immediately ask questions about the 19 indexed videos.

---

## 🔄 Indexing New Videos (Optional)

To add more YouTube videos to the knowledge base:

### 1. Extract video metadata

Add URLs to `architecture/data/urls.txt` (one per line), then:

`ash
python -m architecture.ingestion.youtube_meta_data
`

### 2. Generate transcripts

`ash
python -m architecture.ingestion.transcript_generate
`

Uses `youtube-transcript-api` with automatic `yt-dlp` fallback on cloud IPs.

### 3. Build / update the vector store

`ash
python -m architecture.ingestion.vector_store_chroma
`

Only new chunks get embedded (SHA-256 dedup). Existing chunks are skipped.

### 4. Restart the application

`ash
streamlit run app.py
`

---

## 📊 Evaluation Methodology

| Metric | What It Measures | Weight |
|---|---|---|
| **Faithfulness** | Did the answer stick strictly to retrieved transcript chunks? | 40% |
| **Answer Relevancy** | Did the answer actually address the question asked? | 40% |
| **Context Recall** | Did retrieval surface the information needed to answer? | 10% |
| **Context Precision** | How signal-to-noise was the retrieved context set? | 10% |

**Confidence Score** = weighted sum of all four metrics (0.0 – 1.0).

- Score < 0.4 → retry suggestion shown
- Score < 0.6 → auto-logged to `logs/rag_evaluation_log.csv`

Ground truth is generated by `qwen/qwen3-32b` — a completely different model family from both the answer model and the RAGAS judge — to ensure genuine independence in evaluation.

---

## 🖥️ UI Walkthrough

### 1️⃣ Question Answering
Ask any question. The system streams a grounded answer with clickable timestamps.

<p align="center">
  <img src="assets/UI-1.png" width="900"/>
</p>

### 2️⃣ RAG Evaluation
Click **Evaluate Answer Quality** to trigger RAGAS evaluation with confidence breakdown.

<p align="center">
  <img src="assets/UI-2.png" width="900"/>
</p>

### 3️⃣ Evaluation History
Low-confidence answers are logged. The history tab visualises trends over time.

<p align="center">
  <img src="assets/UI-3.png" width="900"/>
</p>

### 4️⃣ Retry on Low Confidence
When confidence is below threshold, retry triggers wider retrieval. Side-by-side comparison shows original vs improved answer.

<p align="center">
  <img src="assets/UI-4.png" width="900"/>
</p>

---

## 🔊 Text-to-Speech (TTS)

| Environment | TTS Engine | Notes |
|---|---|---|
| Local (paths set) | **Piper** (offline) | High quality, no internet needed |
| Local (paths empty) | **gTTS** (Google) | Cloud-based, requires internet |
| Streamlit Cloud | **gTTS** (Google) | Piper binaries not available |

TTS backend is **auto-selected** — no configuration flags needed.

<details>
<summary><strong>Local Piper Setup (optional)</strong></summary>

1. Download Piper: https://github.com/rhasspy/piper/releases
2. Download a voice model: https://huggingface.co/rhasspy/piper-voices
3. Download FFmpeg: https://www.gyan.dev/ffmpeg/builds/
4. Set in `.env`:

`env
PIPER_EXE=D:\piper\piper.exe
PIPER_VOICE=D:\piper\en_US-lessac-medium.onnx
FFMPEG_EXE=C:\ffmpeg\bin\ffmpeg.exe
`

If any path is missing, the system silently falls back to gTTS.
</details>

---

## 🎯 Design Decisions

| Decision | Rationale |
|---|---|
| **No text splitter** | Transcripts are already line-by-line with timestamps. Splitting further destroys temporal grounding. |
| **SHA-256 deduplication** | Each chunk is hashed before embedding. Re-runs skip duplicates, saving compute. |
| **Per sub-question reranking** | CrossEncoder scores each chunk against its specific sub-question, not the compound query. A chunk about geopolitics scores ~0.8 against "What is geopolitics?" vs ~0.3 against "geopolitics and deep learning". |
| **Two-stage prompt with hard gate** | STEP 1 is a complete halt: "I don't know" and nothing else. Prevents hallucinated timestamps or fabricated content for unknown topics. |
| **chroma_db/ committed to git** | Streamlit Cloud resets filesystem on each deploy. Committing ensures the app works immediately without rebuild. On Cloud, copied to `/tmp/` for full read/write access. |
| **No authentication** | This is a demo project — a recruiter should be able to open the link and immediately interact, without signup friction. |
| **No fine-tuning** | The system ingests arbitrary YouTube videos on any topic. Fine-tuning on specific domain knowledge would hurt generalization. Quality comes from retrieval + prompt engineering instead. |
| **Free-tier only** | All LLMs, embeddings, and infrastructure use free APIs (Groq, Google AI Studio, HuggingFace, Streamlit Cloud). No paid services. |

---

## ⚠️ Known Limitations

- Scope is limited to indexed videos — anything outside returns "I don't know"
- `youtube-transcript-api` is blocked on most cloud server IPs — yt-dlp fallback adds latency
- RAGAS evaluation is slow (~2 min) on Streamlit Cloud shared infrastructure
- Piper TTS requires local binaries — not available on Streamlit Cloud
- `chroma_db/` grows with each video; for 100+ videos, a hosted vector DB (Qdrant, Pinecone) would be more appropriate
- No user authentication (intentional for demo accessibility)

---

## 🔮 Roadmap

- [ ] **FastAPI backend** — decouple API from Streamlit UI (currently learning)
- [ ] **Docker deployment** — reproducible builds for self-hosted setups
- [ ] **Qdrant Cloud migration** — eliminates binary-in-git, scales to thousands of videos
- [ ] **Async / batched RAGAS evaluation** — reduce evaluation latency
- [ ] **Multi-language transcript and TTS support**
- [ ] **User feedback loop** — feed ratings back into retrieval tuning
- [ ] **Improved frontend** — custom React/Next.js UI (planned after learning)
- [ ] **CI/CD pipeline** — GitHub Actions for automated testing on push
- [ ] **Multiple playlist / domain collections** — isolated knowledge bases per topic

---

## 📌 Use Cases

- Educational video search across a course playlist
- Podcast and lecture Q&A with timestamped citations
- Research over long-form interview content
- Knowledge extraction and summarisation from YouTube channels

---
