# 🎥 LexiChat — YouTube Video RAG System

LexiChat is an **end-to-end Retrieval-Augmented Generation (RAG) system** built for **grounded, timestamped question answering over YouTube video transcripts**.

It ingests YouTube playlists, extracts transcripts with timestamps, builds a persistent vector database, and retrieves relevant context using a **hybrid retrieval pipeline** (MMR + MultiQuery + BM25) with **cross-encoder reranking**. Answers are strictly grounded to transcript content and come with **clickable YouTube timestamps**.

When the system cannot answer from its transcript knowledge base, an integrated **DuckDuckGo web-search fallback** lets users fetch an answer from the open web in one click.

---

## 🔗 Live Demo

👉 **https://videorag-app-system-for-youtube-videos-3vpsxvnappai2vxtpjgon3.streamlit.app/**

> ⚠️ Deployed on Streamlit Cloud (shared infrastructure).
> **Answer generation** is fast (~2–3 seconds).
> **RAG evaluation** takes ~2–2.5 minutes due to multiple sequential LLM calls on shared, rate-limited infrastructure.

---

## ⚠️ Content Scope — Read Before Asking Questions

This is a **content-grounded system**, not a general-purpose chatbot. It can only answer questions about the **19 indexed YouTube videos**. Questions outside this scope return:

> **"I don't know. The transcripts do not contain the answer."**

This is **by design** — the system refuses to answer from its own parametric knowledge to prevent hallucination. When this happens, a **DuckDuckGo search** button appears to let you fetch an open-web answer.

### 🎥 Indexed Videos

**🧠 Lex Fridman — Personal Development & Mindset**
- https://www.youtube.com/watch?v=wKw1tpN7NVE — Do Something Difficult Every Day | AMA #1
- https://www.youtube.com/watch?v=_ySbzVXiwzQ — How to Learn and Master a New Skill
- https://www.youtube.com/watch?v=pXEl0R1BX-g — Make Disadvantage Your Superpower | AMA #6
- https://www.youtube.com/watch?v=KceRmxCnXDA — Who is Hedgy? A Story of Minimalism | AMA #5
- https://www.youtube.com/watch?v=l5Uw8qG7vZU — Impostor Syndrome — Pave Your Own Path | AMA #4
- https://www.youtube.com/watch?v=GFB0o1QQyLw — Dealing with Negative Comments | AMA #3
- https://www.youtube.com/watch?v=wuJa1lnv_TQ — Sleep and Burnout | AMA #2

**🤖 Deep Learning & AI**
- https://www.youtube.com/watch?v=0VH1Lim8gL8 — Deep Learning State of the Art (2020)
- https://www.youtube.com/watch?v=O5xeyoRL95U — Deep Learning Basics: Introduction and Overview
- https://www.youtube.com/watch?v=hmtuvNfytjM — Sam Altman Shows Me GPT-5... And What's Next
- https://www.youtube.com/watch?v=cGskUxUgzY8 — Are Alien Civilizations Powered by Nuclear Fusion?

**🌍 Geopolitics**
- https://www.youtube.com/watch?v=EUowNpYL120 — The Art of Geopolitics, Part 1: Introduction
- https://www.youtube.com/watch?v=tXgtV_P87ZE — The Art of Geopolitics, Part 2: Human Dimension
- https://www.youtube.com/watch?v=w7mTuL1hfzA — Art of Geopolitics Part 3: The Great Game
- https://www.youtube.com/watch?v=Zgx8lcf_qqg — Geopolitical Momentum: Is EU-India Deal a Challenge to Great Powers?
- https://www.youtube.com/watch?v=FdjnOdLDAbg — Greenland's Secret SUPERPOWER | Geopolitical Case Study
- https://www.youtube.com/watch?v=YwcXLZE-EHo — How Board Games Explain the New World Order

**🇮🇳 Republic Day 2026**
- https://www.youtube.com/watch?v=lMmPM6TS1e8 — Republic Day 2026: Four-Legged Warriors to Debut at R-Day Parade
- https://www.youtube.com/watch?v=M3xg4v8Kv54 — Republic Day 2026: Indian Army To Showcase Elite Animal Force

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
- What is said about discipline, consistency, and long-term thinking?

**Republic Day 2026**
- What animal units are being showcased at the Republic Day 2026 parade?
- How does the Indian Army use animals in its operations?

---

## 🚀 Key Features

- **Hybrid Retrieval** — MMR (relevance + diversity) + MultiQuery (LLM query expansion) + BM25 (keyword matching), all combined and deduplicated before reranking
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` rescores candidates per sub-question for maximum precision
- **Compound Query Decomposition** — Multi-topic questions (e.g., "What is geopolitics and deep learning?") are split into sub-questions, each retrieved and reranked independently against its own sub-question
- **Hallucination-Resistant Prompt Design** — Two-stage prompt: STEP 1 is a hard relevance gate ("output I don't know and STOP"); answer formatting rules are only reached when context is relevant
- **DuckDuckGo Web Search Fallback** — When RAG returns "I don't know", a single click invokes DDG search + page scraping + Groq summarisation
- **RAGAS Evaluation** — On-demand 4-metric evaluation (faithfulness, answer relevancy, context precision, context recall) with weighted confidence scoring, retry-on-low-confidence, and evaluation history dashboard
- **Dual TTS Backend** — Piper (offline, high quality) for local use; gTTS (cloud-safe) for Streamlit Cloud — auto-selected at runtime
- **SHA-256 Deduplication** — Transcript chunks are hashed; only genuinely new content gets embedded, saving embedding API cost
- **Committed Vector Store** — `chroma_db/` is committed to git so the deployed app requires no rebuild step

---

## 📁 Project Structure

```
LexiChat/
│
├── config.py                   ← Central config: all paths, model names, thresholds, toggles
├── frontend1.py                ← Streamlit UI — the application entry point
├── requirements.txt            ← Python dependencies
├── .env.example                ← Template for required API keys and local paths
│
├── architecture/               ← All core logic, layered by concern
│   │
│   ├── data/                   ← Source data files
│   │   ├── videos.csv                          ← Video metadata (title, URL, channel)
│   │   ├── video_with_meta_data_and_transcript.csv  ← Per-line transcript with timestamps
│   │   └── urls.txt                            ← Raw YouTube URLs for ingestion
│   │
│   ├── ingestion/              ← Data pipeline: metadata → transcript → embed
│   │   ├── youtube_meta_data.py    ← yt-dlp: extract metadata + dump playlist URLs
│   │   ├── transcript_generate.py  ← youtube-transcript-api (yt-dlp fallback for cloud IPs)
│   │   ├── doc_loader.py           ← CSV → LangChain Documents + SHA-256 dedup
│   │   ├── ingestion_pipeline.py   ← Streamlit-facing orchestrator (ingest single video)
│   │   └── vector_store_chroma.py  ← Build / incrementally update ChromaDB collection
│   │
│   ├── retrieval/              ← Hybrid retrieval + query decomposition + answer generation
│   │   └── retriever_pipeline2.py  ← MMR + MultiQuery + BM25 + CrossEncoder + Groq LLM
│   │
│   ├── evaluation/             ← RAG quality measurement
│   │   ├── evaluation_pipeline.py  ← RAGAS runner, confidence scoring, CSV logger
│   │   └── metrics_tracker.py      ← Latency and error rate instrumentation
│   │
│   └── web_search/             ← Open-web fallback when RAG cannot answer
│       └── web_search.py           ← DDG search + httpx page scraping + Groq synthesis
│
├── chroma_db/                  ← Persistent ChromaDB vector store (committed to git)
├── logs/                       ← Runtime logs + evaluation CSV (gitignored)
├── assets/                     ← UI screenshots used in this README
│
└── ignore/                     ← Non-essential / experimental files (gitignored)
    ├── retriever_pipeline1.py      ← Earlier retrieval prototype
    ├── vector_db_check.py          ← Debug utility: inspect ChromaDB collections
    ├── test.ipynb                  ← Scratch notebook
    ├── interview_questions_and_answers.txt
    ├── piper_voices_output/        ← Local TTS audio samples
    ├── METRICS_TRACKING.md
    ├── CODE_QUALITY_IMPROVEMENTS.md
    └── BUGFIX_EVALUATION_HISTORY.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or higher
- A **Google API key** (Gemini — used for MultiQuery LLM and previously for embeddings)
- A **Groq API key** (used for answer generation, evaluation, and ground truth LLMs)

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/LexiChat.git
cd LexiChat
```

> ✅ `chroma_db/` is committed to the repository. The **19 pre-indexed videos are ready to query** — you do not need to run the ingestion pipeline before starting.

---

### Step 2: Create a Virtual Environment and Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 3: Configure Environment Variables

```bash
# Copy the template
cp .env.example .env      # macOS / Linux
copy .env.example .env    # Windows
```

Open `.env` and fill in your API keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Optional — only needed for local Piper TTS (see TTS section)
# Leave empty to auto-use gTTS instead
PIPER_EXE=
PIPER_VOICE=
FFMPEG_EXE=
```

---

### Step 4: Run the Application

```bash
streamlit run frontend1.py
```

The app opens at **http://localhost:8501**. That's it — the vector store is pre-built. You can immediately ask questions about any of the 19 indexed videos.

---

## 🔄 Indexing New Videos (Optional)

To add more YouTube videos to the knowledge base, run the ingestion pipeline step by step from the project root:

### 1. Extract video metadata

Add URLs to `architecture/data/urls.txt` (one per line) or run the playlist extractor:

```bash
python -m architecture.ingestion.youtube_meta_data
```

This populates `architecture/data/videos.csv` with title, channel, duration, and URL.

### 2. Generate transcripts

```bash
python -m architecture.ingestion.transcript_generate
```

Fetches transcripts line-by-line with timestamps using `youtube-transcript-api`. On cloud servers (where YouTube API calls are blocked), it automatically falls back to `yt-dlp` subtitle extraction.

Transcripts are saved to `architecture/data/video_with_meta_data_and_transcript.csv`.

### 3. Rebuild / update the vector store

```bash
python -m architecture.ingestion.vector_store_chroma
```

Reads the transcript CSV, skips chunks that already exist in ChromaDB (via SHA-256 hash), embeds only new content, and persists to `chroma_db/`.

### 4. Restart the application

```bash
streamlit run frontend1.py
```

New videos are immediately queryable.

---

## 🔊 Text-to-Speech (TTS)

LexiChat can read answers aloud. The TTS backend is **automatically selected** at runtime — no flags needed.

| Environment | TTS Engine | Notes |
|---|---|---|
| Local machine (paths set) | **Piper** (offline) | High quality, natural voice, no internet needed |
| Local machine (paths empty) | **gTTS** (Google) | Cloud-based, lower quality, requires internet |
| Streamlit Cloud | **gTTS** (Google) | Piper binaries not available on cloud |

The active TTS mode is shown in the UI:
```
🔊 TTS Mode: Piper (Local)
```

### Local Piper Setup

1. Download Piper: **https://github.com/rhasspy/piper/releases**
2. Extract to a local directory (e.g., `D:\piper\`)
3. Download a voice model (e.g., `en_US-lessac-medium.onnx`): **https://huggingface.co/rhasspy/piper-voices**
4. Download FFmpeg: **https://www.gyan.dev/ffmpeg/builds/** — add to system PATH
5. Set the three variables in your `.env` file:

```env
PIPER_EXE=D:\piper\piper.exe
PIPER_VOICE=D:\piper\en_US-lessac-medium.onnx
FFMPEG_EXE=C:\ffmpeg\bin\ffmpeg.exe
```

If any of these paths are missing or the files don't exist, the system silently falls back to gTTS.

---

## 🖥️ UI Walkthrough

### 1️⃣ Question Answering
Ask any question. The system streams a grounded answer with clickable timestamps.

<p align="center">
  <img src="assets/UI-1.png" width="900"/>
</p>

### 2️⃣ RAG Evaluation
Click **Evaluate Answer Quality** to trigger RAGAS evaluation. A confidence score breakdown is shown.

<p align="center">
  <img src="assets/UI-2.png" width="900"/>
</p>

### 3️⃣ Evaluation History
Low-confidence answers are automatically logged. The history tab visualises trends over time.

<p align="center">
  <img src="assets/UI-3.png" width="900"/>
</p>

### 4️⃣ Retry on Low Confidence
When confidence is below 0.5, a retry button triggers a wider retrieval pass. Side-by-side comparison shows the original vs improved answer and updated metrics.

<p align="center">
  <img src="assets/UI-4.png" width="900"/>
</p>

---

## 🧠 System Architecture

```
YouTube URL / Playlist
        ↓
  youtube_meta_data.py        yt-dlp: extract titles, channels, durations → videos.csv
        ↓
  transcript_generate.py      youtube-transcript-api (yt-dlp fallback on cloud IPs)
                              Timestamped lines → video_with_meta_data_and_transcript.csv
        ↓
  doc_loader.py               CSV → LangChain Documents, SHA-256 hash per chunk
        ↓
  vector_store_chroma.py      sentence-transformers/all-MiniLM-L6-v2 embeddings
                              Dedup via hash → persist to chroma_db/
        ↓
  retriever_pipeline2.py      Per-sub-question hybrid retrieval:
  ├── _decompose_query()        compound questions → list of sub-questions (Groq LLM)
  ├── MMR Retriever             k=6, fetch_k=20 — relevance + diversity
  ├── MultiQuery Retriever      gemini-2.5-flash query expansion — handles ambiguity
  ├── BM25 Retriever            k=6 keyword matching
  └── CrossEncoder Reranker     ms-marco-MiniLM-L-6-v2, scored per sub-question
        ↓
  rank_prompt (Groq llama-3.1-8b-instant)
  STEP 1: Relevance gate — "I don't know" and STOP if no relevant context
  STEP 2: Format rules — 3–4 sentences per answer, ONE clickable timestamp each
        ↓
  ┌──────────────────────────────────────────┐
  │  Answer known?                           │
  │  YES → Streamed answer + TTS audio       │
  │        (Piper local / gTTS cloud)        │
  │  NO  → DuckDuckGo search button          │
  │        web_search.py:                    │
  │        DDG results → httpx page scrape   │
  │        → Groq synthesis → answer         │
  └──────────────────────────────────────────┘
        ↓  (optional)
  evaluation_pipeline.py      RAGAS: context precision/recall, faithfulness,
                              answer relevancy → weighted confidence score
                              < 0.6 → append to logs/rag_evaluation_log.csv
        ↓
  frontend1.py                Streamlit: streamed answers, evaluation panel,
                              retry flow, evaluation history dashboard
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language & Framework | Python 3.10+, Streamlit |
| LLM — Answers | Groq `llama-3.1-8b-instant` |
| LLM — MultiQuery expansion | Google `gemini-2.5-flash` |
| LLM — RAGAS evaluation judge | Groq `openai/gpt-oss-120b` |
| LLM — Ground truth generation | Groq `qwen/qwen3-32b` |
| LLM — Compound query decomposition | Groq `llama-3.1-8b-instant` (cached) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local) |
| Vector Store | ChromaDB (disk-persisted) |
| Retrieval | LangChain MMR, MultiQuery, BM25 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` (SentenceTransformers) |
| Evaluation | RAGAS |
| Web Search | `ddgs` (DuckDuckGo), `httpx`, `beautifulsoup4` |
| TTS (local) | Piper + FFmpeg |
| TTS (cloud) | gTTS |
| Data Handling | Pandas, CSV |

---

## 📊 Evaluation Methodology

| Metric | What it measures | Weight |
|---|---|---|
| **Faithfulness** | Did the answer stick strictly to retrieved transcript chunks? | 40% |
| **Answer Relevancy** | Did the answer actually address the question asked? | 40% |
| **Context Recall** | Did retrieval surface the information needed to answer? | 10% |
| **Context Precision** | How signal-to-noise was the retrieved context set? | 10% |

**Confidence Score** = weighted sum of all four metrics (0.0 – 1.0).

- Score < 0.4 → retry suggestion shown  
- Score < 0.6 → answer auto-logged to `logs/rag_evaluation_log.csv` for monitoring

Ground truth is generated by `qwen/qwen3-32b` (a completely different model family from both the answer model and RAGAS judge) to ensure genuine independence in evaluation.

---

## 🎯 Design Decisions

**No text splitter** — Transcripts are already segmented line-by-line with precise timestamps. Splitting further would destroy temporal grounding.

**SHA-256 deduplication** — Each transcript chunk is hashed before embedding. Identical chunks from re-runs are skipped, saving embedding cost.

**Per-sub-question reranking** — The CrossEncoder scores each candidate chunk against its specific sub-question, not the compound query. A chunk about geopolitics gets scored against "What is geopolitics?" (score ~0.8), not "geopolitics and deep learning" (score ~0.3).

**Two-stage prompt with hard gate** — The STEP 1 relevance check is a complete halt: the LLM outputs "I don't know" and nothing else. This prevents the model from hallucinating timestamps or fabricating content for unknown topics, which it would do if format rules were in scope before the gate check.

**`chroma_db/` committed to git** — Streamlit Cloud uses an ephemeral filesystem that resets on each deploy. Committing the vector store ensures the deployed app works immediately without requiring a rebuild step.

**`ignore/` folder in `.gitignore`** — Experimental notebooks, old prototypes, internal dev docs, and debug utilities are collected here rather than deleted. They remain accessible locally but are excluded from git history to keep the repository clean.

---

## ⚠️ Known Limitations

- Scope is limited to 19 indexed videos — anything outside returns "I don't know"  
- `youtube-transcript-api` is blocked on most cloud server IPs — yt-dlp fallback is used during ingestion but adds latency  
- RAGAS evaluation is slow (~2 min) on Streamlit Cloud shared infrastructure  
- Piper TTS requires local binaries — not available on Streamlit Cloud (gTTS used automatically)  
- `chroma_db/` grows in size with each new video; for large-scale use (100+ videos), migrating to Qdrant Cloud or Pinecone is recommended  

---

## 🔮 Future Improvements

- Qdrant Cloud migration (eliminates binary-in-git, scales to thousands of videos)
- Async / batched RAGAS evaluation to reduce latency
- Multi-language transcript and TTS support
- User feedback loop feeding back into retrieval tuning
- Support for multiple playlists / custom domain collections
- Docker deployment for self-hosted production use

---

## 📌 Use Cases

- Educational video search across a course playlist
- Podcast and lecture Q&A
- Research over long-form interview content
- Knowledge extraction and summarisation from YouTube channels

VideoRAG is an **end-to-end Retrieval-Augmented Generation (RAG) system** that enables **grounded, timestamped question answering over YouTube videos**.
It ingests YouTube playlists, extracts transcripts with metadata, builds a **persistent vector database**, retrieves relevant context using **hybrid retrieval**, generates **strictly grounded answers**, and **automatically evaluates RAG quality** using industry-standard metrics.

The system also includes a **Streamlit frontend** with confidence tracking, retry mechanisms, and evaluation history for continuous improvement.

## 🔗 Live Demo (Streamlit App)

You can try the deployed **VideoRAG** application here:

👉 **https://videorag-app-system-for-youtube-videos-3vpsxvnappai2vxtpjgon3.streamlit.app/**

> ⚠️ **Note:** This is a demo deployment on **Streamlit Cloud**.  
> Answer evaluation can take longer (≈2–2.5 minutes) because it triggers **multiple sequential LLM calls** for RAG quality metrics and runs on **shared, rate-limited cloud infrastructure**.  
> See the *Evaluation Latency* section below for details.

---

## ⚠️ Important Usage Note (Read Before Asking Questions)

This application is a **content-grounded VideoRAG system**, not a general-purpose chatbot.

### 📌 What content is indexed?
Embeddings have been created **only for transcripts from a limited set of YouTube videos (currently 19 videos)**.  
The system can answer questions **only if the information exists explicitly in these transcripts**.

If you ask a question **outside the scope of this content**, the system will intentionally respond with:

> **"I don't know. The transcripts do not contain the answer."**

This behavior is **by design** to prevent hallucinations and ensure faithful, grounded answers.

---

### 🎥 Indexed YouTube Videos

Please ask questions **only related to the following videos** to get relevant and high-confidence answers:

**🧠 Lex Fridman — Personal Development & Mindset**
- https://www.youtube.com/watch?v=wKw1tpN7NVE — Do Something Difficult Every Day | AMA #1
- https://www.youtube.com/watch?v=_ySbzVXiwzQ — How to Learn and Master a New Skill
- https://www.youtube.com/watch?v=pXEl0R1BX-g — Make Disadvantage Your Superpower | AMA #6
- https://www.youtube.com/watch?v=KceRmxCnXDA — Who is Hedgy? A Story of Minimalism | AMA #5
- https://www.youtube.com/watch?v=l5Uw8qG7vZU — Impostor Syndrome — Pave Your Own Path | AMA #4
- https://www.youtube.com/watch?v=GFB0o1QQyLw — Dealing with Negative Comments | AMA #3
- https://www.youtube.com/watch?v=wuJa1lnv_TQ — Sleep and Burnout | AMA #2

**🤖 Deep Learning & AI**
- https://www.youtube.com/watch?v=0VH1Lim8gL8 — Deep Learning State of the Art (2020)
- https://www.youtube.com/watch?v=O5xeyoRL95U — Deep Learning Basics: Introduction and Overview
- https://www.youtube.com/watch?v=hmtuvNfytjM — Sam Altman Shows Me GPT-5... And What's Next
- https://www.youtube.com/watch?v=cGskUxUgzY8 — Are Alien Civilizations Powered by Nuclear Fusion? | David Kirtley & Lex Fridman

**🌍 Geopolitics**
- https://www.youtube.com/watch?v=EUowNpYL120 — The Art of Geopolitics, Part 1: Introduction
- https://www.youtube.com/watch?v=tXgtV_P87ZE — The Art of Geopolitics, Part 2: Human Dimension
- https://www.youtube.com/watch?v=w7mTuL1hfzA — Art of Geopolitics Part 3: The Great Game
- https://www.youtube.com/watch?v=Zgx8lcf_qqg — Geopolitical Momentum: Is EU-India Deal a Challenge to Great Powers?
- https://www.youtube.com/watch?v=FdjnOdLDAbg — Greenland's Secret SUPERPOWER | Geopolitical Case Study
- https://www.youtube.com/watch?v=YwcXLZE-EHo — How Board Games Explain the New World Order

**🇮🇳 Republic Day 2026**
- https://www.youtube.com/watch?v=lMmPM6TS1e8 — Republic Day 2026: Four-Legged Warriors to Debut at R-Day Parade
- https://www.youtube.com/watch?v=M3xg4v8Kv54 — Republic Day 2026: Indian Army To Showcase Elite Animal Force

---

### 💡 Recommended Questions to Try

To see VideoRAG perform at its best, try asking **questions that are explicitly discussed in the indexed videos** listed above.

**🎓 Deep Learning & AI Concepts**
- What possibilities does the speaker say deep learning can bring to the world?
- What does the speaker say about learning representations from data?
- How are neural networks explained in the context of real-world problems?
- What does Sam Altman say about GPT-5 and what comes next after it?
- What is the speaker's view on nuclear fusion as an energy source for civilization?

**🌍 Geopolitics**
- What is geopolitics?
- How is the EU-India deal described in the context of great power competition?
- What makes Greenland geopolitically significant?
- How do board games explain the structure of the new world order?
- What is the Great Game and how does it relate to modern geopolitics?

**🇮🇳 Republic Day 2026**
- What animal units are being showcased at the Republic Day 2026 parade?
- How does the Indian Army use animals in its operations?

**🧠 Philosophy & Mindset**
- What is said about discipline and consistency in learning or research?
- How does the speaker approach long-term thinking and progress?
- What mindset is recommended for mastering difficult technical subjects?
- How does the speaker suggest dealing with impostor syndrome?

**🛠️ Practical Advice**
- What advice is given for beginners entering machine learning or AI?
- How does the speaker suggest dealing with failure or confusion while learning?
- What role does practice play according to the speaker?
- How should one deal with sleep deprivation and burnout?

---

### ⚠️ Reminder
If you ask questions **outside the scope of the indexed videos**, the system will intentionally respond with:

> **"I don't know. The transcripts do not contain the answer."**

This indicates that the system is **avoiding hallucination by answering only from verified transcript context**.

---

## 🚀 Key Features

* **YouTube Playlist Ingestion**
  * Extracts video URLs and metadata using `yt-dlp`
  * Deduplicates videos across runs

* **Transcript Extraction**
  * Fetches English transcripts line-by-line
  * Preserves timestamps for precise citations
  * Incremental processing (no rework on already processed videos)

* **Cost-Efficient Embedding Pipeline**
  * Uses SHA-256 hashing to detect duplicate text
  * Reuses embeddings when transcript text is identical
  * Avoids unnecessary calls to embedding APIs

* **Persistent Vector Store**
  * ChromaDB for disk-backed persistence
  * Designed for long-term usage and incremental updates

* **Hybrid Retrieval Strategy**
  * **MMR Retriever** → relevance + diversity
  * **Multi-Query Retriever** → handles ambiguous queries
  * **BM25 Retriever** → keyword-based lexical search
  * Automatic deduplication across retrievers

* **Strictly Grounded Answer Generation**
  * Answers ONLY from retrieved transcript chunks
  * Clickable YouTube timestamps for every answer
  * Hallucination-resistant prompt design

* **RAG Evaluation**
  * Evaluated on:
    * Context Precision
    * Context Recall
    * Faithfulness
    * Answer Relevancy
  * Confidence score computed with weighted metrics
  * Low-confidence answers automatically logged

* **Interactive Frontend**
  * Streamlit UI with streamed answers
  * Confidence visualization (high / medium / low)
  * Retry on low confidence with re-evaluation
  * Evaluation history dashboard

---

## 🖥️ Streamlit Interface Walkthrough

Below is the end-to-end Streamlit interface for **VideoRAG**, demonstrating question answering, automatic RAG evaluation, and evaluation history tracking.

### 1️⃣ Question Answering over YouTube Transcripts
Users ask questions over video transcripts.  
The system retrieves relevant context, generates grounded answers, and provides clickable timestamps.

<p align="center">
  <img src="assets/UI-1.png" width="900"/>
</p>

---

### 2️⃣ RAG Evaluation (RAGAS)
Each answer is evaluated using **RAGAS** on context precision, context recall, faithfulness, and answer relevancy.  
A confidence score is computed to assess answer reliability.

<p align="center">
  <img src="assets/UI-2.png" width="900"/>
</p>

---

### 3️⃣ Evaluation History & Monitoring
Low-confidence answers are logged and visualized over time, enabling continuous monitoring and debugging of the RAG system.

<p align="center">
  <img src="assets/UI-3.png" width="900"/>
</p>

---

### 4️⃣ Retry, Confidence Improvement & Answer Comparison

When an answer receives a low confidence score, users can **retry the same question**.  
Retry triggers a fresh retrieval pass, often surfacing more explicit or better-grounded context.

The interface highlights:
- **Confidence improvement** after retry
- **Side-by-side comparison** of the original vs retried answer
- **Updated RAGAS evaluation metrics** reflecting improved grounding

This makes retrieval instability transparent and demonstrates how improved context directly impacts answer faithfulness and confidence.

<p align="center">
  <img src="assets/UI-4.png" width="900"/>
</p>

---

## 🧠 System Architecture
```
YouTube Playlist
      ↓
Metadata Extraction (yt-dlp)
      ↓
Transcript Extraction (youtube-transcript-api)
      ↓
CSV → LangChain Documents
      ↓
Embedding + Deduplication (Gemini)
      ↓
ChromaDB (Persistent Vector Store)
      ↓
Hybrid Retrieval (MMR + MultiQuery + BM25)
      ↓
LLM Answer Generation (Groq)
      ↓
RAGAS Evaluation/Generate an audio
      ↓
Streamlit Frontend + Logs
```

---

## 🛠️ Tech Stack

**Languages & Frameworks**
* Python
* Streamlit

**LLMs & Embeddings**
* Google Gemini (`text-embedding-004`)
* Gemini 2.5 Flash (multi-query generation)
* Groq (`openai/gpt-oss-120B`) for answer generation & evaluation

**RAG & Retrieval**
* LangChain
* ChromaDB
* BM25 Retriever
* MultiQuery Retriever
* MMR Retriever

**Text-to-Speech**
* Piper (offline, local)
* gTTS (cloud deployment)

**Evaluation**
* RAGAS

**Data Handling**
* Pandas
* CSV-based incremental pipelines

---

## 📁 Project Structure
```
.
├── youtube_meta_data.py
├── transcript_generate.py
├── doc_loader.py
├── vector_store_chroma.py
├── retriever_pipeline2.py
├── evaluation_pipeline.py
├── frontend1.py
├── logs/
│   ├── app.log
│   └── rag_evaluation_log.csv
├── chroma_db/
├── README.md
└── assets/
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/VideoRAG.git
cd VideoRAG
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Environment Variables

Create a `.env` file:
```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## 🔊 Text-to-Speech (TTS) Support

VideoRAG includes **spoken answer generation** using a **dual Text-to-Speech (TTS) backend** that automatically adapts to the execution environment.

The system selects the appropriate TTS engine **without any manual configuration**, ensuring smooth local development and cloud deployment.

---

### 🔁 Automatic TTS Backend Selection

| Environment | TTS Engine | Why |
|------------|-----------|-----|
| **Local machine** | **Piper (Offline TTS)** | Fast, high-quality, fully offline,No API required |
| **Streamlit Cloud** | **gTTS (Google Text-to-Speech)** | Cloud-safe, no native binaries required |

The UI clearly displays the active mode:
```
🔊 TTS Mode: Piper (Local)
```
or
```
🔊 TTS Mode: gTTS (Cloud)
```

---

## 🖥️ Local Setup: Piper (Offline Text-to-Speech)

Piper is used **only in local environments**, as Streamlit Cloud does not allow native binaries.

### 1️⃣ Download Piper

Download the latest Piper release from the official repository:

**https://github.com/rhasspy/piper/releases**

Extract the files to the following directory (Windows):
```
D:\piper\
```

Your folder should contain:
```
D:\piper
├── piper.exe
├── en_US-lessac-medium.onnx
```

---

### 2️⃣ Verify Piper Installation

Run:
```bash
D:\piper\piper.exe --help
```

If the help menu appears, Piper is installed correctly.

---

### 3️⃣ Install FFmpeg (Required)

Piper outputs WAV audio, which is converted to MP3 using FFmpeg.

Download FFmpeg (Windows static build):

**https://www.gyan.dev/ffmpeg/builds/**

Add FFmpeg to your system PATH, then verify:
```bash
ffmpeg -version
```


## ☁️ Streamlit Cloud Setup: gTTS (Google Text-to-Speech)

Streamlit Cloud does not support native executables like Piper.  
To ensure compatibility, VideoRAG automatically switches to gTTS when deployed.

### Cloud TTS Behavior

On Streamlit Cloud:

-   Piper is unavailable
-  gTTS is used automatically
-  Audio is generated directly as MP3
-  No additional setup or environment variables are required

## ⚠️ Known Limitations

- gTTS requires an active internet connection
- Piper is currently configured for English voices only

## ▶️ Running the Pipeline

### Step 1: Extract Video Metadata
```bash
python youtube_meta_data.py
```

### Step 2: Generate Transcripts
```bash
python transcript_generate.py
```

### Step 3: Build / Update Vector Store
```bash
python vector_store_chroma.py
```

### Step 4: Launch Frontend
```bash
streamlit run frontend1.py
```

---

## 📊 Evaluation Methodology

VideoRAG evaluates each answer using **RAGAS** on:

| Metric            | Description                                |
| ----------------- | ------------------------------------------ |
| Context Precision | Measures noise in retrieved context        |
| Context Recall    | Measures coverage of required information  |
| Faithfulness      | Checks grounding to retrieved context      |
| Answer Relevancy  | Measures how well the question is answered |

A **custom confidence score** is computed with higher weight on faithfulness to discourage hallucinations.

Low-confidence answers are automatically logged for inspection and improvement.

---

### ⏱️ Evaluation Latency on Streamlit Cloud

Because this app is deployed on **Streamlit Cloud (shared infrastructure)**, evaluation may take:

> ⏳ **~2–2.5 minutes per evaluation**

This latency is expected and caused by:
- Multiple sequential LLM calls per metric
- Rate limits on hosted LLM providers
- Cold starts on shared cloud resources

⚠️ **Important:**  
This does **not** affect answer generation speed.  
Only the **evaluation phase** is slow.

In a production or self-hosted environment, evaluation latency can be reduced significantly by:
- Using smaller evaluation models
- Running evaluation asynchronously
- Moving evaluation to offline or batch pipelines

---

## 🎯 Design Decisions

* **No Text Splitter**  
  Transcripts are already timestamped line-by-line. Further splitting would break temporal grounding.

* **Hash-Based Deduplication**  
  Prevents repeated embedding generation → saves cost and time.

* **Hybrid Retrieval**  
  Combines semantic, lexical, and query-expansion techniques for robust recall.

* **Strict Prompt Constraints**  
  Forces grounded, timestamped, multi-sentence answers only from context.

* **Evaluation-Driven Development**  
  RAG quality is continuously measured, logged, and improved.

* **Capability-Based TTS Selection**  
  Automatically switches between Piper (local) and gTTS (cloud) without manual configuration.

---

## 🔮 Future Improvements

* Add multi-language transcript support
* Chunk-level reranking with cross-encoders
* User feedback loop integrated into evaluation
* Dockerized deployment
* Support for multiple playlists / domains
* Multi-language TTS support

---

## 📌 Use Cases

* Educational video search
* Podcast & lecture Q&A
* Research over long-form video content
* Knowledge extraction from YouTube channels

---