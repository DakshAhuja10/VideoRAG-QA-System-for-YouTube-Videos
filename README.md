# 🎥 VideoRAG: Question Answering over YouTube Video Transcripts

VideoRAG is an **end-to-end Retrieval-Augmented Generation (RAG) system** that enables **grounded, timestamped question answering over YouTube videos**.
It ingests YouTube playlists, extracts transcripts with metadata, builds a **persistent vector database**, retrieves relevant context using **hybrid retrieval**, generates **strictly grounded answers**, and **automatically evaluates RAG quality** using industry-standard metrics.

The system also includes a **Streamlit frontend** with confidence tracking, retry mechanisms, and evaluation history for continuous improvement.

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

* **Automated RAG Evaluation**

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

### 2️⃣ Automatic RAG Evaluation (RAGAS)
Each answer is automatically evaluated using **RAGAS** on context precision, context recall, faithfulness, and answer relevancy.  
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
RAGAS Evaluation
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
└── README.md
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

---

## 🔮 Future Improvements

* Add multi-language transcript support
* Chunk-level reranking with cross-encoders
* User feedback loop integrated into evaluation
* Dockerized deployment
* Support for multiple playlists / domains

---

## 📌 Use Cases

* Educational video search
* Podcast & lecture Q&A
* Research over long-form video content
* Knowledge extraction from YouTube channels

---


