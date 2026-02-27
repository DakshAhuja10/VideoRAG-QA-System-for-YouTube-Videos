import streamlit as st
import logging
import os
import pandas as pd
import time
import re
import threading
import subprocess
from pathlib import Path


from retriever_pipeline2 import ask, ask_stream, ask_stream_broad
from evaluation_pipeline import evaluate_answer
from config import (
    PIPER_EXE,
    PIPER_VOICE,
    FFMPEG_EXE,
    RAG_EVALUATION_LOG_FILE,
    EvaluationConfig,
    UIConfig,
    TTS_AUDIO_BITRATE,
    TTS_TEMP_WAV,
    TTS_OUTPUT_FILE,
)
# from metrics_tracker import (
#     track_latency,
#     track_query,
#     track_error,
#     get_session_summary,
#     save_metrics_summary,
#     get_metrics_tracker,
# )

# Dummy functions to reduce load time/disable latency tracking
def track_latency(*args, **kwargs):
    class DummyContext:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    return DummyContext()

def track_query(*args, **kwargs): pass
def track_error(*args, **kwargs): pass
def save_metrics_summary(*args, **kwargs): pass
def get_session_summary(): 
    return {"total_queries":0, "total_errors":0,"error_rate":0,"retry_rate":0,"latency_stats":{},"api_usage":{"calls":{}, "tokens":{}}}

class DummyTracker:
    def track_audio_generation(self, *args, **kwargs): pass
    @property
    def metrics_file(self):
        from pathlib import Path
        return Path("dummy.csv")

def get_metrics_tracker(): return DummyTracker()


LOG_FILE = str(RAG_EVALUATION_LOG_FILE)
CONFIDENCE_THRESHOLD = EvaluationConfig.CONFIDENCE_THRESHOLD
os.makedirs("logs", exist_ok=True)

root_logger = logging.getLogger()
if not root_logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

#ignore these warning and not write in logs
for noisy in ["httpx", "groq", "ragas"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title=UIConfig.PAGE_TITLE,
    page_icon=UIConfig.PAGE_ICON,
    layout=UIConfig.LAYOUT
)


IS_STREAMLIT_CLOUD = (
    os.environ.get("SF_PARTNER") == "streamlit"
    or os.environ.get("HOSTNAME") == "streamlit"
)

if IS_STREAMLIT_CLOUD:
    logger.info("Running on Streamlit Cloud - using gTTS")
    USE_GTTS = True
else:
    logger.info("Running locally - using Piper")
    USE_GTTS = False
    # Paths are now loaded from config.py


def generate_voice_mp3(text: str, out_mp3: str):
    """
    Generate MP3 voice from text Uses gTTS on Streamlit Cloud, Piper locally
    """
    if USE_GTTS:
        # Cloud: Use gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(out_mp3)
            logger.info("Audio generated using gTTS")
        except ImportError:
            logger.error("gTTS not installed. Install with: pip install gtts")
            raise
    else:
        # Local: Use Piper + FFmpeg
        wav_path = Path(TTS_TEMP_WAV)

        # Run Piper (stdin → wav)
        subprocess.run(
            [PIPER_EXE, "-m", PIPER_VOICE, "-f", str(wav_path)],
            input=text.encode("utf-8"),
            check=True
        )

        # Convert WAV → MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-ab", TTS_AUDIO_BITRATE, out_mp3],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if wav_path.exists():
            wav_path.unlink()
        
        logger.info("Audio generated using Piper")


def split_answers(answer_text: str):
    pattern = r"\n(?=\d+\.\s)"
    return [a.strip() for a in re.split(pattern, answer_text.strip()) if a.strip()]


#default session state is initialized used to see evaluation history
default_state = {
    "question": "",
    "rag_answer": None,
    "contexts": None,
    "eval_out": None,
    "stream_done": False,
    "answer_placeholder_content": "",
    "original_answer": None,
    "retried_answer": None,
    "original_confidence": None,
    "evaluation_requested": False,
    "evaluation_in_progress": False,
    
    
    #audio state
    "audio_file": None,
    "audio_generating": False,
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

def run_evaluation_async(result_container, question, rag_answer, contexts):
    try:
        result_container["output"] = evaluate_answer(
            question=question,
            rag_answer=rag_answer,
            retrieved_contexts=contexts
        )
    except Exception as e:
        result_container["error"] = str(e)



def run_audio_async(result_container, text, out_path):
    try:
        if USE_GTTS:
            # Cloud: Use gTTS
            from gtts import gTTS
            logger.info("Starting gTTS audio generation...")
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(out_path)
            logger.info(f"gTTS audio saved to {out_path}")
        else:
            # Local: Use Piper + FFmpeg
            logger.info("Starting Piper audio generation...")
            wav_path = Path(TTS_TEMP_WAV)

            # Check if Piper exists
            if not os.path.exists(PIPER_EXE):
                raise FileNotFoundError(f"Piper executable not found at: {PIPER_EXE}")
            
            if not os.path.exists(PIPER_VOICE):
                raise FileNotFoundError(f"Piper voice model not found at: {PIPER_VOICE}")

            subprocess.run(
                [PIPER_EXE, "-m", PIPER_VOICE, "-f", str(wav_path)],
                input=text.encode("utf-8"),
                check=True
            )
            logger.info(f"Piper generated WAV file: {wav_path}")

            # Convert WAV to MP3 using FFmpeg
            try:
                subprocess.run(
                    [FFMPEG_EXE, "-y", "-i", str(wav_path), "-ab", TTS_AUDIO_BITRATE, out_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )
                logger.info(f"FFmpeg converted to MP3: {out_path}")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "FFmpeg not found. Please install FFmpeg and add it to your system PATH, "
                    "or set FFMPEG_EXE to the full path of ffmpeg.exe in the code."
                )

            if wav_path.exists():
                wav_path.unlink()
                logger.info("Cleaned up temporary WAV file")

        result_container["audio_file"] = out_path
        logger.info(f"Audio generation complete. File: {out_path}")

    except Exception as e:
        logger.error(f"Audio generation failed: {str(e)}")
        result_container["error"] = str(e)


#Sidebar
st.sidebar.markdown("<div style='font-size: 0.75rem; font-weight: 600; color: #6b7280; margin-bottom: 10px; letter-spacing: 0.05em;'>NAVIGATION</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["💬 Ask a Question", "🕒 Evaluation History"], label_visibility="collapsed")

# Show session metrics in sidebar
# st.sidebar.markdown("---")
# st.sidebar.markdown("### 📊 Session Metrics")
# try:
#     summary = get_session_summary()
#     st.sidebar.metric("Total Queries", summary["total_queries"])
#     st.sidebar.metric("Error Rate", f"{summary['error_rate']:.1%}")
#     
#     if summary["latency_stats"].get("retrieval"):
#         retrieval_stats = summary["latency_stats"]["retrieval"]
#         st.sidebar.metric(
#             "Avg Retrieval Time",
#             f"{retrieval_stats['mean_ms']:.0f}ms"
#         )
#         )
#     
#     if summary["api_usage"]["tokens"]:
#         total_tokens = sum(summary["api_usage"]["tokens"].values())
#         st.sidebar.metric("Total Tokens Used", f"{total_tokens:,}")
# except Exception as e:
#     st.sidebar.caption("Metrics loading...")



#page1
if page == "💬 Ask a Question":

    st.markdown("""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
        <div style="background-color: #8b5cf6; width: 48px; height: 48px; border-radius: 12px; display: flex; justify-content: center; align-items: center;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        </div>
        <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: 700; color: #e5e7eb;">VideoRAG</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 0; font-size: 1.5rem; font-weight: 600; color: #d1d5db;'>Question Answering over YouTube Transcripts</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size: 1rem; margin-bottom: 20px; color: #9ca3af;'>"
        "Ask questions over video transcripts. Answers are streamed and then evaluated. "
        "Before asking a question, please review the README to understand the scope of inquiries."
        "</p>", unsafe_allow_html=True
    )
    
    # Show TTS mode indicator
    tts_mode = "gTTS (Cloud)" if USE_GTTS else "Piper (Local)"
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; background-color: rgba(249, 250, 251, 0.1); border: 1px solid rgba(229, 231, 235, 0.2); border-radius: 20px; font-size: 0.875rem; color: #d1d5db; margin-bottom: 30px;">
        🔊 TTS Mode: {tts_mode}
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic Ingestion Section
    with st.expander("➕ Add New Video to Knowledge Base"):
        st.markdown("Enter a YouTube URL to fetch its transcript and add it to the RAG knowledge base.")
        
        # Use a dynamic key to allow clearing
        if "url_input_key" not in st.session_state:
            st.session_state.url_input_key = 0
            
        input_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...", key=f"input_url_{st.session_state.url_input_key}")
        if st.button("Ingest Video"):
            if input_url:
                if "youtube.com" in input_url or "youtu.be" in input_url:
                    with st.status("Ingesting video content...", expanded=True) as status:
                        from ingestion_pipeline import ingest_video
                        progress_bar = st.progress(0)
                        
                        def update_progress(pct, msg):
                            progress_bar.progress(pct/100)
                            # Only write major steps to avoid cluttering the UI
                            if not msg.startswith("Generating embeddings... ("):
                                st.write(f"**{pct}%**: {msg}")
                        
                        success, message = ingest_video(input_url, progress_callback=update_progress)
                        if success:
                            if "already processed" in message:
                                status.update(label="✅ Video Already Exists!", state="complete", expanded=False)
                                st.info(message)
                            else:
                                status.update(label="✅ Ingestion Complete!", state="complete", expanded=False)
                                st.success(message)
                                st.info("You can now ask questions about this video.")
                        else:
                            status.update(label="❌ Ingestion Failed", state="error", expanded=True)
                            st.error(message)
                else:
                    st.error("Please enter a valid YouTube URL.")
            else:
                st.warning("Please enter a URL first.")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("💬 **Ask a question**")
        
        # text area
        st.session_state.question = st.text_area(
            "Ask a question",
            value=st.session_state.question,
            placeholder="e.g., How can geopolitics be operationalized as a tool for statecraft?",
            label_visibility="collapsed",
            height=150
        )

        col1, col2, col3 = st.columns([1, 0.15, 0.15])
        with col1:
            st.empty() # Removed the "Press Ctrl + Enter to submit" text
        with col2:
            clear_clicked = st.button("Clear", use_container_width=True)
        with col3:
            ask_clicked = st.button("Ask ➢", type="primary", use_container_width=True)

    #clear
    if clear_clicked:
        for k in default_state:
            if k == "input_url":
                continue # Skip input_url to avoid StreamlitAPIException
            st.session_state[k] = default_state[k]
        
        # Increment the key to force Streamlit to render a fresh, empty text input
        if "url_input_key" in st.session_state:
            st.session_state.url_input_key += 1
            
        if os.path.exists("answer.mp3"):
            os.remove("answer.mp3")
        st.rerun()

    
    if ask_clicked and st.session_state.question.strip():

        logger.info(f"Question asked: {st.session_state.question}")

        # reset per-question state
        st.session_state.rag_answer = None
        st.session_state.eval_out = None
        st.session_state.stream_done = False
        st.session_state.answer_placeholder_content = ""
        st.session_state.original_answer = None
        st.session_state.retried_answer = None
        st.session_state.original_confidence = None
        st.session_state.evaluation_requested = False
        #for audio
        st.session_state.audio_file = None
        st.session_state.audio_generating = False

        # Create placeholder for streaming answer
        answer_placeholder = st.empty()
        
        # Stream the answer in real-time
        full_answer = ""
        retrieved_contexts = []
        query_id = None
        
        with st.spinner("Retrieving relevant context..."):
            for chunk in ask_stream(st.session_state.question):
                if chunk["type"] == "token":
                    # Append token and update display in real-time (without header)
                    full_answer += chunk["content"]
                    answer_placeholder.markdown(f"{full_answer}▌", unsafe_allow_html=True)
                elif chunk["type"] == "done":
                    # Streaming complete, get contexts and query_id
                    retrieved_contexts = chunk["retrieved_contexts"]
                    query_id = chunk.get("query_id")
        
        # Clear the placeholder - the answer will be shown in the static section below
        answer_placeholder.empty()
        
        # Store in session state
        st.session_state.rag_answer = full_answer
        st.session_state.contexts = retrieved_contexts
        st.session_state.original_answer = full_answer
        st.session_state.answer_placeholder_content = full_answer
        st.session_state.stream_done = True
        st.session_state.query_id = query_id  # Store for metrics tracking

    # Display the answer (only once)
    if st.session_state.answer_placeholder_content:
        st.subheader("🧠 Answer")
        st.markdown(st.session_state.answer_placeholder_content, unsafe_allow_html=True)

    # Show evaluate button if answer is ready but evaluation not done
    if st.session_state.stream_done and not st.session_state.eval_out:
        st.divider()
        if st.button("🔍 Evaluate Answer Quality", use_container_width=True):
            st.session_state.evaluation_requested = True
            st.session_state.evaluation_in_progress = True
            st.rerun()

    # Run evaluation if requested
    if st.session_state.evaluation_requested and not st.session_state.eval_out and st.session_state.evaluation_in_progress:
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        result_container = {}
        question = st.session_state.question
        rag_answer = st.session_state.rag_answer
        contexts = st.session_state.contexts
        
        eval_thread = threading.Thread(
            target=run_evaluation_async,
            args=(result_container,question,rag_answer,contexts)
        )
        
        eval_thread.start()
        start_time = time.time()
        MAX_WAIT = 180  # 3 minutes expected
        
        while eval_thread.is_alive():
                elapsed = time.time() - start_time
                progress = min(elapsed / MAX_WAIT, 0.95)  # cap at 95%
                progress_bar.progress(progress)
                status.text(f"Evaluating answer quality(this may take 2–3 minutes)… {int(progress*100)}%")
                time.sleep(1)

        eval_thread.join()

        progress_bar.progress(1.0)
        status.text("✅ Evaluation complete")

        st.session_state.eval_out = result_container["output"]
        st.session_state.original_confidence = st.session_state.eval_out["confidence"]
        st.session_state.evaluation_requested = False
        st.session_state.evaluation_in_progress = False
        
        # Track query completion with confidence score
        if hasattr(st.session_state, 'query_id') and st.session_state.query_id:
            track_query(
                st.session_state.query_id,
                st.session_state.eval_out["confidence"],
                was_retry=False
            )
        
        
        time.sleep(0.5)
        progress_bar.empty()
        status.empty()
        st.rerun()

    # Show evaluation results
    if st.session_state.eval_out:

        st.divider()
        confidence = st.session_state.eval_out["confidence"]
        st.subheader("🔐 Answer Confidence")

        if confidence >= 0.75:
            st.success(f"High confidence: {confidence:.3f}")
        elif confidence >= 0.40:
            st.warning(f"Medium confidence: {confidence:.3f}")
        else:
            st.error(f"Low confidence: {confidence:.3f}")

        # Retry Button
        if confidence < CONFIDENCE_THRESHOLD:
            if st.button("🔁 Retry "):

                logger.info("Retry triggered")

                st.session_state.stream_done = False
                st.session_state.answer_placeholder_content = ""

                answer_placeholder = st.empty()

                # Stream the retry answer in real-time
                full_answer = ""
                retrieved_contexts = []
                
                with st.spinner("Retrying with broader retrieval (MMR ×2 candidates, BM25 ×2 docs, reranker top-15)..."):
                    for chunk in ask_stream_broad(st.session_state.question):
                        if chunk["type"] == "token":
                            full_answer += chunk["content"]
                            answer_placeholder.markdown(f"{full_answer}▌", unsafe_allow_html=True)
                        elif chunk["type"] == "done":
                            retrieved_contexts = chunk["retrieved_contexts"]
                
                # Clear the placeholder - the answer will be shown after rerun
                answer_placeholder.empty()

                st.session_state.retried_answer = full_answer
                st.session_state.rag_answer = full_answer
                st.session_state.contexts = retrieved_contexts
                st.session_state.answer_placeholder_content = full_answer
                st.session_state.stream_done = True

                with st.spinner("Re-evaluating retried answer..."):
                    st.session_state.eval_out = evaluate_answer(
                        question=st.session_state.question,
                        rag_answer=st.session_state.rag_answer,
                        retrieved_contexts=st.session_state.contexts
                    )
                
                st.rerun()

        
        if st.session_state.retried_answer:
            new_conf = st.session_state.eval_out["confidence"]
            delta = new_conf - st.session_state.original_confidence

            st.metric(
                label="Confidence Improvement",
                value=f"{new_conf:.3f}",
                delta=f"{delta:+.3f}"
            )

        # comparison of original vs retrieval answer
        if st.session_state.original_answer and st.session_state.retried_answer:

            st.divider()
            st.subheader("🆚 Original vs Retried Answer")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ❌ Original Answer")
                st.markdown(st.session_state.original_answer, unsafe_allow_html=True)

            with col2:
                st.markdown("### ✅ Retried Answer (Broader Retrieval)")
                st.markdown(st.session_state.retried_answer, unsafe_allow_html=True)

        # Final Evaluation Summary table
        st.divider()
        st.subheader("📊 Evaluation Summary")

        m = st.session_state.eval_out["metrics"]

        summary_df = pd.DataFrame([{
            "question": st.session_state.eval_out["question"],
            "answer": st.session_state.eval_out["rag_answer"],
            "reference": st.session_state.eval_out["reference_answer"],
            "context_precision": m.loc["context_precision", "score"],
            "context_recall": m.loc["context_recall", "score"],
            "answer_relevancy": m.loc["answer_relevancy", "score"],
            "faithfulness": m.loc["faithfulness", "score"],
            "confidence": st.session_state.eval_out["confidence"],
        }])

        st.dataframe(summary_df, use_container_width=True)

    # ═══════════════════════════════════════════════════
    # 🔊 AUDIO SECTION (Independent, Non-blocking)
    # ═══════════════════════════════════════════════════

    if st.session_state.stream_done and st.session_state.rag_answer:
            st.divider()
            st.subheader("🔊 Listen to the Answer")
            is_evaluating = st.session_state.evaluation_in_progress
            # If audio already exists
            if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                st.audio(st.session_state.audio_file, format="audio/mp3")
                with open(st.session_state.audio_file, "rb") as f:
                    st.download_button(
                        "⬇️ Download Audio",
                        f,
                        file_name="VideoRAG_Answer.mp3",
                        mime="audio/mpeg"
                    )
            # Show generate button (no reload)
            elif not st.session_state.audio_generating:
                if is_evaluating:
                    st.info("🔒 Audio generation will be available after evaluation completes")
                    st.button("🎧 Generate Audio", use_container_width=True, disabled=True)
                else:
                    if st.button("🎧 Generate Audio", use_container_width=True):
                        st.session_state.audio_generating = True
                        st.rerun()
            # Audio generation in progress
            if st.session_state.audio_generating and not st.session_state.audio_file:
                audio_container = {}
                audio_thread = threading.Thread(
                    target=run_audio_async,
                    args=(audio_container, st.session_state.rag_answer, "answer.mp3")
                )
                audio_thread.start()

                audio_progress = st.progress(0)
                audio_status = st.empty()
                start = time.time()
                while audio_thread.is_alive():
                    elapsed = time.time() - start
                    progress = min(elapsed / 20, 0.95)
                    audio_progress.progress(progress)
                    audio_status.text(f"🎵 Generating audio… {int(progress*100)}%")
                    time.sleep(0.3)

                audio_thread.join()
                audio_duration_ms = (time.time() - start) * 1000
                audio_progress.progress(1.0)
                audio_status.text("✅ Audio ready!")
                
                # Set audio file path and reset generating flag
                if "audio_file" in audio_container:
                    st.session_state.audio_file = audio_container["audio_file"]
                    # Track successful audio generation
                    if hasattr(st.session_state, 'query_id') and st.session_state.query_id:
                        get_metrics_tracker().track_audio_generation(
                            st.session_state.query_id,
                            audio_duration_ms,
                            success=True
                        )
                elif "error" in audio_container:
                    st.error(f"Audio generation failed: {audio_container['error']}")
                    # Track failed audio generation
                    if hasattr(st.session_state, 'query_id') and st.session_state.query_id:
                        get_metrics_tracker().track_audio_generation(
                            st.session_state.query_id,
                            audio_duration_ms,
                            success=False
                        )
                        track_error("audio_generation", audio_container['error'], st.session_state.query_id)
                    st.session_state.audio_generating = False
                    audio_progress.empty()
                    audio_status.empty()
                    st.stop()
                
                st.session_state.audio_generating = False
                
                # Clean up progress indicators before rerun
                time.sleep(1.0)  # Give user time to see "Audio ready!"
                audio_progress.empty()
                audio_status.empty()
                st.rerun()

# Page2 : Evaluation History
elif page == "🕒 Evaluation History":
    st.title("🕒 Evaluation History")
    if not os.path.exists(LOG_FILE):
        st.info("No evaluation logs found yet.")
        st.stop()

    df = pd.read_csv(LOG_FILE)

    st.metric("Total Logged Queries", len(df))
    st.metric("Average Confidence", round(df["confidence"].mean(), 3))

    st.divider()
    st.subheader("📉 Confidence Over Time")
    st.line_chart(df["confidence"])

    st.divider()
    st.subheader("❌ Top 10 Lowest Confidence Queries")
    st.dataframe(
        df.sort_values("confidence").head(10),
        use_container_width=True
    )

    st.divider()
    st.subheader("📄 Full Evaluation Log")
    st.dataframe(df, use_container_width=True)

# Page3: Metrics Dashboard (DISABLED/COMMENTED OUT)
# else:
#     st.title("📈 Performance Metrics Dashboard")
#     st.markdown("Real-time system performance and usage statistics")
#     
#     # Get current session summary
#     summary = get_session_summary()
#     
#     # Save metrics summary
#     if st.button("💾 Save Metrics Summary"):
#         save_metrics_summary()
#         st.success("Metrics summary saved to logs/metrics_summary.json")
#     
#     # Overview metrics
#     st.subheader("📊 Overview")
#     col1, col2, col3, col4 = st.columns(4)
#     
#     with col1:
#         st.metric("Total Queries", summary["total_queries"])
#     with col2:
#         st.metric("Total Errors", summary["total_errors"])
#     with col3:
#         st.metric("Error Rate", f"{summary['error_rate']:.1%}")
#     with col4:
#         st.metric("Retry Rate", f"{summary['retry_rate']:.1%}")
#     
#     st.divider()
#     
#     # Latency Statistics
#     st.subheader("⚡ Latency Statistics")
#     
#     if summary["latency_stats"]:
#         latency_data = []
#         for operation, stats in summary["latency_stats"].items():
#             latency_data.append({
#                 "Operation": operation.replace("_", " ").title(),
#                 "Count": stats["count"],
#                 "Mean (ms)": f"{stats['mean_ms']:.1f}",
#                 "Min (ms)": f"{stats['min_ms']:.1f}",
#                 "Max (ms)": f"{stats['max_ms']:.1f}",
#                 "P50 (ms)": f"{stats['p50_ms']:.1f}",
#                 "P95 (ms)": f"{stats['p95_ms']:.1f}",
#                 "P99 (ms)": f"{stats['p99_ms']:.1f}",
#             })
#         
#         latency_df = pd.DataFrame(latency_data)
#         st.dataframe(latency_df, use_container_width=True)
#         
#         # Latency chart
#         st.markdown("#### Latency Distribution")
#         chart_data = pd.DataFrame({
#             "Operation": [op.replace("_", " ").title() for op in summary["latency_stats"].keys()],
#             "Mean Latency (ms)": [stats["mean_ms"] for stats in summary["latency_stats"].values()],
#             "P95 Latency (ms)": [stats["p95_ms"] for stats in summary["latency_stats"].values()],
#         })
#         st.bar_chart(chart_data.set_index("Operation"))
#     else:
#         st.info("No latency data available yet. Run some queries to see statistics.")
#     
#     st.divider()
#     
#     # API Usage
#     st.subheader("🔌 API Usage & Cost Tracking")
#     
#     if summary["api_usage"]["calls"]:
#         col1, col2 = st.columns(2)
#         
#         with col1:
#             st.markdown("#### API Calls by Provider")
#             calls_df = pd.DataFrame([
#                 {"Provider": provider.title(), "Calls": count}
#                 for provider, count in summary["api_usage"]["calls"].items()
#             ])
#             st.dataframe(calls_df, use_container_width=True)
#         
#         with col2:
#             st.markdown("#### Token Usage by Provider")
#             tokens_df = pd.DataFrame([
#                 {"Provider": provider.title(), "Tokens": count}
#                 for provider, count in summary["api_usage"]["tokens"].items()
#             ])
#             st.dataframe(tokens_df, use_container_width=True)
#         
#         # Cost estimation (rough approximation)
#         st.markdown("#### 💰 Estimated Costs")
#         st.caption("Based on approximate pricing (actual costs may vary)")
#         
#         total_tokens = sum(summary["api_usage"]["tokens"].values())
#         # Rough cost estimates (update with actual pricing)
#         groq_tokens = summary["api_usage"]["tokens"].get("groq", 0)
#         google_tokens = summary["api_usage"]["tokens"].get("google", 0)
#         
#         # Groq is often free/very cheap, Gemini Flash is ~$0.075 per 1M tokens
#         estimated_cost = (google_tokens / 1_000_000) * 0.075
#         
#         st.metric("Total Tokens", f"{total_tokens:,}")
#         st.metric("Estimated Cost (USD)", f"${estimated_cost:.4f}")
#     else:
#         st.info("No API usage data available yet.")
#     
#     st.divider()
#     
#     # Raw metrics file
#     st.subheader("📄 Raw Metrics Data")
#     
#     metrics_file = get_metrics_tracker().metrics_file
#     if metrics_file.exists():
#         metrics_df = pd.read_csv(metrics_file)
#         
#         st.markdown(f"**Total Events Logged:** {len(metrics_df)}")
#         st.markdown(f"**Metrics File:** `{metrics_file}`")
#         
#         # Show recent events
#         st.markdown("#### Recent Events (Last 20)")
#         st.dataframe(
#             metrics_df.tail(20).sort_values("timestamp", ascending=False),
#             use_container_width=True
#         )
#         
#         # Download button
#         csv_data = metrics_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Full Metrics CSV",
#             data=csv_data,
#             file_name="videorag_metrics.csv",
#             mime="text/csv"
#         )
#     else:
#         st.info("No metrics file found yet.")