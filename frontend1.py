import streamlit as st
import logging
import os
import pandas as pd
import time
import re
import threading
import subprocess
from pathlib import Path


from retriever_pipeline2 import ask
from evaluation_pipeline import evaluate_answer


LOG_FILE = "logs/rag_evaluation_log.csv"
CONFIDENCE_THRESHOLD = 0.5
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


st.set_page_config(page_title="VideoRAG",page_icon="🎥",layout="wide")


# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true"


if IS_STREAMLIT_CLOUD:
    logger.info("Running on Streamlit Cloud - using gTTS")
    USE_GTTS = True
else:
    logger.info("Running locally - using Piper")
    USE_GTTS = False
    PIPER_EXE = r"D:\piper\piper.exe"
    PIPER_VOICE = r"D:\piper\en_US-lessac-medium.onnx"


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
        wav_path = Path("temp_voice.wav")

        # Run Piper (stdin → wav)
        subprocess.run(
            [PIPER_EXE, "-m", PIPER_VOICE, "-f", str(wav_path)],
            input=text.encode("utf-8"),
            check=True
        )

        # Convert WAV → MP3
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-ab", "192k", out_mp3],
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
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(out_path)
        else:
            # Local: Use Piper + FFmpeg
            wav_path = Path("temp_voice.wav")

            subprocess.run(
                [PIPER_EXE, "-m", PIPER_VOICE, "-f", str(wav_path)],
                input=text.encode("utf-8"),
                check=True
            )

            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_path), "-ab", "192k", out_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )

            if wav_path.exists():
                wav_path.unlink()

        result_container["audio_file"] = out_path

    except Exception as e:
        result_container["error"] = str(e)


#Sidebar
page = st.sidebar.radio("Navigation",["Ask a Question", "📊 Evaluation History"])



#page1
if page == "Ask a Question":

    st.title("🎥 VideoRAG")
    st.subheader("Question Answering over YouTube Transcripts")
    st.markdown(
        "Ask questions over video transcripts. "
        "Answers are streamed and then evaluated.Before Asking a Question Go Through the Readme file to know which all questions you can ask."
    )
    
    # Show TTS mode indicator
    tts_mode = "gTTS (Cloud)" if USE_GTTS else "Piper (Local)"
    st.caption(f"🔊 TTS Mode: {tts_mode}")
    
    st.divider()

    # text area
    st.session_state.question = st.text_area(
        "Ask a question",
        value=st.session_state.question,
        placeholder="How can geopolitics be operationalized as a tool for statecraft?"
    )

    col1, col2 = st.columns(2)
    ask_clicked = col1.button("Ask")
    clear_clicked = col2.button("Clear")

    #clear
    if clear_clicked:
        for k in default_state:
            st.session_state[k] = default_state[k]
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

        # we call the ask function in retriever_pipeline2.py
        with st.spinner("Generating answer..."):
            rag_out = ask(st.session_state.question)

        st.session_state.rag_answer = rag_out["answer"]
        st.session_state.contexts = rag_out["retrieved_contexts"]
        st.session_state.original_answer = rag_out["answer"]

        # stream the retrieved answer
        answers = split_answers(st.session_state.rag_answer)
        rendered = ""

        for ans in answers:
            rendered += ans + "\n\n"
            time.sleep(0.2)

        st.session_state.answer_placeholder_content = rendered
        st.session_state.stream_done = True

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
        elif confidence >= 0.45:
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

                with st.spinner("Retrying ..."):
                    retry_out = ask(st.session_state.question)

                st.session_state.retried_answer = retry_out["answer"]
                st.session_state.rag_answer = retry_out["answer"]
                st.session_state.contexts = retry_out["retrieved_contexts"]

                answers = split_answers(st.session_state.rag_answer)
                rendered = ""

                for ans in answers:
                    rendered += ans + "\n\n"
                    answer_placeholder.markdown(rendered, unsafe_allow_html=True)
                    time.sleep(0.2)

                st.session_state.answer_placeholder_content = rendered
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
                audio_progress.progress(1.0)
                audio_status.text("✅ Audio ready!")
                
                if "audio_file" in audio_container:
                    st.session_state.audio_file = audio_container["audio_file"]

                st.session_state.audio_generating = False
                time.sleep(0.5)
                st.rerun()

# Page2 : Evaluation History
else:
    st.title("📊 Evaluation History")
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