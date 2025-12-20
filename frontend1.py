# import streamlit as st
# import logging
# import os
# import pandas as pd
# import time
# import re

# from retriever_pipeline2 import ask
# from evaluation_pipeline import evaluate_answer

# # ------------------------------------------------------------
# # CONSTANTS
# # ------------------------------------------------------------
# LOG_FILE = "logs/rag_evaluation_log.csv"
# CONFIDENCE_THRESHOLD = 0.6

# # ------------------------------------------------------------
# # LOGGING (STREAMLIT-SAFE)
# # ------------------------------------------------------------
# os.makedirs("logs", exist_ok=True)

# root_logger = logging.getLogger()

# if not root_logger.handlers:
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(message)s",
#         handlers=[
#             logging.FileHandler("logs/app.log"),
#             logging.StreamHandler()
#         ]
#     )

# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("groq").setLevel(logging.WARNING)
# logging.getLogger("ragas").setLevel(logging.WARNING)

# logger = logging.getLogger(__name__)

# # ------------------------------------------------------------
# # STREAMLIT CONFIG
# # ------------------------------------------------------------
# st.set_page_config(
#     page_title="VideoRAG",
#     page_icon="🎥",
#     layout="wide"
# )

# # ------------------------------------------------------------
# # HELPERS
# # ------------------------------------------------------------
# def split_answers(answer_text: str):
#     """
#     Splits LLM output into full answers based on numbering.
#     Preserves multi-sentence structure.
#     """
#     pattern = r"\n(?=\d+\.\s)"
#     return [a.strip() for a in re.split(pattern, answer_text.strip()) if a.strip()]

# # ------------------------------------------------------------
# # SESSION STATE INITIALIZATION
# # ------------------------------------------------------------
# default_state = {
#     "question": "",
#     "rag_answer": None,
#     "contexts": None,
#     "eval_out": None,
#     "stream_done": False,
#     "rendered_answer": "",   # 🔑 persists streamed UI
# }

# for k, v in default_state.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # ------------------------------------------------------------
# # SIDEBAR NAVIGATION
# # ------------------------------------------------------------
# page = st.sidebar.radio(
#     "Navigation",
#     ["Ask a Question", "📊 Evaluation History"]
# )

# # ============================================================
# # PAGE 1: ASK A QUESTION
# # ============================================================
# if page == "Ask a Question":

#     st.title("🎥 VideoRAG")
#     st.subheader("Question Answering over YouTube Transcripts")

#     st.markdown(
#         "Ask questions over video transcripts. "
#         "Answers are streamed and then evaluated."
#     )

#     st.divider()

#     # --------------------------------------------------------
#     # QUESTION INPUT
#     # --------------------------------------------------------
#     st.session_state.question = st.text_area(
#         "Ask a question",
#         value=st.session_state.question,
#         placeholder="What does Lex say about deep learning?"
#     )

#     col1, col2 = st.columns(2)
#     ask_clicked = col1.button("Ask")
#     clear_clicked = col2.button("Clear")

#     # --------------------------------------------------------
#     # CLEAR BUTTON
#     # --------------------------------------------------------
#     if clear_clicked:
#         for k in default_state:
#             st.session_state[k] = default_state[k]
#         st.rerun()

#     # --------------------------------------------------------
#     # ASK BUTTON
#     # --------------------------------------------------------
#     if ask_clicked and st.session_state.question.strip():

#         logger.info(f"Question asked: {st.session_state.question}")

#         st.session_state.rag_answer = None
#         st.session_state.eval_out = None
#         st.session_state.stream_done = False
#         st.session_state.rendered_answer = ""

#         answer_placeholder = st.empty()

#         # -------- 1️⃣ GENERATE ANSWER --------
#         with st.spinner("Generating answer..."):
#             rag_out = ask(st.session_state.question)

#         st.session_state.rag_answer = rag_out["answer"]
#         st.session_state.contexts = rag_out["retrieved_contexts"]

#         # -------- 2️⃣ STREAM ANSWER --------
#         answers = split_answers(st.session_state.rag_answer)

#         rendered = ""
#         for ans in answers:
#             rendered += ans + "\n\n"
#             answer_placeholder.markdown(rendered, unsafe_allow_html=True)
#             time.sleep(0.2)

#         st.session_state.rendered_answer = rendered
#         st.session_state.stream_done = True

#         # -------- 3️⃣ RUN EVALUATION --------
#         with st.spinner("Evaluating answer quality..."):
#             st.session_state.eval_out = evaluate_answer(
#                 question=st.session_state.question,
#                 rag_answer=st.session_state.rag_answer,
#                 retrieved_contexts=st.session_state.contexts
#             )

#     # --------------------------------------------------------
#     # DISPLAY RESULTS
#     # --------------------------------------------------------
#     if st.session_state.stream_done and st.session_state.eval_out:


#         st.divider()

#         confidence = st.session_state.eval_out["confidence"]
#         st.subheader("🔐 Answer Confidence")

#         if confidence >= 0.75:
#             st.success(f"High confidence: {confidence:.3f}")
#         elif confidence >= 0.5:
#             st.warning(f"Medium confidence: {confidence:.3f}")
#         else:
#             st.error(f"Low confidence: {confidence:.3f}")

#         # ----------------------------------------------------
#         # RETRY BUTTON
#         # ----------------------------------------------------
#         if confidence < CONFIDENCE_THRESHOLD:
#             if st.button("🔁 Retry with broader retrieval"):

#                 logger.info("Retry triggered")

#                 st.session_state.stream_done = False
#                 st.session_state.rendered_answer = ""

#                 answer_placeholder = st.empty()

#                 with st.spinner("Retrying answer..."):
#                     retry_out = ask(st.session_state.question)

#                 st.session_state.rag_answer = retry_out["answer"]
#                 st.session_state.contexts = retry_out["retrieved_contexts"]

#                 answers = split_answers(st.session_state.rag_answer)

#                 rendered = ""
#                 for ans in answers:
#                     rendered += ans + "\n\n"
#                     answer_placeholder.markdown(rendered, unsafe_allow_html=True)
#                     time.sleep(0.2)

#                 st.session_state.rendered_answer = rendered
#                 st.session_state.stream_done = True

#                 with st.spinner("Re-evaluating retried answer..."):
#                     st.session_state.eval_out = evaluate_answer(
#                         question=st.session_state.question,
#                         rag_answer=st.session_state.rag_answer,
#                         retrieved_contexts=st.session_state.contexts
#                     )

#         # ----------------------------------------------------
#         # EVALUATION SUMMARY TABLE
#         # ----------------------------------------------------
#         st.subheader("📊 Evaluation Summary")

#         m = st.session_state.eval_out["metrics"]

#         summary_df = pd.DataFrame([{
#             "question": st.session_state.eval_out["question"],
#             "answer": st.session_state.eval_out["rag_answer"],
#             "reference": st.session_state.eval_out["reference_answer"],
#             "context_precision": m.loc["context_precision", "score"],
#             "context_recall": m.loc["context_recall", "score"],
#             "answer_relevancy": m.loc["answer_relevancy", "score"],
#             "faithfulness": m.loc["faithfulness", "score"],
#             "confidence": confidence,
#         }])

#         st.dataframe(summary_df, use_container_width=True)

# # ============================================================
# # PAGE 2: EVALUATION HISTORY
# # ============================================================
# else:

#     st.title("📊 Evaluation History")

#     if not os.path.exists(LOG_FILE):
#         st.info("No evaluation logs found yet.")
#         st.stop()

#     df = pd.read_csv(LOG_FILE)

#     st.metric("Total Logged Queries", len(df))
#     st.metric("Average Confidence", round(df["confidence"].mean(), 3))

#     st.divider()
#     st.subheader("📉 Confidence Over Time")
#     st.line_chart(df["confidence"])

#     st.divider()
#     st.subheader("❌ Lowest Confidence Queries")
#     st.dataframe(
#         df.sort_values("confidence").head(10),
#         use_container_width=True
#     )

#     st.divider()
#     st.subheader("📄 Full Evaluation Log")
#     st.dataframe(df, use_container_width=True)




# ============================================================
# frontend.py
# ============================================================

import streamlit as st
import logging
import os
import pandas as pd
import time
import re

from retriever_pipeline2 import ask
from evaluation_pipeline import evaluate_answer

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
LOG_FILE = "logs/rag_evaluation_log.csv"
CONFIDENCE_THRESHOLD = 0.6

# ------------------------------------------------------------
# LOGGING (STREAMLIT-SAFE)
# ------------------------------------------------------------
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

for noisy in ["httpx", "groq", "ragas"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="VideoRAG",
    page_icon="🎥",
    layout="wide"
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def split_answers(answer_text: str):
    """
    Splits numbered answers while preserving paragraphs.
    """
    pattern = r"\n(?=\d+\.\s)"
    return [a.strip() for a in re.split(pattern, answer_text.strip()) if a.strip()]

# ------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------
default_state = {
    "question": "",
    "rag_answer": None,
    "contexts": None,
    "eval_out": None,
    "stream_done": False,
    "rendered_answer": "",
    "original_answer": None,
    "retried_answer": None,
    "original_confidence": None,
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Ask a Question", "📊 Evaluation History"]
)

# ============================================================
# PAGE 1: ASK A QUESTION
# ============================================================
if page == "Ask a Question":

    st.title("🎥 VideoRAG")
    st.subheader("Question Answering over YouTube Transcripts")
    st.markdown(
        "Ask questions over video transcripts. "
        "Answers are streamed and then evaluated."
    )
    st.divider()

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------
    st.session_state.question = st.text_area(
        "Ask a question",
        value=st.session_state.question,
        placeholder="How can geopolitics be operationalized as a tool for statecraft?"
    )

    col1, col2 = st.columns(2)
    ask_clicked = col1.button("Ask")
    clear_clicked = col2.button("Clear")

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------
    if clear_clicked:
        for k in default_state:
            st.session_state[k] = default_state[k]
        st.rerun()

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------
    if ask_clicked and st.session_state.question.strip():

        logger.info(f"Question asked: {st.session_state.question}")

        # reset per-question state
        st.session_state.rag_answer = None
        st.session_state.eval_out = None
        st.session_state.stream_done = False
        st.session_state.rendered_answer = ""
        st.session_state.original_answer = None
        st.session_state.retried_answer = None
        st.session_state.original_confidence = None

        answer_placeholder = st.empty()

        # -------- 1️⃣ RETRIEVE + GENERATE --------
        with st.spinner("Generating answer..."):
            rag_out = ask(st.session_state.question)

        st.session_state.rag_answer = rag_out["answer"]
        st.session_state.contexts = rag_out["retrieved_contexts"]
        st.session_state.original_answer = rag_out["answer"]

        # -------- 2️⃣ STREAM ANSWER --------
        answers = split_answers(st.session_state.rag_answer)
        rendered = ""

        for ans in answers:
            rendered += ans + "\n\n"
            answer_placeholder.markdown(rendered, unsafe_allow_html=True)
            time.sleep(0.2)

        st.session_state.rendered_answer = rendered
        st.session_state.stream_done = True

        # -------- 3️⃣ EVALUATE --------
        with st.spinner("Evaluating answer quality..."):
            st.session_state.eval_out = evaluate_answer(
                question=st.session_state.question,
                rag_answer=st.session_state.rag_answer,
                retrieved_contexts=st.session_state.contexts
            )

        st.session_state.original_confidence = st.session_state.eval_out["confidence"]

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------
    if st.session_state.stream_done and st.session_state.eval_out:

        st.divider()

        confidence = st.session_state.eval_out["confidence"]
        st.subheader("🔐 Answer Confidence")

        if confidence >= 0.75:
            st.success(f"High confidence: {confidence:.3f}")
        elif confidence >= 0.5:
            st.warning(f"Medium confidence: {confidence:.3f}")
        else:
            st.error(f"Low confidence: {confidence:.3f}")

        # ----------------------------------------------------
        # RETRY
        # ----------------------------------------------------
        if confidence < CONFIDENCE_THRESHOLD:
            if st.button("🔁 Retry with broader retrieval"):

                logger.info("Retry triggered")

                st.session_state.stream_done = False
                st.session_state.rendered_answer = ""

                answer_placeholder = st.empty()

                with st.spinner("Retrying with broader retrieval..."):
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

                st.session_state.rendered_answer = rendered
                st.session_state.stream_done = True

                with st.spinner("Re-evaluating retried answer..."):
                    st.session_state.eval_out = evaluate_answer(
                        question=st.session_state.question,
                        rag_answer=st.session_state.rag_answer,
                        retrieved_contexts=st.session_state.contexts
                    )

        # ----------------------------------------------------
        # CONFIDENCE IMPROVEMENT
        # ----------------------------------------------------
        if st.session_state.retried_answer:
            new_conf = st.session_state.eval_out["confidence"]
            delta = new_conf - st.session_state.original_confidence

            st.metric(
                label="Confidence Improvement",
                value=f"{new_conf:.3f}",
                delta=f"{delta:+.3f}"
            )

        # ----------------------------------------------------
        # ORIGINAL vs RETRIED COMPARISON
        # ----------------------------------------------------
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

        # ----------------------------------------------------
        # EVALUATION SUMMARY
        # ----------------------------------------------------
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

# ============================================================
# PAGE 2: EVALUATION HISTORY
# ============================================================
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
    st.subheader("❌ Lowest Confidence Queries")
    st.dataframe(
        df.sort_values("confidence").head(10),
        use_container_width=True
    )

    st.divider()
    st.subheader("📄 Full Evaluation Log")
    st.dataframe(df, use_container_width=True)
