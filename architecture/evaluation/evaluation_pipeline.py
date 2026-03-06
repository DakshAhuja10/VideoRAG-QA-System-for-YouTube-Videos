#This file contains the code which is used evaluate our RAG system
#We evaluate our RAG System on the following 4 parameters-
        # 1.Context Precision : How noisy the retrieved context was
        # 2.Context Recall : Did retrieval cover most needed info
        # 3.Faithfulness : Did the answer stick to the provided context
        # 4.Answer Relevancy : Did the model actually answer the question
#for evaluating a rag on above parameters we need question,answer,context and ground truth
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import os
import re
import json
import csv
import pandas as pd
from datetime import datetime
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
)


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st

from ragas.prompt import PydanticPrompt

# Import configuration
from config import (
    RAG_EVALUATION_LOG_FILE,
    EmbeddingConfig,
    LLMConfig,
    EvaluationConfig,
    get_confidence_weights,
)

# Observability
from architecture.observability.langfuse_tracing import (
    start_span, end_span, log_generation, score_trace, timed_span,
)

# ── Why two separate LLM instances? ──────────────────────────────────────────
# Using the same model for both ground-truth generation and RAGAS evaluation
# creates circular scoring: the judge trivially rewards its own phrasing.
#
#   ground_truth_llm  →  llama-3.3-70b-versatile  (writes the reference answer)
#   llm (eval)        →  openai/gpt-oss-120b       (judges RAG answer vs reference)
#
# These are different model families on Groq, so the reference and the judge
# have genuinely independent perspectives.

# RAGAS configuration
PydanticPrompt.default_n = EvaluationConfig.RAGAS_DEFAULT_N

# Path where log file is stored/should be created
LOG_FILE = str(RAG_EVALUATION_LOG_FILE)
os.makedirs("logs", exist_ok=True)


# ── Cached resource loaders ──────────────────────────────────────────────────
# Both models are large and slow to initialise.  Caching them ensures they are
# loaded once per process and shared across all evaluation calls and sessions.

@st.cache_resource(show_spinner=False)
def _load_eval_embeddings():
    return HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)


@st.cache_resource(show_spinner=False)
def _load_eval_llm():
    return ChatGroq(
        model=LLMConfig.EVAL_MODEL,
        temperature=LLMConfig.EVAL_TEMPERATURE,
    )


@st.cache_resource(show_spinner=False)
def _load_ground_truth_llm():
    # Deliberately separate from _load_eval_llm — different model family
    return ChatGroq(
        model=LLMConfig.GROUND_TRUTH_MODEL,
        temperature=LLMConfig.GROUND_TRUTH_TEMPERATURE,
    )


embedding_model   = _load_eval_embeddings()
llm               = _load_eval_llm()        # RAGAS judge
ground_truth_llm  = _load_ground_truth_llm()  # reference answer writer


def extract_json(text: str):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())

# Compute confidence score using weighted metrics from config
# Weights are defined in config.py with explanations
def compute_confidence(row):
    weights = get_confidence_weights()
    return (
        weights["answer_relevancy"] * row["answer_relevancy"]
        + weights["faithfulness"] * row["faithfulness"]
        + weights["context_recall"] * row["context_recall"]
        + weights["context_precision"] * row["context_precision"]
    )


#to write the into log file
def log_evaluation(question, answer, reference, scores, confidence):
    write_header = not os.path.exists(LOG_FILE)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer,
        "reference": reference,
        "context_precision": scores["context_precision"],
        "context_recall": scores["context_recall"],
        "answer_relevancy": scores["answer_relevancy"],
        "faithfulness": scores["faithfulness"],
        "confidence": confidence,
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)



REFERENCE_PROMPT = """
You are a reference answer generator for a RAG evaluation system.

Using ONLY the context provided below, write a concise, factually grounded answer
to the question. Synthesize information from the context — do not require the exact
phrase to appear verbatim; it is enough if the context supports the answer.

If the context contains absolutely no information relevant to the question,
respond with exactly: {{"answer": "I don't know"}}

Return ONLY valid JSON with no preamble or explanation.

Format:
{{"answer": "..."}}

Context:
{context}

Question:
{question}
"""

# ground_truth_llm (qwen/qwen3-32b) generates the reference.
# llm (openai/gpt-oss-120b) judges via RAGAS — different model families, no circularity.
def generate_reference_answer(question, context):
    ground_truth_prompt = REFERENCE_PROMPT.format(question=question, context=context)
    response = ground_truth_llm.invoke(ground_truth_prompt).content

    try:
        return extract_json(response)["answer"]
    except Exception:
        return "I don't know"



def clean_answer_for_evaluation(answer: str) -> str:
    """
    Removes UI formatting (like markdown links and timestamps) from the answer
    so RAGAS can evaluate the semantic text without penalizing for 'hallucinated' links.
    """
    # Remove markdown links like [12:34](https://...)
    cleaned = re.sub(r'\[\d{1,2}:\d{2}\]\(.*?\)', '', answer)
    # Remove standalone URLs
    cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
    # Remove numbered lists if they are just formatting
    cleaned = re.sub(r'^\d+\.\s+', '', cleaned, flags=re.MULTILINE)
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def evaluate_answer(question, rag_answer, retrieved_contexts, trace=None):
    eval_span = start_span(trace, name="evaluation", input={"question": question})

    formatted_context = "\n".join(retrieved_contexts)

    with timed_span(eval_span, name="generate_reference") as (ref_span, _):
        reference = generate_reference_answer(question, formatted_context)
        end_span(ref_span, output={"reference": reference[:300]})

    log_generation(
        eval_span, name="ground_truth_llm", model=LLMConfig.GROUND_TRUTH_MODEL,
        input=question, output=reference,
        metadata={"purpose": "reference_answer_generation"},
    )
    
    # Clean the answer so RAGAS doesn't penalize UI formatting
    cleaned_rag_answer = clean_answer_for_evaluation(rag_answer)

    dataset = Dataset.from_pandas(pd.DataFrame([{
        "user_input": question,
        "response": cleaned_rag_answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference,
    }]))

    with timed_span(eval_span, name="ragas_evaluate") as (ragas_span, _):
        results = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                answer_relevancy,
                faithfulness,
            ],
            llm=llm,
            embeddings=embedding_model,
        )
        end_span(ragas_span)

    scores_df = results.to_pandas()
    numeric_scores = scores_df.loc[0, [
                "context_precision",
                "context_recall",
                "answer_relevancy",
                "faithfulness",
            ]]
    numeric_scores = pd.to_numeric(numeric_scores, errors="coerce").fillna(0.0)
    final_score_mean = numeric_scores.mean()


    confidence = compute_confidence(numeric_scores)

    # Push RAGAS scores + confidence to Langfuse
    if trace:
        for metric_name in ["context_precision", "context_recall", "answer_relevancy", "faithfulness"]:
            score_trace(trace, name=metric_name, value=float(numeric_scores[metric_name]))
        score_trace(trace, name="confidence", value=float(confidence))
        score_trace(trace, name="ragas_mean", value=float(final_score_mean))

    end_span(eval_span, output={
        "confidence": round(float(confidence), 4),
        "context_precision": round(float(numeric_scores["context_precision"]), 4),
        "context_recall": round(float(numeric_scores["context_recall"]), 4),
        "answer_relevancy": round(float(numeric_scores["answer_relevancy"]), 4),
        "faithfulness": round(float(numeric_scores["faithfulness"]), 4),
    })

    # Always log all evaluations to build a complete history
    log_evaluation(
        question,
        rag_answer,
        reference,
        numeric_scores,
        confidence,
    )

    return {
        "question": question,
        "rag_answer": rag_answer,
        "reference_answer": reference,
        "retrieved_contexts": retrieved_contexts,
        "metrics": numeric_scores.to_frame(name="score"),
        "confidence": confidence,
        "final_score_mean": final_score_mean,
    }
