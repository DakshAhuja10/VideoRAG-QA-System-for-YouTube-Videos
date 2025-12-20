# ============================================================
# evaluation_pipeline.py
# ============================================================

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
from ragas.prompt import PydanticPrompt

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
PydanticPrompt.default_n = 1

LOG_FILE = "logs/rag_evaluation_log.csv"
os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------------
# LLM + EMBEDDINGS (ONLY FOR EVALUATION)
# ------------------------------------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def extract_json(text: str):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())


def compute_confidence(row):
    """
    Single scalar confidence score
    """
    return (
        0.40 * row["answer_relevancy"]
        + 0.30 * row["faithfulness"]
        + 0.20 * row["context_recall"]
        + 0.10 * row["context_precision"]
    )


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


# ------------------------------------------------------------
# REFERENCE ANSWER (GROUND TRUTH)
# ------------------------------------------------------------
REFERENCE_PROMPT = """
Answer the question using ONLY the context.
If the answer is not explicitly present, say "I don't know".

Return ONLY valid JSON.

Format:
{{"answer": "..."}}

Context:
{context}

Question:
{question}
"""


def generate_reference_answer(question, context):
    response = llm.invoke(
        REFERENCE_PROMPT.format(
            question=question,
            context=context
        )
    ).content

    try:
        return extract_json(response)["answer"]
    except:
        return "I don't know"


# ------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ------------------------------------------------------------
def evaluate_answer(question, rag_answer, retrieved_contexts):
    """
    Inputs come FROM retriever_pipeline2.ask()
    """

    formatted_context = "\n".join(retrieved_contexts)

    reference = generate_reference_answer(
        question,
        formatted_context
    )

    dataset = Dataset.from_pandas(pd.DataFrame([{
        "user_input": question,
        "response": rag_answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference,
    }]))

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

    scores_df = results.to_pandas()
    numeric_scores = scores_df.loc[0, [
                "context_precision",
                "context_recall",
                "answer_relevancy",
                "faithfulness",
            ]].astype(float)

    final_score_mean = numeric_scores.mean()


    confidence = compute_confidence(numeric_scores)

    # Auto-log low confidence or "I don't know"
    if confidence < 0.6 or "I don't know" in rag_answer:
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
