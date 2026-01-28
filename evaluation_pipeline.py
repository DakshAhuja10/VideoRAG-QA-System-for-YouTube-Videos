#This file contains the code which is used evaluate our RAG system
#We evaluate our RAG System on the following 4 parameters-
        # 1.Context Precision : How noisy the retrieved context was
        # 2.Context Recall : Did retrieval cover most needed info
        # 3.Faithfulness : Did the answer stick to the provided context
        # 4.Answer Relevancy : Did the model actually answer the question
#for evaluating a rag on above parameters we need question,answer,context and ground truth
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

from ragas.prompt import PydanticPrompt
PydanticPrompt.default_n = 1

#Path Where log file is stored/should be created
LOG_FILE = "logs/rag_evaluation_log.csv"
os.makedirs("logs", exist_ok=True)


embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0)


def extract_json(text: str):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found")
    return json.loads(match.group())

#returns the confidence score and we focused mainly on did answer stick to the context or not??
def compute_confidence(row):
    return (
        0.30 * row["answer_relevancy"]
        + 0.40 * row["faithfulness"]
        + 0.20 * row["context_recall"]
        + 0.10 * row["context_precision"]
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

#here we are using the llm to generate the reference answer or the ground truth
def generate_reference_answer(question, context):
    ground_truth_prompt=REFERENCE_PROMPT.format(question=question,context=context)
    response = llm.invoke(ground_truth_prompt).content

    try:
        return extract_json(response)["answer"]
    except:
        return "I don't know"



def evaluate_answer(question, rag_answer, retrieved_contexts):
    formatted_context = "\n".join(retrieved_contexts)
    reference = generate_reference_answer(question,formatted_context)

    dataset = Dataset.from_pandas(pd.DataFrame([{
        "user_input": question,
        "response": rag_answer,
        "retrieved_contexts": retrieved_contexts,
        "reference": reference,
    }]))

    
    # ragas framwork to calculate the values of all ragas parameters
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

    # It automatically logs all low scores
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
