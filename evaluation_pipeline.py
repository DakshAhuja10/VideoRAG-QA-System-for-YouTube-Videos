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

# Import configuration
from config import (
    RAG_EVALUATION_LOG_FILE,
    EmbeddingConfig,
    LLMConfig,
    EvaluationConfig,
    get_confidence_weights,
)

# RAGAS configuration
PydanticPrompt.default_n = EvaluationConfig.RAGAS_DEFAULT_N

# Path where log file is stored/should be created
LOG_FILE = str(RAG_EVALUATION_LOG_FILE)
os.makedirs("logs", exist_ok=True)


# Initialize models with config
embedding_model = HuggingFaceEmbeddings(model_name=EmbeddingConfig.MODEL_NAME)

llm = ChatGroq(
    model=LLMConfig.EVAL_MODEL,
    temperature=LLMConfig.EVAL_TEMPERATURE
)


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
    except Exception as e:
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

def evaluate_answer(question, rag_answer, retrieved_contexts):
    formatted_context = "\n".join(retrieved_contexts)
    reference = generate_reference_answer(question,formatted_context)
    
    # Clean the answer so RAGAS doesn't penalize UI formatting
    cleaned_rag_answer = clean_answer_for_evaluation(rag_answer)

    dataset = Dataset.from_pandas(pd.DataFrame([{
        "user_input": question,
        "response": cleaned_rag_answer,
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
            ]]
    numeric_scores = pd.to_numeric(numeric_scores, errors="coerce").fillna(0.0)
    final_score_mean = numeric_scores.mean()


    confidence = compute_confidence(numeric_scores)

    # Always log all evaluations to build a complete history
    # This allows tracking performance trends over time
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
