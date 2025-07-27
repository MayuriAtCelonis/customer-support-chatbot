import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from ai.conversational.orchestrator import process_chat_history

def evaluate_with_process_chat_history(chat_history):
    """
    Evaluates the output of process_chat_history using RAGAS metrics (without ground truth).
    
    Args:
        chat_history (List[Dict[str, str]]): The chat history to process.
    """
    response = process_chat_history(chat_history)
    question = None
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            question = msg.get("content")
            break
    if not question:
        raise ValueError("No user message found in chat_history.")

    answer = response.get("answer", "")
    relevant_documents = response.get("relevant_documents", [])
    contexts = []
    if relevant_documents:
        for doc in relevant_documents:
            doc_str = ""
            if isinstance(doc, dict):
                doc_str = f"Question: {doc.get('question', '')} Answer: {doc.get('answer', '')}"
            else:
                doc_str = str(doc)
            contexts.append(doc_str)
    if not contexts:
        contexts = [""] 

    data_samples = {
        'question': [question],
        'answer': [answer],
        'contexts': [contexts],
    }
    dataset = Dataset.from_dict(data_samples)

    metrics_to_use = [
        faithfulness,
        answer_relevancy,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_use,
    )

    df = result.to_pandas()
    return df
