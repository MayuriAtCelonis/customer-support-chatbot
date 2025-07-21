import logging
# from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models  # <-- ADD THIS IMPORT
from qdrant_client.http.models import VectorParams
import numpy as np

def top_p_filtering_with_temperature(results, p=0.9, temperature=0.1, score_key="score"):
    """
    Apply temperature to scores, then do top-p (nucleus) filtering.
    """
    if not results:
        return []
    scores = np.array([r[score_key] for r in results])
    # Apply temperature to scores
    if temperature != 1.0:
        scores = scores / temperature
    # Softmax to get probabilities
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    probs = exp_scores / np.sum(exp_scores)
    # Sort results by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_results = [results[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]
    # Top-p filtering
    cumulative = 0.0
    filtered = []
    for r, prob in zip(sorted_results, sorted_probs):
        filtered.append(r)
        cumulative += prob
        if cumulative >= p:
            break
    return filtered

def search_similar(client, model, query, collection_name, top_k=5):
    """
    Given a query string, embed and search in Qdrant.
    Returns list of matched documents with scores.
    """
    query_vec = model.encode([query])[0].tolist()
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "input": hit.payload.get("input"),
            "reply": hit.payload.get("reply")
        })

    top_p_results = top_p_filtering_with_temperature(results=results)
    print(f"{len(results)=} | {len(top_p_results)=} ")
    return top_p_results