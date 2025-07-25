import logging
import pandas as pd
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
    Returns list of matched documents with scores, and mean/median distances for all and top_p results.
    """
    query_vec = model.encode([query])[0].tolist()
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vec,
        limit=top_k
    )
    if not hits or len(hits) == 0:
        return [], None, None, None, None
    results = []
    for hit in hits:
        results.append({
            "id": hit.id,
            "score": hit.score,
            "input": hit.payload.get("input"),
            "reply": hit.payload.get("reply")
        })

    top_p_results = top_p_filtering_with_temperature(results=results)
    mean_inter_document_similarity, median_inter_document_similarity = find_inter_document_similarity(results, model)
    # Calculate mean/median of query-to-document similarity (score) for top_p_results
    if top_p_results:
        scores = [r["score"] for r in top_p_results]
        mean_document_query_similarity = float(np.mean(scores)) if scores else None
        median_document_query_similarity = float(np.median(scores)) if scores else None
    else:
        mean_document_query_similarity, median_document_query_similarity = None, None
    return top_p_results, mean_inter_document_similarity, median_inter_document_similarity, mean_document_query_similarity, median_document_query_similarity


def find_inter_document_similarity(results, model=None):
    """
    Find inter-document similarity within a collection.
    Returns mean and median of pairwise cosine similarities between documents.
    """
    df_documents = pd.DataFrame(results)
    df_documents['input'] = df_documents['input'].astype(str)
    df_documents['reply'] = df_documents['reply'].astype(str)
    df_documents['text'] = 'Question : ' + df_documents['input'] + " Answer : " + df_documents['reply']
    doc_vectors = model.encode(df_documents['text'].to_list(),
                          batch_size=32,  # Adjust batch_size based on your GPU memory
                          show_progress_bar=True,
                          convert_to_tensor=True, # Returns torch.Tensor, can be faster for some downstream tasks
                    )

    # Compute pairwise cosine similarity
    similarities = np.dot(doc_vectors.cpu().numpy(), np.transpose(doc_vectors.cpu().numpy()))
    np.fill_diagonal(similarities, np.nan)  # Ignore self-similarity

    # Flatten the similarity matrix and remove NaN values
    similarity_values = similarities[np.isfinite(similarities)].flatten()

    mean_similarity = np.mean(similarity_values) if similarity_values.size > 0 else None
    median_similarity = np.median(similarity_values) if similarity_values.size > 0 else None

    return mean_similarity, median_similarity
