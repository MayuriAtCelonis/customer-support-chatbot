
from ai.retrieval.document_retrieval import search_similar
from ai.retrieval.embeddings_helper import (
    QDRANT_COLLECTION,
    init_embeddings_helperions
)

def retrieve_releveant_context(
    query,
    client=None,
    model=None,
    collection_name=None,
    top_k=5
):
    """
    Retrieve answers for a given query.
    In production, only query is required.
    For testing, client/model/collection_name can be injected.
    Uses embeddings_helper for all setup/caching.
    """
    if collection_name is None:
        collection_name = QDRANT_COLLECTION
    if client is None or model is None:
        client, model = init_embeddings_helperions(collection_name)
    results, mean_inter_document_similarity, median_inter_document_similarity, mean_document_query_similarity, median_document_query_similarity = search_similar(client, model, query, collection_name, top_k=top_k)
    return [
        {
            "score": r["score"],
            "document": {
                "question": r["input"],
                "answer": r["reply"]
            }
        }
        for r in results
    ], mean_inter_document_similarity, median_inter_document_similarity, mean_document_query_similarity, median_document_query_similarity
