
from ai.retrieval.document_retrieval import search_similar
from ai.retrieval.vector_db_connect import (
    QDRANT_COLLECTION,
    init_vector_db_connections
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
    Uses vector_db_connect for all setup/caching.
    """
    if collection_name is None:
        collection_name = QDRANT_COLLECTION
    if client is None or model is None:
        client, model = init_vector_db_connections(collection_name)
    results = search_similar(client, model, query, collection_name, top_k=top_k)
    return [
        {
            "score": r["score"],
            "document": {
                "question": r["input"],
                "answer": r["reply"]
            }
        }
        for r in results
    ]
