from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_COLLECTION = "customer_support_tweets"

_cached_client = None
_cached_model = None
qdrant_port = 6333

def get_sentence_transformer(model_name="all-MiniLM-L6-v2"):
    global _cached_model
    if _cached_model is None:
        _cached_model = SentenceTransformer(model_name)
    return _cached_model

def initialize_qdrant(collection_name, vector_size, client=None):
    if client is None:
        client = QdrantClient(host="localhost", port=qdrant_port)
    if client.collection_exists(collection_name) == False:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )
    return client

def get_qdrant_client(collection_name=QDRANT_COLLECTION, model_name="all-MiniLM-L6-v2"):
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    model = get_sentence_transformer(model_name)
    vector_size = model.get_sentence_embedding_dimension()
    _cached_client = initialize_qdrant(collection_name, vector_size)
    return _cached_client

def init_vector_db_connections(collection_name=QDRANT_COLLECTION, model_name="all-MiniLM-L6-v2"):
    client = get_qdrant_client(collection_name, model_name)
    model = get_sentence_transformer(model_name)
    return client, model
