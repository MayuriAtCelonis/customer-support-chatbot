# customer-support-chatbot
This project is to build an AI customer support assistant using a foundational LLM and Retrieval-Augmented Generation (RAG). The main tasks involve implementing the RAG system, ensuring the AI's decisions are explainable, and deploying it with a simple API.

# env variables:
openai_api_key=

# Command to start qdrant db
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant