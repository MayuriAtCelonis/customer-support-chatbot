# customer-support-chatbot
This project is to build an AI customer support assistant using a foundational LLM and Retrieval-Augmented Generation (RAG). The main tasks involve implementing the RAG system, ensuring the AI's decisions are explainable, and deploying it with a simple API.

# env variables:
openai_api_key=

# Command to start qdrant db
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

## API Documentation

### POST /process_chat_history

Processes a list of chat messages and returns an answer, reasoning, and evaluation scores.

#### Request Body

```json
{
  "chat_history": [
    {"role": "user", "content": "How do I reset my password?"},
    {"role": "assistant", "content": "You can reset it via the settings page."}
  ],
  "enable_reasoning": true
}
```

- `chat_history` (array, required): List of chat messages, each with:
  - `role` (string): "user" or "assistant"
  - `content` (string): The message text
- `enable_reasoning` (boolean, optional): Whether to include step-by-step reasoning in the response (default: false)

#### Response

```json
{
  "answer": "You can reset your password by clicking on 'Forgot Password' at the login page.",
  "reasoning": "The user asked about password reset. The answer is based on standard procedure.",
  "scores": null,
  "success": true,
  "evaluations": {"retrieved_examples_mean_distance": 0.12}
}
```

- `answer` (string): The generated answer to the chat history
- `reasoning` (string, optional): Step-by-step reasoning for the answer (if enabled)
- `scores` (object, optional): Scoring or evaluation details, if available
- `success` (boolean, optional): Indicates if the operation was successful
- `evaluations` (object, optional): Additional evaluation metrics, if available

#### Possible Errors

- **400**: Chat history cannot be empty
- **422**: Invalid request format (e.g., missing required fields)
- **500**: Internal server error (e.g., LLM or retrieval failure)