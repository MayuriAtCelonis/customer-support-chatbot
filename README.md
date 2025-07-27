# customer-support-chatbot
This project is to build an AI customer support assistant using a foundational LLM and Retrieval-Augmented Generation (RAG). The main tasks involve implementing the RAG system, ensuring the AI's decisions are explainable, and deploying it with a simple API.

## Project Structure

- `ai/` — Core AI modules (generation, retrieval, orchestration, evaluation)
- `api/` — API server code
- `requirements.txt` — Python dependencies
- `README.md` — This documentation

## Quickstart

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd customer-support-chatbot
   ```

2. **Set your environment variables**

   Create a `.env` file or export the required variables in your shell:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```
3. **build and run the docker**

4. **Run the load_example_data.ipynb notebook to upload the sample data(only 1000 rows as test)**


5. **Test the API**

   Use :
   ```bash
curl -X POST "http://localhost:8000/generate_response" \
     -H "Content-Type: application/json" \
     -d '{"user_query": "Please help me login", "enable_reasoning": true}'

   ```


## Notes

- Retrieval-augmented generation (RAG) is uses public dataset

For more details, see the API documentation below.

## 🛠️ API Documentation

### POST `/generate_response`

Process a user query, optionally continue a conversation, and receive an AI-generated answer with optional reasoning and evaluation metrics.

---

#### Request Body

Send a JSON object with the following fields:

| Field            | Type    | Required | Description                                                                                 |
|------------------|---------|----------|---------------------------------------------------------------------------------------------|
| user_query       | string  | Yes      | The user's message or question.                                                             |
| enable_reasoning | boolean | No       | Whether to enable step-by-step reasoning in the response. Default: `true`.                  |
| conversation_id  | string  | No       | The conversation ID to continue an existing conversation. Omit or set to `null` to start a new one. |

**Example:**
```json
{
  "user_query": "How do I reset my password?",
  "enable_reasoning": true,
  "conversation_id": null
}
```

---

#### Session (Conversation) Management

- **Start a new conversation:**  
  Omit the `conversation_id` field or set it to `null`. The API will return a new `conversation_id` in the response.
- **Continue an existing conversation:**  
  Pass the `conversation_id` from a previous response. The backend will maintain the chat history and context for that session.

---

#### Response

Returns a JSON object with the following fields:

| Field           | Type    | Description                                                                                      |
|-----------------|---------|--------------------------------------------------------------------------------------------------|
| answer          | string  | The AI-generated answer to the user's query.                                                     |
| reasoning       | string  | (Optional) Step-by-step reasoning for the answer, if enabled.                                    |
| success         | boolean | Indicates if the operation was successful.                                                       |
| evaluations     | object  | (Optional) Additional evaluation metrics, such as similarity scores, if available.               |
| conversation_id | string  | The conversation ID for future requests (use this to continue the conversation).                 |

**Evaluations fields:**
- `mean_inter_document_similarity`: Average similarity between all retrieved context documents (higher = more similar to each other).
- `median_inter_document_similarity`: Median similarity between all retrieved context documents.
- `mean_document_query_similarity`: Average similarity between the user query and each retrieved document (higher = more relevant retrieval).
- `median_document_query_similarity`: Median similarity between the user query and each retrieved document.

**Example:**
```json
{
  "answer": "You can reset your password by clicking on 'Forgot Password' at the login page.",
  "reasoning": "The user asked about password reset. The answer is based on standard procedure.",
  "success": true,
  "evaluations": {
    "mean_document_query_similarity": 0.87,
    "median_document_query_similarity": 0.89,
    "mean_inter_document_similarity" : 0,
    "median_inter_document_similarity":0
  },
  "conversation_id": "b1a2c3d4-5678-90ab-cdef-1234567890ab"
}
```

---

#### Error Handling

- **400 Bad Request:**  
  The `user_query` field is missing or empty.
- **500 Internal Server Error:**  
  An unexpected error occurred (e.g., LLM or retrieval failure).

**Example:**
```json
{
  "detail": "User query cannot be empty."
}
```

