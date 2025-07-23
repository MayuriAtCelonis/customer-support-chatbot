from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from ai.conversational.orchestrator import process_chat_history as process_chat_history_core

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str
    # Optionally, you can add timestamp or other fields if needed
    timestamp: Optional[str] = None

class ProcessChatHistoryRequest(BaseModel):
    chat_history: List[ChatMessage]
    enable_reasoning: Optional[bool] = False

class ProcessChatHistoryResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    scores: Optional[Any] = None
    success: Optional[bool] = None
    evaluations: Optional[Any] = None

def convert_numpy_types(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    else:
        return obj

@app.post("/process_chat", response_model=ProcessChatHistoryResponse)
async def process_chat_history_api(request: ProcessChatHistoryRequest):
    """
    Process a list of chat messages and return an answer, reasoning, and scores.

    **Request Body:**
    - `chat_history` (List[ChatMessage], required): List of chat messages, each with 'role' and 'content'.
    - `enable_reasoning` (bool, optional): Whether to enable reasoning in the response (default: False).

    **Response:**
    - `answer` (str): The generated answer to the chat history.
    - `reasoning` (str, optional): Step-by-step reasoning for the answer (if enabled).
    - `scores` (object, optional): Scoring or evaluation details, if available.
    - `success` (bool, optional): Indicates if the operation was successful.
    - `evaluations` (object, optional): Additional evaluation metrics, if available.

    **Example Request:**
    ```json
    {
      "chat_history": [
        {"role": "user", "content": "How do I reset my password?"},
        {"role": "assistant", "content": "You can reset it via the settings page."}
      ],
      "enable_reasoning": true
    }
    ```

    **Example Response:**
    ```json
    {
      "answer": "You can reset your password by clicking on 'Forgot Password' at the login page.",
      "reasoning": "The user asked about password reset. The answer is based on standard procedure.",
      "scores": null,
      "success": true,
      "evaluations": {"retrieved_examples_mean_distance": 0.12}
    }
    ```

    **Possible Errors:**
    - 400: Chat history cannot be empty.
    - 500: Internal server error (e.g., LLM or retrieval failure).
    """
    if not request.chat_history:
        raise HTTPException(status_code=400, detail="Chat history cannot be empty.")
    # Convert to list of dicts for core function
    try:
        chat_history_dicts = [msg.dict(exclude_none=True) for msg in request.chat_history]
        result = process_chat_history_core(chat_history_dicts, enable_reasoning=request.enable_reasoning)
        result = convert_numpy_types(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")