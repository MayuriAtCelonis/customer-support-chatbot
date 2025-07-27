# Simple curl example:
# curl -X POST "http://localhost:8000/process_chat" \
#      -H "Content-Type: application/json" \
#      -d '{"user_query": "Hello, how are you?", "enable_reasoning": true}'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from backend.orchestrator import process_chat_history_api

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ProcessChatHistoryRequest(BaseModel):
    user_query: str
    enable_reasoning: Optional[bool] = True
    conversation_id: Optional[str] = None

class ProcessChatHistoryResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None
    scores: Optional[Any] = None
    success: Optional[bool] = None
    evaluations: Optional[Any] = None
    conversation_id: Optional[str] = None

@app.post("/process_chat", response_model=ProcessChatHistoryResponse)
async def process_chat(request: ProcessChatHistoryRequest):
    """
    Process a user query and manage conversation state.

    **Request Body:**
    - `user_query` (str, required): The user's message.
    - `enable_reasoning` (bool, optional): Whether to enable reasoning in the response (default: True).
    - `conversation_id` (str, optional): The conversation ID to continue, or None to start a new conversation.

    **Response:**
    - `answer` (str): The generated answer to the user query.
    - `reasoning` (str, optional): Step-by-step reasoning for the answer (if enabled).
    - `scores` (object, optional): Scoring or evaluation details, if available.
    - `success` (bool, optional): Indicates if the operation was successful.
    - `evaluations` (object, optional): Additional evaluation metrics, if available.
    - `conversation_id` (str, optional): The conversation ID for future requests.

    **Possible Errors:**
    - 400: User query cannot be empty.
    - 500: Internal server error (e.g., LLM or retrieval failure).
    """
    if not request.user_query or not request.user_query.strip():
        raise HTTPException(status_code=400, detail="User query cannot be empty.")

    try:
        operation_result, result, operation_error_message = process_chat_history_api(
            user_query=request.user_query,
            enable_reasoning=request.enable_reasoning,
            conversation_id=request.conversation_id
        )
        if operation_error_message:
            raise HTTPException(status_code=500, detail=operation_error_message)
        if not result:
            raise HTTPException(status_code=500, detail="No result returned from orchestrator.")

        # Compose response
        return ProcessChatHistoryResponse(
            answer=result.get("answer"),
            reasoning=result.get("reasoning"),
            scores=result.get("scores"),
            success=True,
            evaluations=result.get("evaluations"),
            conversation_id=result.get("conversation_id")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")