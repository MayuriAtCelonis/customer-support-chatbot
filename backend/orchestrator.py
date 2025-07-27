import logging
from typing import Tuple, Optional, Dict, Any
from backend.conversation_management import (
    get_conversation_history,
    set_new_conversation_history,
    append_conversation_history,
    validate_chat_history,
)
from ai.conversational.orchestrator import process_chat_history as process_chat_history_core

logger = logging.getLogger(__name__)

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def validate_user_query(user_query: str) -> Tuple[bool, Optional[str]]:
    """Validate the user query format and content."""
    if not isinstance(user_query, str):
        return False, "User query must be a string"
    if not user_query.strip():
        return False, "User query cannot be empty"
    if len(user_query.strip()) > 10000:  # Reasonable limit
        return False, "User query is too long (max 10000 characters)"
    return True, None

def process_chat_history_api(
    user_query: str, 
    enable_reasoning: bool = True, 
    conversation_id: Optional[str] = None
) -> Tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Handles chat history processing and conversation management.

    Args:
        user_query: The user's message
        enable_reasoning: Whether to enable reasoning in the response
        conversation_id: Optional conversation ID to continue an existing conversation

    Returns:
        Tuple of (operation_result, result, operation_error_message)
        - operation_result: The result of any conversation management operation
        - result: The result from process_chat_history_core (LLM response etc.), or None if error
        - operation_error_message: Any error message encountered, or None
    """
    operation_result = None
    operation_error_message = None
    chat_history = []
    result = None

    # Validate user query
    is_valid, validation_error = validate_user_query(user_query)
    if not is_valid:
        operation_error_message = validation_error
        logger.error(f"Invalid user query: {validation_error}")
        return operation_result, result, operation_error_message

    # Clean the user query
    user_query = user_query.strip()

    try:
        if conversation_id:
            logger.info(f"Retrieving conversation history for conversation_id: {conversation_id}")
            success, chat_history, error_message = get_conversation_history(conversation_id)
            if not success:
                operation_error_message = error_message or "Failed to retrieve conversation history."
                logger.error(f"Failed to retrieve conversation history: {operation_error_message}")
                return operation_result, result, operation_error_message
            
            # Validate existing chat history
            if not validate_chat_history(chat_history):
                operation_error_message = "Invalid chat history format retrieved from database"
                logger.error(f"Invalid chat history format for conversation {conversation_id}")
                return operation_result, result, operation_error_message

            new_chat = [{"role": "user", "content": user_query}]
            logger.info(f"Appending new user query to conversation_id {conversation_id}: {user_query[:50]}...")
            
            append_success, op_result, append_error = append_conversation_history(
                conversation_id, new_chat
            )
            operation_result = op_result
            if not append_success:
                operation_error_message = append_error or "Failed to append to conversation history."
                logger.error(f"Failed to append to conversation history: {operation_error_message}")
                return operation_result, result, operation_error_message
            
            chat_history = chat_history + new_chat
        else:
            logger.info("Creating new conversation history.")
            success, data, error_message = set_new_conversation_history(user_query)
            if not success or not data:
                operation_error_message = error_message or "Failed to create new conversation."
                logger.error(f"Failed to create new conversation: {operation_error_message}")
                return operation_result, result, operation_error_message
            
            conversation_id, chat_history = data
            logger.info(f"New conversation created with conversation_id: {conversation_id}")

        # Process the chat history
        logger.info(f"Processing chat history for conversation_id: {conversation_id}")
        result = process_chat_history_core(chat_history, enable_reasoning=enable_reasoning)
        
        if not result or not isinstance(result, dict):
            operation_error_message = "Invalid response from chat processing"
            logger.error(f"Invalid response from process_chat_history_core for conversation {conversation_id}")
            return operation_result, result, operation_error_message
        
        # Convert numpy types for JSON serialization
        result = convert_numpy_types(result)
        result["conversation_id"] = conversation_id
        
        # Prepare assistant response for storage
        new_content = [{
            "role": "assistant",
            "content": result.get("answer", ""),
            "summarized_query": result.get("summarised_query", ""),
            "reference_documents": result.get("relevant_documents", []),
            "evaluations": result.get("evaluations", {}),
            "reasoning": result.get("reasoning", "")
        }]
        
        # Validate the assistant response before storing
        if not validate_chat_history(new_content):
            operation_error_message = "Invalid assistant response format"
            logger.error(f"Invalid assistant response format for conversation {conversation_id}")
            return operation_result, result, operation_error_message
        
        logger.info(f"Appending assistant response to conversation_id {conversation_id}")
        append_success, op_result, append_error = append_conversation_history(
            conversation_id, new_content
        )
        
        if not append_success:
            operation_error_message = append_error or "Failed to append assistant response to conversation history."
            logger.error(f"Failed to append assistant response: {operation_error_message}")
            # Don't return here - the main processing was successful, just logging failed
            # The user still gets their response
        
        logger.info(f"Successfully processed conversation {conversation_id}")
        
    except Exception as e:
        operation_error_message = f"Unexpected error during chat processing: {str(e)}"
        logger.exception(f"Exception occurred during chat history processing: {operation_error_message}")

    return operation_result, result, operation_error_message
