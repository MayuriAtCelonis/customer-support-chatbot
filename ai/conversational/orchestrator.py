import traceback
from typing import List, Dict, Any, Optional
from ai.retrieval.orchestrator import retrieve_releveant_context
from ai.generation.orchestrator import generate_response, summarise_query_from_chat_history


def process_chat_history(
    chat_history: List[Dict[str, str]], 
    enable_reasoning: bool = False
) -> Dict[str, Any]:
    """
    Processes chat history, retrieves relevant context, and generates a response.

    Args:
        chat_history (List[Dict[str, str]]): List of messages, each a dict with 'role' and 'content' keys.
        enable_reasoning (bool, optional): Whether to enable reasoning in the response. Defaults to False.

    Returns:
        Dict[str, Any]: A response dictionary containing:
            - answer (str): The generated answer or an error message.
            - reasoning (str): Reasoning or error details.
            - relevant_documents (Optional[List[str]]): List of relevant documents, if any.
            - summarised_query (str): The summarised query from chat history.
            - evaluations (Optional[Dict[str, float]]): Similarity evaluation metrics, if available.
            - success (bool): Whether the response was successfully generated.
    """
    if not isinstance(chat_history, list):
        raise ValueError("chat_history must be a list of dicts")
    for message in chat_history:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise ValueError("Each message must be a dict with 'role' and 'content' keys")

    # Summarise the query from chat history (failsafe: returns "" if not possible)
    summarised_query = summarise_query_from_chat_history(chat_history)
    if not summarised_query:
        return {
            "answer": "",
            "reasoning": "No user message found in chat history.",
            "relevant_documents": None,
            "summarised_query": "",
            "evaluations": None,
            "success": False
        }

    # Retrieve relevant context (failsafe: returns [] if not possible)
    try:
        (
            relevant_context, 
            mean_inter_document_similarity, 
            median_inter_document_similarity, 
            mean_document_query_similarity, 
            median_document_query_similarity
        ) = retrieve_releveant_context(summarised_query, top_k=20)
    except Exception:
        relevant_context = []
        mean_inter_document_similarity = None
        median_inter_document_similarity = None
        mean_document_query_similarity = None
        median_document_query_similarity = None

    relevant_documents = [item["document"] for item in relevant_context] if relevant_context else None

    # Generate the response (failsafe: returns default if error)
    try:
        response = generate_response(
            chat_history=chat_history,
            relevant_documents=relevant_documents,
            summarised_query=summarised_query,
            enable_reasoning=enable_reasoning
        )
        response['relevant_documents'] = relevant_documents
        response['summarised_query'] = summarised_query
        response["evaluations"] = {
            "mean_inter_document_similarity": mean_inter_document_similarity,
            "median_inter_document_similarity": median_inter_document_similarity,
            "mean_document_query_similarity": mean_document_query_similarity,
            "median_document_query_similarity": median_document_query_similarity
        }
        response["success"] = True
    except Exception as e:
        response = {
            "answer": "An error occurred while generating the response!",
            "reasoning": f"An error occurred while generating the response: {str(e)}",
            "relevant_documents": relevant_documents,
            "summarised_query": summarised_query,
            "evaluations": {
                "mean_inter_document_similarity": mean_inter_document_similarity,
                "median_inter_document_similarity": median_inter_document_similarity,
                "mean_document_query_similarity": mean_document_query_similarity,
                "median_document_query_similarity": median_document_query_similarity
            },
            "success": False
        }
        traceback.print_exc()

    return response
