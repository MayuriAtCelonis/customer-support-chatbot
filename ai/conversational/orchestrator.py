import traceback
from ai.retrieval.orchestrator import retrieve_releveant_context
from ai.generation.orchestrator import generate_response, summarise_query_from_chat_history


def process_chat_history(chat_history, enable_reasoning=False):
    """
    Processes chat history, retrieves relevant context, and generates a response.

    Args:
        chat_history (list): List of dicts with 'role' and 'content' keys.
        enable_reasoning (bool): Whether to enable reasoning in the response.

    Returns:
        dict: A response dict with answer, reasoning, and scores.
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
            "scores": None
        }

    # Retrieve relevant context (failsafe: returns [] if not possible)
    try:
        relevant_context , mean_distance , median_distance= retrieve_releveant_context(summarised_query, top_k=20)
    except Exception:
        relevant_context = []
    relevant_documents = [item["document"] for item in relevant_context] if relevant_context else None

    # Generate the response (failsafe: returns default if error)
    try:
        response = generate_response(
            chat_history=chat_history,
            relevant_documents=relevant_documents,
            summarised_query=summarised_query,
            enable_reasoning=enable_reasoning
        )
        response["evaluations"] = {
                "retrieved_examples_mean_distance": mean_distance,
                "retrieved_examples_median_distance": median_distance
            }
        response["success"] = True
    except Exception as e:
        response = {
            "answer": "An error occurred while generating the response!",
            "reasoning": f"An error occurred while generating the response: {str(e)}",
            "scores": None,
            "success": False
        }
        traceback.print_exc()

    return response
