def generate_response(chat_history, relevant_documents=None, summarised_query=None, enable_reasoning=True):
    """
    Accepts chat history, relevant documents, a summarised query, and an enable_reasoning flag. Returns a dictionary with answer, reasoning (optional), and scores.

    Args:
        chat_history (list of dict): List of chat messages, each dict should have at least 'role' and 'content' keys.
        relevant_documents (list of str, optional): List of relevant documents to inform the response.
        summarised_query (str, optional): A summarised version of the user's query.
        enable_reasoning (bool, optional): Whether to include reasoning in the response. Defaults to True.

    Returns:
        dict: {
            "answer": <str>,
            "reasoning": <str, optional>,
            "scores": <dict>
        }
    """
    if not chat_history or not isinstance(chat_history, list):
        response = {
            "answer": "",
            "reasoning": "No chat history provided.",
            "scores": None,
            "relevant_documents": relevant_documents,
            "summarised_query": summarised_query
        }
        if not enable_reasoning:
            response.pop("reasoning", None)
        return response
    
    


def summarise_query_from_chat_history(chat_history):
    """
    Summarises the user's query from the chat history.

    Args:
        chat_history (list of dict): List of chat messages, each dict should have at least 'role' and 'content' keys.

    Returns:
        str: A summarised query string (currently just the last user message).
    """
    if not chat_history or not isinstance(chat_history, list):
        return ""
    for message in reversed(chat_history):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""
