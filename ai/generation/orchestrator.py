from ai.generation.llm_usage import generate_response_from_llm 
from ai.generation.query_response_generation import get_prompt_for_generation , get_user_query_and_reasoning_tool_definition , handle_llm_response
import os

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

    # Compose the prompt for the LLM
    prompt = get_prompt_for_generation(chat_history, relevant_documents, summarised_query)
   
    # Compose the reasoning instruction if enabled
    if enable_reasoning:
        prompt += "\n\nAlso, provide a step-by-step explanation of your reasoning for the answer."

    print(f"{prompt=}")
    # Call the LLM to generate the response
    llm_response = generate_response_from_llm(
        provider="openai",
        prompt=prompt,
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4.1",
        # function_call=get_user_query_and_reasoning_tool_definition(),
        tools=get_user_query_and_reasoning_tool_definition()
    )



    # Parse the LLM response to extract answer and reasoning
    answer , reasoning = handle_llm_response(llm_response)
    print(f"{answer=} {reasoning=}")
    # Try to split answer and reasoning if possible

    response = {
        "answer": answer,
        "scores": None,
        "relevant_documents": relevant_documents,
        "summarised_query": summarised_query
    }
    if enable_reasoning:
        response["reasoning"] = reasoning

    return response

def summarise_query_from_chat_history(chat_history):
    """
    Summarises the user's query from the chat history.

    Args:
        chat_history (list of dict): List of chat messages, each dict should have at least 'role' and 'content' keys.

    Returns:
        str: A summarised query string.
    """
    if not chat_history or not isinstance(chat_history, list):
        return ""
    if len(chat_history) > 2:
        from ai.generation.summarized_query_generation import summarise_query_from_chat_history as summarise_query_llm
        return summarise_query_llm(chat_history)
    for message in reversed(chat_history):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""
