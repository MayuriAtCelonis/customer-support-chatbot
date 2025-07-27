import os
from ai.generation.llm_usage import generate_response_from_llm

def get_prompt_for_query_summarization(chat_history):
    """
    Constructs a prompt for the LLM to summarize the user's query as crisply as possible.
    """
    prompt = """
    You are a helpful assistant that summarizes the user's latest query as crisply and concisely as possible. 
    Point of view is from the user's perspective.
    
    Example Chat history:
    USER: I'm having trouble logging in.
    ASSISTANT: Could you please describe the issue. 
    USER: I see 'unable to login' in the error message.
    Summarized query: While logging in, I am seeing 'unable to login' in the error message.
    """
    prompt_parts = []
    prompt_parts.append(prompt)
    prompt_parts.append("Chat history:")
    for message in chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        prompt_parts.append(f"{role.capitalize()}: {content}")
    prompt_parts.append("")
    prompt_parts.append("Summarize the user's latest query as crisp and detailed .")
    return "\n".join(prompt_parts)

def summarise_query_from_chat_history(chat_history):
    """
    Uses the LLM to summarize the user's query from the chat history as crisply as possible.

    Args:
        chat_history (list of dict): List of chat messages, each dict should have at least 'role' and 'content' keys.

    Returns:
        str: A crisply summarized query string.
    """
    if not chat_history or not isinstance(chat_history, list):
        return ""
    prompt = get_prompt_for_query_summarization(chat_history)
    llm_response = generate_response_from_llm(
        provider="openai",
        prompt=prompt,
        api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4.1" #TODO: model and its provider details ideally must come from the env
    )
    # Try to extract the summary from the LLM response
    if isinstance(llm_response, dict):
        content = llm_response.get("content", "")
        if content:
            return content.strip()
    elif isinstance(llm_response, str):
        return llm_response.strip()
    return ""
