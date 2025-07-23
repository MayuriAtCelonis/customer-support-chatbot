def get_prompt_for_generation(chat_history, relevant_documents=None, summarised_query=None):
    """
    Constructs a prompt for LLM generation based on chat history, relevant documents, and a summarised query.

    Args:
        chat_history (list of dict): List of chat messages, each dict should have at least 'role' and 'content' keys.
        relevant_documents (list of str, optional): List of relevant documents to inform the response.
        summarised_query (str, optional): A summarised version of the user's query.

    Returns:
        str: The constructed prompt string.
    """
    prompt_parts = []

    # Add summarised query if available
    if summarised_query:
        prompt_parts.append(f"Summarised user query: {summarised_query}\n")

    # Add relevant documents if available
    if relevant_documents:
        docs_text = "\n".join([f"- {doc}" for doc in relevant_documents])
        prompt_parts.append(f"Relevant QnA to refer:\n{docs_text}\n")

    # Add chat history
    if chat_history:
        prompt_parts.append("Chat history:")
        for message in chat_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"{role.capitalize()}: {content}")
        prompt_parts.append("")  # Add a blank line
    base_prompt = (
        """Based on the above, provide a helpful, accurate, and concise answer to the user's query.\n"
        "Instructuons to follow strictly:\n"
        "1. You must ONLY use information from the relevant QnA and the past conversation above.\n"
        "2. If the answer is not present in the provided `QnA exmples or chat history, respond with: 'I'm sorry, I do not have enough information to answer that question.'\n"
        "3. Do NOT use any external knowledge or make up information.\n"
        """
    )
    prompt_parts.append(base_prompt)

    return "\n".join(prompt_parts)


def get_user_query_and_reasoning_tool_definition():
    """
    Returns a function/tool definition for LLM function calling to generate both the response to the user query and the reasoning for that response.
    """
    return [{
        "type": "function",
        "function": {
            "name": "generate_response_and_reasoning",
            "description": "Generates a response to the user's query and provides a step-by-step reasoning for the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "The answer or response to the user's query."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "A step-by-step explanation or reasoning for the answer provided. Do not disclose that past QnA were referred ."
                    }
                },
                "required": ["response", "reasoning"]
            }
        }
    }]


def handle_llm_response(response, enable_reasoning=True):
    """
    Handles the LLM response and extracts answer and reasoning if present.

    Args:
        response (dict): The response dict returned by generate_response_from_llm.
        enable_reasoning (bool): Whether to extract reasoning from the response.

    Returns:
        tuple: (answer, reasoning)
    """
    import json
    content = ""
    answer = ""
    reasoning = None
    print(f"{response=}")

    if response and response.get("type") == "message":
        content = response.get("content", "")
        # If content is empty, check for tool_calls in the raw message (OpenAI v1+)
        if not content and response.get("raw"):
            try:
                raw = response["raw"]
                # OpenAI v1+ tool call format
                choices = getattr(raw, "choices", None)
                if choices and hasattr(choices[0], "message"):
                    message = choices[0].message
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        for tool_call in tool_calls:
                            arguments = getattr(tool_call.function, "arguments", "")
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except Exception:
                                    arguments = {}
                            answer = arguments.get("response") or arguments.get("user_query") or ""
                            reasoning = arguments.get("reasoning", None)
                            break  # Only use the first tool call
            except Exception as e:
                print(f"Error parsing tool_calls: {e}")
        else:
            answer = content.strip() if content else ""
            if enable_reasoning and content:
                # Look for a "Reasoning:" or "Explanation:" section in the response
                if "Reasoning:" in content:
                    parts = content.split("Reasoning:", 1)
                    answer = parts[0].strip()
                    reasoning = parts[1].strip()
                elif "Explanation:" in content:
                    parts = content.split("Explanation:", 1)
                    answer = parts[0].strip()
                    reasoning = parts[1].strip()
                else:
                    answer = content.strip()
                    reasoning = None
            else:
                answer = content.strip()
                reasoning = None
    else:
        answer = ""
        reasoning = None

    return answer, reasoning
