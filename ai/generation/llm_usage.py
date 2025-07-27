from typing import Any, Dict, List, Optional, Union
import os

_LLM_CLIENTS = {}

def get_llm_client(provider: str, api_key: Optional[str] = None, **kwargs) -> Any:
    """
    Create or get an LLM client for the given provider.
    Supported providers: 'openai', 'groq'
    """
    provider = provider.lower()
    if provider in _LLM_CLIENTS:
        return _LLM_CLIENTS[provider]

    if provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider.")
        if api_key:
            openai.api_key = api_key
        client = openai
    elif provider == "groq":
        try:
            import groq
        except ImportError:
            raise ImportError("groq package is required for Groq provider.")
        if not api_key:
            raise ValueError("api_key is required for Groq provider.")
        client = groq.Client(api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    _LLM_CLIENTS[provider] = client
    return client

def generate_response_from_llm(
    provider: str,
    prompt: Optional[str] = None,
    api_key: Optional[str] = os.getenv('OPENAI_API_KEY', ''),
    # api_key: Optional[str] = '',
    model: Optional[str] = None,
    messages: Optional[List[Dict]] = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    tools: Optional[List[Dict]] = None,
    functions: Optional[List[Dict]] = None,  # for backward compatibility
    function_call: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a response from the LLM, supporting multiple providers.

    Args:
        provider: The LLM provider name ('openai', 'groq', etc.)
        prompt: The input prompt string (for completion endpoints).
        api_key: API key for the provider.
        model: Model name to use (optional).
        messages: List of chat messages (for chat endpoints).
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        tools: List of function/tool definitions for function calling (OpenAI/groq).
        functions: (Deprecated) Alias for tools.
        function_call: Function call control (OpenAI/groq).
        kwargs: Additional parameters for the provider's API.

    Returns:
        Dict with the following keys:
            - "type": "message" or "function_call"
            - "content": The message content (if type == "message")
            - "function_call": Dict with function call details (if type == "function_call")
            - "raw": The raw response object from the provider

    Response format details:
        - If the LLM returns a normal message, the return dict will be:
            {
                "type": "message",
                "content": <string>,
                "raw": <raw response object>
            }
        - If the LLM returns a function call (function/tool call), the return dict will be:
            {
                "type": "function_call",
                "function_call": {
                    "name": <function name>,
                    "arguments": <arguments as string or dict>
                },
                "raw": <raw response object>
            }
        - The "raw" key always contains the full provider response for further inspection.

    Notes:
        - For OpenAI, function calling is supported via the 'tools' or 'functions' parameter.
        - For Groq, function calling is supported if the client and model support it.
        - If neither a message nor a function call is found, "content" will be an empty string.
    """
    print(f"api_key: {api_key} {os.getenv('OPENAI_API_KEY','not found')}")  # Print the api_key as requested
    client = get_llm_client(provider, api_key=api_key)

    if provider.lower() == "openai":
        # Prefer ChatCompletion with function calling if tools/functions provided
        chat_completion = client.chat.completions
        if chat_completion is None :
            chat_completion = client.chat.completions
        if chat_completion:
            # Prepare messages
            msgs = messages
            if not msgs:
                if prompt:
                    msgs = [{"role": "user", "content": prompt}]
                else:
                    raise ValueError("Either messages or prompt must be provided.")
            # Prefer 'tools', fallback to 'functions' for backward compatibility
            tools_param = tools if tools is not None else functions
            create_kwargs = {
                "model": model or "gpt-3.5-turbo",
                "messages": msgs,
                "temperature": temperature,
                "top_p": top_p,
            }
            if tools_param:
                create_kwargs["tools"] = tools_param
            if function_call:
                create_kwargs["function_call"] = function_call
            # Accept extra kwargs
            create_kwargs.update({k: v for k, v in kwargs.items() if k not in create_kwargs})
            response = chat_completion.create(**create_kwargs)
            msg = response.choices[0].message
            if hasattr(msg, "function_call") and msg.function_call:
                # OpenAI returns function_call as an object with name and arguments
                return {
                    "type": "function_call",
                    "function_call": {
                        "name": getattr(msg.function_call, "name", None),
                        "arguments": getattr(msg.function_call, "arguments", None)
                    },
                    "raw": response
                }
            return {
                "type": "message",
                "content": msg.content.strip() if msg.content else "",
                "raw": response
            }
        else:
            # OpenAI Completion API (legacy)
            completion = client.chat.completions.create(
                model=model or "text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            return {
                "type": "message",
                "content": completion.choices[0].text.strip(),
                "raw": completion
            }
    elif provider.lower() == "groq":
        # Try to use function calling if supported
        # Assume groq client has a .chat method with tools/function_call support
        if hasattr(client, "chat"):
            msgs = messages
            if not msgs:
                if prompt:
                    msgs = [{"role": "user", "content": prompt}]
                else:
                    raise ValueError("Either messages or prompt must be provided.")
            chat_kwargs = {
                "model": model or "mixtral-8x7b-32768",
                "messages": msgs,
                "temperature": temperature,
                "top_p": top_p,
            }
            # Try to pass tools/function_call if supported
            if tools:
                chat_kwargs["tools"] = tools
            if function_call:
                chat_kwargs["function_call"] = function_call
            chat_kwargs.update({k: v for k, v in kwargs.items() if k not in chat_kwargs})
            response = client.chat(**chat_kwargs)
            msg = response.choices[0].message
            if hasattr(msg, "function_call") and msg.function_call:
                return {
                    "type": "function_call",
                    "function_call": {
                        "name": getattr(msg.function_call, "name", None),
                        "arguments": getattr(msg.function_call, "arguments", None)
                    },
                    "raw": response
                }
            return {
                "type": "message",
                "content": msg.content.strip() if msg.content else "",
                "raw": response
            }
        elif hasattr(client, "complete"):
            # Fallback to completion endpoint
            completion = client.complete(
                model=model or "mixtral-8x7b-32768",
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            return {
                "type": "message",
                "content": completion.choices[0].text.strip(),
                "raw": completion
            }
        else:
            raise NotImplementedError("Groq client does not support chat or complete methods.")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

