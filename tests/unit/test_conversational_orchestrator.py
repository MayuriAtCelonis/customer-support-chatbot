import pytest
from ai.conversational.orchestrator import process_chat_history

class TestProcessChatHistory:
    def test_basic_success(self, monkeypatch):
        monkeypatch.setattr(
            'ai.conversational.orchestrator.summarise_query_from_chat_history',
            lambda chat_history: 'What is your name?'
        )
        monkeypatch.setattr(
            'ai.conversational.orchestrator.retrieve_releveant_context',
            lambda query: [{'document': 'My name is Chatbot.'}]
        )
        monkeypatch.setattr(
            'ai.conversational.orchestrator.generate_response',
            lambda chat_history, relevant_documents, summarised_query, enable_reasoning: {
                'answer': 'My name is Chatbot.',
                'reasoning': 'Based on the document.',
                'scores': {'relevance': 1.0}
            }
        )
        chat_history = [
            {'role': 'user', 'content': 'What is your name?'}
        ]
        result = process_chat_history(chat_history)
        assert result['answer'] == 'My name is Chatbot.'
        assert 'reasoning' in result
        assert 'scores' in result

    def test_invalid_input_type(self):
        with pytest.raises(ValueError):
            process_chat_history('not a list')
        with pytest.raises(ValueError):
            process_chat_history([{'role': 'user'}])  # missing 'content'
        with pytest.raises(ValueError):
            process_chat_history([{'content': 'Hello'}])  # missing 'role'

    def test_no_user_message(self, monkeypatch):
        monkeypatch.setattr(
            'ai.conversational.orchestrator.summarise_query_from_chat_history',
            lambda chat_history: ''
        )
        chat_history = [
            {'role': 'system', 'content': 'Welcome!'}
        ]
        result = process_chat_history(chat_history)
        assert result['answer'] == ''
        assert 'No user message' in result['reasoning'] 