import pytest
from ai.conversational.orchestrator import process_chat_history

class TestChatbotIntegration:
    def test_full_chat_flow(self, monkeypatch, sample_chat_history, mock_document):
        # Mock summarise_query_from_chat_history to return a query
        monkeypatch.setattr(
            'ai.conversational.orchestrator.summarise_query_from_chat_history',
            lambda chat_history: 'What is the weather today?'
        )
        # Mock retrieve_releveant_context to return a document
        monkeypatch.setattr(
            'ai.conversational.orchestrator.retrieve_releveant_context',
            lambda query: [mock_document]
        )
        # Mock generate_response to return a final answer
        monkeypatch.setattr(
            'ai.conversational.orchestrator.generate_response',
            lambda chat_history, relevant_documents, summarised_query, enable_reasoning: {
                'answer': 'The weather today is sunny.',
                'reasoning': 'Based on the latest weather report.',
                'scores': {'relevance': 0.95}
            }
        )
        result = process_chat_history(sample_chat_history)
        assert result['answer'] == 'The weather today is sunny.'
        assert 'reasoning' in result
        assert result['scores']['relevance'] > 0.9 