import pytest
from unittest.mock import MagicMock, patch
import ai.retrieval.embeddings_helper as vdbc

class TestVectorDBConnect:
    def test_get_sentence_transformer_caches_model(self):
        with patch('ai.retrieval.embeddings_helper.SentenceTransformer') as MockModel:
            MockModel.return_value = 'mock_model'
            model1 = vdbc.get_sentence_transformer('mock-model')
            model2 = vdbc.get_sentence_transformer('mock-model')
            assert model1 == 'mock_model'
            assert model2 == 'mock_model'
            assert MockModel.call_count == 1  # Only constructed once

    def test_initialize_qdrant_creates_collection(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        result = vdbc.initialize_qdrant('test_collection', 128, client=mock_client)
        mock_client.collection_exists.assert_called_once_with('test_collection')
        mock_client.delete_collection.assert_called_once_with('test_collection')
        mock_client.create_collection.assert_called_once()
        assert result == mock_client

    def test_get_qdrant_client_caches_client(self):
        with patch('ai.retrieval.embeddings_helper.SentenceTransformer') as MockModel, \
            patch('ai.retrieval.embeddings_helper.initialize_qdrant') as mock_init_qdrant:
            mock_model_instance = MagicMock()
            mock_model_instance.get_sentence_embedding_dimension.return_value = 42
            MockModel.return_value = mock_model_instance
            mock_init_qdrant.return_value = 'mock_client'
            # First call creates and caches
            client1 = vdbc.get_qdrant_client('test_collection', 'mock-model')
            # Second call returns cached
            client2 = vdbc.get_qdrant_client('test_collection', 'mock-model')
            assert client1 == 'mock_client'
            assert client2 == 'mock_client'
            assert mock_init_qdrant.call_count == 1

    def test_init_embeddings_helperions_returns_client_and_model(self):
        with patch('ai.retrieval.embeddings_helper.get_qdrant_client') as mock_get_client, \
             patch('ai.retrieval.embeddings_helper.get_sentence_transformer') as mock_get_model:
            mock_get_client.return_value = 'mock_client'
            mock_get_model.return_value = 'mock_model'
            client, model = vdbc.init_embeddings_helperions('test_collection', 'mock-model')
            assert client == 'mock_client'
            assert model == 'mock_model'
            mock_get_client.assert_called_once_with('test_collection', 'mock-model')
            mock_get_model.assert_called_once_with('mock-model') 