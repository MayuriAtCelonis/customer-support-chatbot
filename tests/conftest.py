import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest

@pytest.fixture
def sample_chat_history():
    return [
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi! How can I help you?'}
    ]

@pytest.fixture
def mock_document():
    return {'document': 'This is a mock document.'} 