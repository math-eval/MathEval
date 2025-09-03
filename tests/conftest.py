"""
Shared pytest fixtures for the testing infrastructure.

This module contains common fixtures that can be used across all test files.
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import json


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    temp_file_path = temp_dir / "test_file.txt"
    temp_file_path.write_text("test content")
    return temp_file_path


@pytest.fixture
def mock_json_file(temp_dir):
    """Create a mock JSON file for testing."""
    json_data = {
        "test_key": "test_value",
        "nested": {"key": "value"},
        "list": [1, 2, 3]
    }
    json_file = temp_dir / "test.json"
    json_file.write_text(json.dumps(json_data))
    return json_file, json_data


@pytest.fixture
def sample_config():
    """Provide sample configuration data for tests."""
    return {
        "model_name": "test_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "data_path": "/path/to/data"
    }


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing shell commands."""
    mock = Mock()
    mock.run.return_value.returncode = 0
    mock.run.return_value.stdout = "success"
    mock.run.return_value.stderr = ""
    return mock


@pytest.fixture
def mock_requests():
    """Mock requests library for API testing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "success"}
    mock_response.text = "success response"
    
    mock_requests = Mock()
    mock_requests.get.return_value = mock_response
    mock_requests.post.return_value = mock_response
    
    return mock_requests


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing."""
    try:
        import pandas as pd
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'score': [85.5, 92.0, 78.5, 95.5, 88.0]
        })
    except ImportError:
        pytest.skip("pandas not available")


@pytest.fixture
def mock_tqdm():
    """Mock tqdm progress bar for testing."""
    def mock_tqdm_func(iterable, *args, **kwargs):
        return iterable
    return mock_tqdm_func


@pytest.fixture
def env_vars():
    """Fixture to manage environment variables during tests."""
    original_env = dict(os.environ)
    
    def set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    def clear_env(*keys):
        for key in keys:
            if key in os.environ:
                del os.environ[key]
    
    class EnvManager:
        set = staticmethod(set_env)
        clear = staticmethod(clear_env)
    
    yield EnvManager()
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def capture_output(capsys):
    """Enhanced output capture with helper methods."""
    class OutputCapture:
        def __init__(self, capsys):
            self._capsys = capsys
        
        def get_stdout(self):
            captured = self._capsys.readouterr()
            return captured.out
        
        def get_stderr(self):
            captured = self._capsys.readouterr()
            return captured.err
        
        def get_both(self):
            captured = self._capsys.readouterr()
            return captured.out, captured.err
    
    return OutputCapture(capsys)