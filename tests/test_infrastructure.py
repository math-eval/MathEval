"""
Infrastructure validation tests.

These tests verify that the testing infrastructure is properly set up
and all fixtures work as expected.
"""

import pytest
import json
from pathlib import Path


class TestInfrastructureSetup:
    """Test the basic testing infrastructure setup."""
    
    def test_pytest_is_working(self):
        """Verify pytest is properly installed and working."""
        assert True
    
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates a valid directory."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can create files in it
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    def test_temp_file_fixture(self, temp_file):
        """Test that temp_file fixture creates a valid file."""
        assert temp_file.exists()
        assert temp_file.is_file()
        assert temp_file.read_text() == "test content"
    
    def test_mock_json_file_fixture(self, mock_json_file):
        """Test that mock_json_file fixture creates valid JSON."""
        json_file, expected_data = mock_json_file
        assert json_file.exists()
        
        with open(json_file, 'r') as f:
            actual_data = json.load(f)
        
        assert actual_data == expected_data
        assert actual_data["test_key"] == "test_value"
        assert actual_data["nested"]["key"] == "value"
        assert actual_data["list"] == [1, 2, 3]
    
    def test_sample_config_fixture(self, sample_config):
        """Test that sample_config fixture provides expected structure."""
        required_keys = ["model_name", "batch_size", "learning_rate", "epochs", "data_path"]
        for key in required_keys:
            assert key in sample_config
        
        assert sample_config["batch_size"] == 32
        assert sample_config["learning_rate"] == 0.001
    
    def test_mock_subprocess_fixture(self, mock_subprocess):
        """Test that mock_subprocess fixture works correctly."""
        # Test the mock returns expected values
        result = mock_subprocess.run(["echo", "test"])
        assert result.returncode == 0
        assert result.stdout == "success"
        assert result.stderr == ""
    
    def test_mock_requests_fixture(self, mock_requests):
        """Test that mock_requests fixture works correctly."""
        # Test GET request
        response = mock_requests.get("http://example.com")
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        
        # Test POST request
        response = mock_requests.post("http://example.com", data={"key": "value"})
        assert response.status_code == 200
        assert response.text == "success response"
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
    
    def test_env_vars_fixture(self, env_vars):
        """Test environment variable management fixture."""
        # Set some environment variables
        env_vars.set(TEST_VAR="test_value", ANOTHER_VAR="another_value")
        
        import os
        assert os.environ.get("TEST_VAR") == "test_value"
        assert os.environ.get("ANOTHER_VAR") == "another_value"
        
        # Clear specific variables
        env_vars.clear("TEST_VAR")
        assert os.environ.get("TEST_VAR") is None
        assert os.environ.get("ANOTHER_VAR") == "another_value"


class TestPytest:
    """Test pytest functionality and configuration."""
    
    def test_pytest_markers_defined(self):
        """Test that custom markers are properly defined."""
        # This test will fail if markers aren't defined in pyproject.toml
        # and --strict-markers is enabled
        pass
    
    def test_coverage_is_configured(self):
        """Test that coverage reporting is configured."""
        # This is a placeholder - actual coverage testing happens
        # when pytest is run with coverage flags
        assert True


class TestOptionalDependencies:
    """Test optional dependencies that might be used in the project."""
    
    def test_pandas_fixture_available(self, sample_dataframe):
        """Test pandas fixture if pandas is available."""
        try:
            import pandas as pd
            assert len(sample_dataframe) == 5
            assert list(sample_dataframe.columns) == ['id', 'name', 'score']
            assert sample_dataframe['name'].iloc[0] == 'Alice'
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_can_import_json(self):
        """Test that json module is available (should always pass)."""
        import json
        assert hasattr(json, 'loads')
        assert hasattr(json, 'dumps')
    
    def test_pathlib_available(self):
        """Test that pathlib is available (should always pass in Python 3.4+)."""
        from pathlib import Path
        assert Path(".").exists()