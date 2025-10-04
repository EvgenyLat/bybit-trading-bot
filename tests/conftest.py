"""
Pytest configuration for trading bot tests
"""

import pytest
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

@pytest.fixture
def test_data_dir():
    """Fixture for test data directory"""
    return Path(__file__).parent / 'data'

@pytest.fixture
def mock_config():
    """Mock configuration for tests"""
    return {
        'api_key': 'test_key',
        'api_secret': 'test_secret',
        'testnet': True,
        'risk_per_trade': 0.02,
        'max_daily_loss': 0.05
    }
