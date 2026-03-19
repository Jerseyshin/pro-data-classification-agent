import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def openai_api_key():
    return os.getenv("OPENAI_API_KEY", "")


@pytest.fixture(scope="session")
def sample_categories():
    from classification_agent.config.default_categories import DEFAULT_DATA_CATEGORIES
    return DEFAULT_DATA_CATEGORIES
