"""A server with fake models whose LLM never answers, for the shutdown test.

    python -m uvicorn tests.serve_fakes:create_app --factory
"""

from app.main import create_app as build_app
from app.startup import Services
from tests.conftest import make_settings
from tests.fakes import FakeLLM, FakeSTT, FakeTTS


def create_app():
    services = Services(FakeSTT(), FakeLLM(first_token_delay=60), FakeTTS())
    return build_app(make_settings(log_level="INFO"), services)
