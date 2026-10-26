import pytest


@pytest.fixture(autouse=True)
def offline_aws(monkeypatch):
    """Dummy credentials and region: botocore can build clients, but nothing reaches AWS."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("SAGEMAKER_ENDPOINT_NAME", "demand-forecast")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
