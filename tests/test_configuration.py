import pytest

from app_config import ConfigurationError, validate_environment


REQUIRED_KEYS = ("OPENAI_API_KEY", "SERPER_API_KEY")


def clear_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_validate_environment_reports_all_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_required_keys(monkeypatch)

    with pytest.raises(ConfigurationError) as error:
        validate_environment()

    message = str(error.value)
    assert "OPENAI_API_KEY" in message
    assert "SERPER_API_KEY" in message


def test_validate_environment_accepts_required_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_required_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")

    validate_environment()
