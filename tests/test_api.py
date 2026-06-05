import pytest

from open_deep_think import api


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return {"ok": True}


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = _FakeChat(completions)


def test_single_turn_api_call_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key must be provided"):
        api.single_turn_api_call(model="m", prompt="p", max_tokens=10)


def test_single_turn_api_call_passes_default_sampling_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("API_BASE_URL", "https://example.invalid/v1/")
    completions = _FakeCompletions()
    captured_client_kwargs: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> _FakeClient:
        captured_client_kwargs.update(kwargs)
        return _FakeClient(completions)

    monkeypatch.setattr(api, "OpenAI", _fake_openai)

    result = api.single_turn_api_call(model="test-model", prompt="hello", max_tokens=32)

    assert result == {"ok": True}
    assert captured_client_kwargs["api_key"] == "test-key"
    assert captured_client_kwargs["base_url"] == "https://example.invalid/v1"
    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32,
            "temperature": None,
            "top_p": None,
            "extra_body": {"reasoning_effort": "high"},
        },
    ]


def test_single_turn_api_call_passes_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_temperature = 0.2
    expected_top_p = 0.7

    monkeypatch.setenv("API_KEY", "test-key")
    completions = _FakeCompletions()

    def _fake_openai(**_: object) -> _FakeClient:
        return _FakeClient(completions)

    monkeypatch.setattr(api, "OpenAI", _fake_openai)

    api.single_turn_api_call(
        model="test-model",
        prompt="hello",
        max_tokens=32,
        temperature=expected_temperature,
        top_p=expected_top_p,
    )

    assert completions.calls[0]["temperature"] == expected_temperature
    assert completions.calls[0]["top_p"] == expected_top_p


def test_chat_api_call_passes_messages_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("API_BASE_URL", "https://example.invalid/v1")
    completions = _FakeCompletions()
    captured_client_kwargs: dict[str, object] = {}

    def _fake_openai(**kwargs: object) -> _FakeClient:
        captured_client_kwargs.update(kwargs)
        return _FakeClient(completions)

    monkeypatch.setattr(api, "OpenAI", _fake_openai)

    result = api.chat_api_call(
        model="test-model",
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
        max_tokens=16,
    )

    assert result == {"ok": True}
    assert captured_client_kwargs["api_key"] == "test-key"
    assert captured_client_kwargs["base_url"] == "https://example.invalid/v1"
    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "question"},
            ],
            "max_tokens": 16,
            "temperature": None,
            "top_p": None,
            "extra_body": {"reasoning_effort": "high"},
        }
    ]
