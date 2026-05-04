from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.reverse_job.adapters.ollama_client import OllamaLLMClient


def test_complete_posts_to_generate_endpoint() -> None:
    client = OllamaLLMClient(model="qwen2.5:14b", host="http://ollama:11434")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"response": "안녕하세요"}
    fake_resp.raise_for_status.return_value = None

    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = mock_client_cls.return_value.__enter__.return_value
        mock_ctx.post.return_value = fake_resp

        out = client.complete("프롬프트")

    assert out == "안녕하세요"
    args, kwargs = mock_ctx.post.call_args
    assert args[0] == "http://ollama:11434/api/generate"
    payload = kwargs["json"]
    assert payload["model"] == "qwen2.5:14b"
    assert payload["prompt"] == "프롬프트"
    assert payload["stream"] is False


def test_default_host_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://custom:9999/")
    client = OllamaLLMClient()
    assert client.host == "http://custom:9999"


def test_healthcheck_returns_true_on_200() -> None:
    client = OllamaLLMClient(host="http://ollama:11434")
    fake_resp = MagicMock()
    fake_resp.status_code = 200

    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = mock_client_cls.return_value.__enter__.return_value
        mock_ctx.get.return_value = fake_resp

        assert client.healthcheck() is True


def test_healthcheck_returns_false_on_error() -> None:
    import httpx

    client = OllamaLLMClient(host="http://ollama:11434")

    with patch("httpx.Client") as mock_client_cls:
        mock_ctx = mock_client_cls.return_value.__enter__.return_value
        mock_ctx.get.side_effect = httpx.ConnectError("refused")

        assert client.healthcheck() is False