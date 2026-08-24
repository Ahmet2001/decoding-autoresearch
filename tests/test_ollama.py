import json

import httpx
import pytest

from prompt_research.models import BenchmarkCase, DecodingConfig
from prompt_research.ollama import OllamaClient, OllamaError


DECODING = DecodingConfig(1.0, 0.95, 20, 0.0)
CASE = BenchmarkCase(
    "x", "screen", "en", "test", [{"role": "user", "content": "hello"}], {"type": "judge"}, "x"
)


def client(handler) -> OllamaClient:
    return OllamaClient(
        base_url="http://ollama.test",
        model="qwen",
        context_length=4096,
        output_tokens=100,
        timeout=1,
        transport=httpx.MockTransport(handler),
    )


def test_generate_requires_thinking_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "ok"}, "done_reason": "stop"})

    ollama = client(handler)
    with pytest.raises(OllamaError, match="thinking"):
        ollama.generate(case=CASE, system_prompt="x", decoding=DECODING, seed=1)
    ollama.close()


def test_timeout_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    ollama = client(handler)
    with pytest.raises(OllamaError, match="failed"):
        ollama.generate(case=CASE, system_prompt="x", decoding=DECODING, seed=1)
    ollama.close()


def test_generate_sends_every_mutable_decoding_option() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "message": {"thinking": "done", "content": "ok"},
                "done_reason": "stop",
            },
        )

    ollama = client(handler)
    ollama.generate(case=CASE, system_prompt="fixed", decoding=DECODING, seed=17)
    ollama.close()
    body = json.loads(requests[0].content)
    assert body["options"] == {
        "seed": 17,
        "num_ctx": 4096,
        "num_predict": 100,
        **DECODING.to_dict(),
    }


def test_preflight_rejects_cpu_fallback() -> None:
    calls = {"ps": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "1"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen"}]})
        if request.url.path == "/api/ps":
            calls["ps"] += 1
            if calls["ps"] == 1:
                return httpx.Response(200, json={"models": []})
            return httpx.Response(200, json={"models": [{"name": "qwen", "size": 100, "size_vram": 0}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"thinking": "", "content": "OK"}, "done_reason": "stop"})
        return httpx.Response(200, json={})

    ollama = client(handler)
    with pytest.raises(OllamaError, match="GPU fraction"):
        ollama.preflight(decoding=DECODING, minimum_gpu_fraction=0.9)
    ollama.close()
