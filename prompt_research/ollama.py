from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .models import BenchmarkCase, DecodingConfig, GenerationResult


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        context_length: int,
        output_tokens: int,
        timeout: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.context_length = context_length
        self.output_tokens = output_tokens
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self.http.close()

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.http.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request {path} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise OllamaError(f"Ollama returned a non-object response from {path}")
        if data.get("error"):
            raise OllamaError(str(data["error"]))
        return data

    def version(self) -> str:
        return str(self._json("GET", "/api/version").get("version", "unknown"))

    def installed_models(self) -> list[str]:
        data = self._json("GET", "/api/tags")
        return [str(model.get("name", "")) for model in data.get("models", [])]

    def running_models(self) -> list[dict[str, Any]]:
        return list(self._json("GET", "/api/ps").get("models", []))

    def unload(self, model: str) -> None:
        self._json("POST", "/api/generate", json={"model": model, "keep_alive": 0})

    def reset_loaded_models(self) -> None:
        for model in self.running_models():
            name = str(model.get("name") or model.get("model") or "")
            if name:
                self.unload(name)

    def generate(
        self,
        *,
        case: BenchmarkCase,
        system_prompt: str,
        decoding: DecodingConfig,
        seed: int,
        output_tokens: int | None = None,
    ) -> GenerationResult:
        messages = [{"role": "system", "content": system_prompt}, *case.messages]
        started = time.monotonic()
        data = self._json(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": True,
                "keep_alive": "30m",
                "options": {
                    "seed": seed,
                    "num_ctx": self.context_length,
                    "num_predict": output_tokens or self.output_tokens,
                    "temperature": decoding.temperature,
                    "top_p": decoding.top_p,
                    "top_k": decoding.top_k,
                    "min_p": decoding.min_p,
                },
            },
        )
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama chat response does not contain a message object")
        if "thinking" not in message:
            raise OllamaError(
                "Ollama did not return message.thinking; update Ollama or verify the model parser"
            )
        return GenerationResult(
            case_id=case.id,
            split=case.split,
            language=case.language,
            category=case.category,
            seed=seed,
            thinking=str(message.get("thinking") or ""),
            content=str(message.get("content") or ""),
            done_reason=str(data.get("done_reason") or "unknown"),
            eval_count=int(data.get("eval_count") or 0),
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            duration_seconds=float(data.get("total_duration") or 0) / 1_000_000_000
            or (time.monotonic() - started),
        )

    def preflight(
        self,
        *,
        decoding: DecodingConfig,
        minimum_gpu_fraction: float,
        require_gpu: bool = True,
    ) -> dict[str, Any]:
        version = self.version()
        installed = self.installed_models()
        if self.model not in installed:
            raise OllamaError(
                f"Target model {self.model!r} is not installed. Installed models: {installed}"
            )
        self.reset_loaded_models()
        probe = BenchmarkCase(
            id="preflight",
            split="screen",
            language="en",
            category="preflight",
            messages=[{"role": "user", "content": "Reply with only OK."}],
            validator={"type": "contains_any", "values": ["ok"]},
            rubric="Reply successfully.",
        )
        result = self.generate(
            case=probe,
            system_prompt="Be concise.",
            decoding=decoding,
            seed=1,
            output_tokens=32,
        )
        running = self.running_models()
        target = next(
            (
                row
                for row in running
                if str(row.get("name") or row.get("model") or "") == self.model
            ),
            None,
        )
        if target is None:
            raise OllamaError("Target model disappeared after the preflight generation")
        size = float(target.get("size") or 0)
        size_vram = float(target.get("size_vram") or 0)
        gpu_fraction = size_vram / size if size > 0 else 0.0
        if require_gpu and gpu_fraction < minimum_gpu_fraction:
            raise OllamaError(
                f"Target model GPU fraction is {gpu_fraction:.1%}; required "
                f"{minimum_gpu_fraction:.1%}. Stop competing models and free VRAM."
            )
        return {
            "ollama_version": version,
            "model": self.model,
            "gpu_fraction": gpu_fraction,
            "thinking_field": True,
            "decoding_config": decoding.to_dict(),
            "probe_answer": result.content,
        }
