from __future__ import annotations

import asyncio
import inspect
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

from ccvf.schemas import ErrorType, InferenceConfig, ModelSpec

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


@dataclass
class Completion:
    text: str
    raw_response: dict[str, Any]
    usage_tokens: dict[str, int]
    latency_ms: float
    request_timestamp: str
    response_timestamp: str
    api_request_id: str | None = None
    attempt_count: int = 0
    attempts: list[dict[str, Any]] | None = None


class CompletionError(RuntimeError):
    def __init__(
        self,
        *,
        error_type: ErrorType,
        message: str,
        request_timestamp: str,
        response_timestamp: str,
        latency_ms: float,
        attempts: list[dict[str, Any]],
    ):
        super().__init__(message)
        self.error_type = error_type
        self.request_timestamp = request_timestamp
        self.response_timestamp = response_timestamp
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.attempt_count = len(attempts)


class CompletionClient(Protocol):
    async def complete(self, prompt: str, *, max_output_tokens: int | None = None) -> Completion:
        ...

    async def health_check(self) -> dict[str, Any]:
        ...

    async def aclose(self) -> None:
        ...


def build_client(spec: ModelSpec, inference: InferenceConfig) -> CompletionClient:
    if spec.provider in {"openai", "local_openai"}:
        return OpenAICompatibleClient(spec, inference)
    if spec.provider == "hf_local":
        return HFLocalClient(spec, inference)
    raise ValueError(f"unknown provider: {spec.provider}")


class OpenAICompatibleClient:
    def __init__(self, spec: ModelSpec, inference: InferenceConfig):
        api_key = os.getenv(spec.api_key_env) if spec.api_key_env else None
        if spec.provider == "local_openai":
            api_key = api_key or "EMPTY"
        self.spec = spec
        self.inference = inference
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=spec.base_url,
            timeout=inference.timeout_seconds,
            max_retries=0,
        )

    async def complete(self, prompt: str, *, max_output_tokens: int | None = None) -> Completion:
        request_ts = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        attempts: list[dict[str, Any]] = []
        token_limit = max_output_tokens or self.inference.max_output_tokens
        if self.spec.api_endpoint == "completions":
            params: dict[str, Any] = {
                "model": self.spec.model,
                "prompt": _wrap_completion_prompt(prompt, self.spec.prompt_wrapper),
                self.spec.max_tokens_param: token_limit,
            }
        else:
            params = {
                "model": self.spec.model,
                "messages": [{"role": "user", "content": prompt}],
                self.spec.max_tokens_param: token_limit,
            }
        if not self.spec.omit_sampling_params:
            params["temperature"] = self.inference.temperature
            params["top_p"] = self.inference.top_p
        if self.spec.api_endpoint == "chat" and self.spec.response_format == "json_object":
            params["response_format"] = {"type": "json_object"}
        if self.spec.extra_body:
            params["extra_body"] = self.spec.extra_body

        last_error: Exception | None = None
        for attempt in range(self.inference.retries + 1):
            attempt_start = datetime.now(timezone.utc).isoformat()
            attempt_timer = time.perf_counter()
            try:
                if self.spec.api_endpoint == "completions":
                    response = await self.client.completions.create(**params)
                    text = response.choices[0].text or ""
                    if self.spec.prompt_wrapper in {"cohere", "aya", "gemma3", "gemma4", "phi4"} and text.strip():
                        text = "{\n" + text
                else:
                    response = await self.client.chat.completions.create(**params)
                    text = response.choices[0].message.content or ""
                attempt_end = datetime.now(timezone.utc).isoformat()
                attempt_latency_ms = (time.perf_counter() - attempt_timer) * 1000
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "ok",
                        "api_endpoint": self.spec.api_endpoint,
                        "request_timestamp": attempt_start,
                        "response_timestamp": attempt_end,
                        "latency_ms": attempt_latency_ms,
                    }
                )
                latency_ms = (time.perf_counter() - start) * 1000
                usage = {}
                if response.usage is not None:
                    usage = {
                        "input_tokens": response.usage.prompt_tokens or 0,
                        "output_tokens": response.usage.completion_tokens or 0,
                    }
                return Completion(
                    text=text,
                    raw_response=response.model_dump(mode="json"),
                    usage_tokens=usage,
                    latency_ms=latency_ms,
                    request_timestamp=request_ts,
                    response_timestamp=datetime.now(timezone.utc).isoformat(),
                    api_request_id=response.id,
                    attempt_count=len(attempts),
                    attempts=attempts,
                )
            except RETRYABLE as exc:
                last_error = exc
                error_type = _error_type(exc)
                attempt_end = datetime.now(timezone.utc).isoformat()
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "error",
                        "api_endpoint": self.spec.api_endpoint,
                        "error_type": error_type.value,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                        "request_timestamp": attempt_start,
                        "response_timestamp": attempt_end,
                        "latency_ms": (time.perf_counter() - attempt_timer) * 1000,
                    }
                )
                if attempt >= self.inference.retries:
                    break
                health = await self.health_check()
                attempts[-1]["server_health_after_error"] = health
                if self.spec.provider == "local_openai" and health.get("ok") is False:
                    break
                await asyncio.sleep(self.inference.retry_backoff_seconds * (2**attempt))
            except Exception as exc:
                last_error = exc
                attempt_end = datetime.now(timezone.utc).isoformat()
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "error",
                        "api_endpoint": self.spec.api_endpoint,
                        "error_type": ErrorType.API_UNEXPECTED.value,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                        "request_timestamp": attempt_start,
                        "response_timestamp": attempt_end,
                        "latency_ms": (time.perf_counter() - attempt_timer) * 1000,
                    }
                )
                break
        error_type = _error_type(last_error)
        raise CompletionError(
            error_type=error_type,
            message=f"{type(last_error).__name__}: {last_error}",
            request_timestamp=request_ts,
            response_timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=(time.perf_counter() - start) * 1000,
            attempts=attempts,
        )

    async def health_check(self) -> dict[str, Any]:
        if not self.spec.base_url:
            return {"ok": None, "reason": "no base_url"}
        root = _service_root(self.spec.base_url)
        models_url = f"{root}/v1/models"
        health_url = f"{root}/health"

        async def check_url(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
            started = datetime.now(timezone.utc).isoformat()
            try:
                response = await client.get(url)
                result: dict[str, Any] = {
                    "url": url,
                    "status_code": response.status_code,
                    "ok": 200 <= response.status_code < 300,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "started_timestamp": started,
                }
                if url.endswith("/v1/models") and result["ok"]:
                    try:
                        payload = response.json()
                        result["models"] = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
                    except Exception:  # noqa: BLE001
                        result["models"] = []
                return result
            except Exception as exc:  # noqa: BLE001
                return {
                    "url": url,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "started_timestamp": started,
                }

        async with httpx.AsyncClient(timeout=self.inference.health_check_timeout_seconds) as client:
            models = await check_url(client, models_url)
            if self.spec.provider != "local_openai":
                return models
            health = await check_url(client, health_url)
            if models["ok"]:
                models["health"] = health
                return models
            return {
                "ok": False,
                "url": models_url,
                "error": models.get("error") or f"HTTP {models.get('status_code')}",
                "models": models,
                "health": health,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def aclose(self) -> None:
        close = getattr(self.client, "close", None) or getattr(self.client, "aclose", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _wrap_completion_prompt(prompt: str, wrapper: str) -> str:
    prompt = prompt.strip()
    if wrapper == "none":
        return prompt
    if wrapper == "qwen":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if wrapper in {"cohere", "aya"}:
        return (
            "<BOS_TOKEN>"
            f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|>"
            "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{\n"
        )
    if wrapper in {"gemma3", "gemma4"}:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{{\n"
    if wrapper == "phi4":
        return f"<|im_start|>user<|im_sep|>{prompt}<|im_end|><|im_start|>assistant<|im_sep|>{{\n"
    raise ValueError(f"unknown prompt_wrapper: {wrapper}")


class HFLocalClient:
    """Direct Transformers fallback for checkpoints not supported by installed vLLM.

    This is intentionally single-flight: one model instance is shared and guarded
    by an async lock. Use vLLM/SGLang for fast 10k-sample production runs.
    """

    def __init__(self, spec: ModelSpec, inference: InferenceConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.inference = inference
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        max_memory = {0: f"{spec.hf_max_memory_gib}GiB", 1: f"{spec.hf_max_memory_gib}GiB", "cpu": "90GiB"}
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            attn_implementation=spec.hf_attn_implementation,
        )
        self.lock = asyncio.Lock()

    async def complete(self, prompt: str, *, max_output_tokens: int | None = None) -> Completion:
        async with self.lock:
            return await asyncio.to_thread(self._complete_sync, prompt, max_output_tokens)

    async def health_check(self) -> dict[str, Any]:
        return {"ok": True, "backend": "hf_local", "timestamp": datetime.now(timezone.utc).isoformat()}

    async def aclose(self) -> None:
        return None

    def _complete_sync(self, prompt: str, max_output_tokens: int | None = None) -> Completion:
        request_ts = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        messages = [{"role": "user", "content": prompt}]
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt_text = prompt
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        first_device = next(iter(self.model.hf_device_map.values())) if hasattr(self.model, "hf_device_map") else self.model.device
        if isinstance(first_device, str) and first_device == "cpu":
            device = "cpu"
        else:
            device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = bool(self.inference.temperature and self.inference.temperature > 0)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_output_tokens or self.inference.max_output_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self._eos_token_ids(),
        }
        if do_sample:
            generation_kwargs["temperature"] = self.inference.temperature
            generation_kwargs["top_p"] = self.inference.top_p
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        input_len = inputs["input_ids"].shape[-1]
        new_ids = output[0][input_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return Completion(
            text=text,
            raw_response={"backend": "transformers", "model": self.spec.model},
            usage_tokens={"input_tokens": int(input_len), "output_tokens": int(new_ids.shape[-1])},
            latency_ms=latency_ms,
            request_timestamp=request_ts,
            response_timestamp=datetime.now(timezone.utc).isoformat(),
            api_request_id=None,
            attempt_count=1,
            attempts=[
                {
                    "attempt": 1,
                    "status": "ok",
                    "request_timestamp": request_ts,
                    "response_timestamp": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": latency_ms,
                }
            ],
        )

    def _eos_token_ids(self) -> int | list[int] | None:
        ids = []
        if self.tokenizer.eos_token_id is not None:
            ids.append(self.tokenizer.eos_token_id)
        for token in ("<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0 and token_id not in ids:
                ids.append(token_id)
        if not ids:
            return None
        return ids[0] if len(ids) == 1 else ids


def _error_type(exc: Exception | None) -> ErrorType:
    if isinstance(exc, APIConnectionError):
        return ErrorType.API_CONNECTION
    if isinstance(exc, APITimeoutError):
        return ErrorType.API_TIMEOUT
    if isinstance(exc, RateLimitError):
        return ErrorType.API_RATE_LIMIT
    if isinstance(exc, InternalServerError):
        return ErrorType.API_INTERNAL
    return ErrorType.API_UNEXPECTED


def _service_root(base_url: str) -> str:
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", "")).rstrip("/")
