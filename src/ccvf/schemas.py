from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Condition(str, Enum):
    N = "N"
    W = "W"
    W_NAIVE = "W_naive"
    SF = "SF"
    CC = "CC"


class VerificationDecision(str, Enum):
    ACCEPT = "Accept"
    REJECT = "Reject"
    NOT_SURE = "Not Sure"
    NA = "NA"


class ParseStatus(str, Enum):
    OK = "ok"
    REPAIRED = "repaired"
    FAILED = "failed"


class ErrorType(str, Enum):
    NONE = "none"
    PARSE = "parse"
    API_CONNECTION = "api_connection"
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_INTERNAL = "api_internal"
    API_UNEXPECTED = "api_unexpected"
    UNEXPECTED = "unexpected"


class DatasetSample(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    context: str = ""
    question: str
    answers: list[str]
    label: int
    selected_distractor: int | None = None
    dataset: str = "unknown"
    explanation: dict[str, Any] | None = None
    reasoning_category: str | None = None
    difficulty: str | None = None

    @model_validator(mode="after")
    def validate_indices(self) -> "DatasetSample":
        if not self.answers:
            raise ValueError("answers must be non-empty")
        n = len(self.answers)
        if not 0 <= self.label < n:
            raise ValueError(f"label {self.label} out of range")
        if self.selected_distractor is None:
            self.selected_distractor = next(i for i in range(n) if i != self.label)
        if not 0 <= self.selected_distractor < n:
            raise ValueError(f"selected_distractor {self.selected_distractor} out of range")
        if self.selected_distractor == self.label:
            raise ValueError("selected_distractor must not equal label")
        return self


class ModelOutput(BaseModel):
    sample_id: str
    condition: Condition
    candidate_index: int | None = None
    selected_answer_index: int
    verification_decision: VerificationDecision
    confidence: float = Field(ge=0, le=100)
    verification_reasoning: str = ""
    decision_justification: str = ""
    alternative_option_analysis: str = ""

    @model_validator(mode="after")
    def validate_condition(self) -> "ModelOutput":
        if self.condition == Condition.N:
            self.candidate_index = None
            self.verification_decision = VerificationDecision.NA
        elif self.verification_decision == VerificationDecision.NA:
            raise ValueError(f"{self.condition.value} cannot use verification_decision=NA")
        return self


class RawRecord(BaseModel):
    run_id: str
    model_id: str
    provider: str
    sample_id: str
    dataset: str
    condition: Condition
    candidate_index: int | None
    committed_answer_index: int | None = None
    candidate_text: str | None
    candidate_type: Literal["none", "distractor", "gold"]
    prompt_template: str
    prompt_template_hash: str
    dataset_hash: str
    request_timestamp: str
    response_timestamp: str
    raw_response: str
    raw_provider_response: dict[str, Any] | None = None
    usage_tokens: dict[str, int] = Field(default_factory=dict)
    latency_ms: float | None = None
    api_request_id: str | None = None
    prompt_chars: int | None = None
    attempt_count: int = 0
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    error_type: ErrorType = ErrorType.NONE
    is_infrastructure_error: bool = False
    server_health: dict[str, Any] | None = None
    parse_status: ParseStatus
    parsed: ModelOutput | None = None
    error: str | None = None


class ModelSpec(BaseModel):
    id: str
    provider: Literal["openai", "local_openai", "hf_local"]
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    response_format: Literal["json_object", "none"] = "none"
    api_endpoint: Literal["chat", "completions"] = "chat"
    prompt_wrapper: Literal["none", "qwen", "cohere", "aya", "gemma3", "gemma4", "phi4"] = "none"
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    omit_sampling_params: bool = False
    max_workers: int | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)
    hf_max_memory_gib: int = 30
    hf_attn_implementation: str = "eager"


class RunConfig(BaseModel):
    run_id: str
    output_dir: str = "results/runs"
    overwrite: bool = False
    resume: bool = True
    seed: int = 13


class DatasetConfig(BaseModel):
    path: str
    allow_non_json_lines: bool = False
    limit: int | None = None


class InferenceConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 512
    preflight: bool = True
    preflight_max_output_tokens: int = 64
    timeout_seconds: float = 120
    concurrency: int = 32
    retries: int = 2
    retry_backoff_seconds: float = 2.0
    server_ready_timeout_seconds: float = 900
    server_ready_poll_seconds: float = 5
    abort_on_infrastructure_error: bool = True
    infrastructure_failure_threshold: int = 8
    infrastructure_failure_window: int = 32
    infrastructure_failure_rate: float = 0.75
    health_check_timeout_seconds: float = 5


class ExperimentConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    prompt_dir: str = "prompts"
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    models: list[ModelSpec]
