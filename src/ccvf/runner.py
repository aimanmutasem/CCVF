from __future__ import annotations

import asyncio
import json
import random
import re
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ccvf.analysis import apply_holm_correction, compute_grouped_metrics_and_stats, records_dataframe
from ccvf.client import CompletionClient, CompletionError, build_client
from ccvf.io import append_jsonl, load_dataset, sha256_file
from ccvf.parser import parse_output
from ccvf.prompts import PromptBuilder
from ccvf.report import write_report, write_tables
from ccvf.schemas import Condition, DatasetSample, ErrorType, ExperimentConfig, ModelSpec, ParseStatus, RawRecord

INFRASTRUCTURE_ERROR_TYPES = {
    ErrorType.API_CONNECTION,
    ErrorType.API_TIMEOUT,
    ErrorType.API_RATE_LIMIT,
    ErrorType.API_INTERNAL,
    ErrorType.API_UNEXPECTED,
}


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


class InfrastructureAbort(RuntimeError):
    pass


def _finish_reason(record: RawRecord) -> str | None:
    response = record.raw_provider_response or {}
    choices = response.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, config_path: Path):
        self.config = config
        self.config_path = config_path
        self.root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        self.dataset_path = self._resolve(config.dataset.path)
        self.prompt_dir = self._resolve(config.prompt_dir)
        self.output_root = self._resolve(config.run.output_dir)
        self.run_dir = self.output_root / config.run.run_id
        if config.run.overwrite and self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        for sub in ("raw", "parsed", "metrics", "reports", "logs"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.prompt_builder = PromptBuilder(self.prompt_dir)
        self.clients: dict[str, CompletionClient] = {}
        self.dataset_hash = sha256_file(self.dataset_path)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def raw_path(self, model_id: str, condition: Condition) -> Path:
        return self.run_dir / "raw" / f"{safe_slug(model_id)}_{condition.value}.jsonl"

    def event_path(self) -> Path:
        return self.run_dir / "logs" / "runner_events.jsonl"

    def log_event(self, event: str, **fields: Any) -> None:
        append_jsonl(
            self.event_path(),
            {
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **fields,
            },
        )

    def load_existing(self, model: ModelSpec, condition: Condition) -> dict[str, RawRecord]:
        path = self.raw_path(model.id, condition)
        if not self.config.run.resume or not path.exists():
            return {}
        out = {}
        skipped_infra = 0
        malformed = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = RawRecord.model_validate_json(line)
                    except Exception:  # noqa: BLE001
                        malformed += 1
                        continue
                    if self.is_infrastructure_record(record):
                        skipped_infra += 1
                        continue
                    out[record.sample_id] = record
        if skipped_infra or malformed:
            self.log_event(
                "resume_skipped_records",
                model_id=model.id,
                condition=condition.value,
                skipped_infrastructure=skipped_infra,
                skipped_malformed=malformed,
                kept=len(out),
            )
        return out

    @staticmethod
    def is_infrastructure_record(record: RawRecord) -> bool:
        if record.is_infrastructure_error:
            return True
        if record.error_type in INFRASTRUCTURE_ERROR_TYPES:
            return True
        error = record.error or ""
        return any(
            marker in error
            for marker in (
                "APIConnectionError",
                "APITimeoutError",
                "RateLimitError",
                "InternalServerError",
            )
        )

    async def run_condition_tasks(
        self,
        *,
        model: ModelSpec,
        samples: list[DatasetSample],
        condition: Condition,
        committed: dict[str, int] | None = None,
    ) -> list[RawRecord]:
        existing = self.load_existing(model, condition)
        records = list(existing.values())
        pending = [s for s in samples if s.id not in existing]
        self.log_event(
            "condition_start",
            model_id=model.id,
            condition=condition.value,
            total_samples=len(samples),
            resumed=len(existing),
            pending=len(pending),
        )
        if not pending:
            self.log_event(
                "condition_complete",
                model_id=model.id,
                condition=condition.value,
                total_records=len(records),
                resumed=len(existing),
                completed_now=0,
                infrastructure_failures=0,
            )
            return records
        client = self.get_client(model)
        limit = model.max_workers or self.config.inference.concurrency
        worker_count = max(1, min(limit, self.config.inference.concurrency, len(pending)))
        queue: asyncio.Queue[DatasetSample] = asyncio.Queue()
        result_queue: asyncio.Queue[RawRecord | None] = asyncio.Queue()
        stop_event = asyncio.Event()
        for sample in pending:
            queue.put_nowait(sample)

        async def worker() -> None:
            try:
                while not stop_event.is_set():
                    try:
                        sample = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        if stop_event.is_set():
                            break
                        record = await self._run_one(client, model, sample, condition, committed)
                        await result_queue.put(record)
                    finally:
                        queue.task_done()
            finally:
                await result_queue.put(None)

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        finished_workers = 0
        completed_now = 0
        infrastructure_failures = 0
        consecutive_infrastructure_failures = 0
        recent_infrastructure = deque(maxlen=max(1, self.config.inference.infrastructure_failure_window))
        pbar = tqdm(desc=f"{model.id} {condition.value}", total=len(pending))
        try:
            while finished_workers < worker_count:
                item = await result_queue.get()
                if item is None:
                    finished_workers += 1
                    continue

                record = item
                if self.is_infrastructure_record(record):
                    infrastructure_failures += 1
                    consecutive_infrastructure_failures += 1
                    recent_infrastructure.append(1)
                    record.server_health = await client.health_check()
                    self.log_event(
                        "infrastructure_failure",
                        model_id=model.id,
                        condition=condition.value,
                        sample_id=record.sample_id,
                        error_type=record.error_type.value,
                        error=record.error,
                        consecutive_infrastructure_failures=consecutive_infrastructure_failures,
                        infrastructure_failures=infrastructure_failures,
                        attempt_count=record.attempt_count,
                        attempts=record.attempts,
                        server_health=record.server_health,
                    )
                else:
                    consecutive_infrastructure_failures = 0
                    recent_infrastructure.append(0)

                append_jsonl(self.raw_path(model.id, condition), record.model_dump(mode="json"))
                records.append(record)
                completed_now += 1
                pbar.update(1)

                if self._should_abort_condition(consecutive_infrastructure_failures, recent_infrastructure):
                    stop_event.set()
                    for task in workers:
                        task.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    message = (
                        f"Aborting {model.id} {condition.value}: infrastructure failure circuit breaker opened "
                        f"after {consecutive_infrastructure_failures} consecutive infrastructure failures "
                        f"({infrastructure_failures} total in this condition). "
                        "Fix or restart the model server, then rerun with resume enabled."
                    )
                    self.log_event(
                        "condition_aborted",
                        model_id=model.id,
                        condition=condition.value,
                        message=message,
                        completed_now=completed_now,
                        resumed=len(existing),
                        pending_remaining=queue.qsize(),
                        infrastructure_failures=infrastructure_failures,
                    )
                    raise InfrastructureAbort(message)

            await asyncio.gather(*workers, return_exceptions=False)
        finally:
            pbar.close()
        self.log_event(
            "condition_complete",
            model_id=model.id,
            condition=condition.value,
            total_records=len(records),
            resumed=len(existing),
            completed_now=completed_now,
            infrastructure_failures=infrastructure_failures,
        )
        return records

    def _should_abort_condition(self, consecutive_infrastructure_failures: int, recent_infrastructure: deque[int]) -> bool:
        if not self.config.inference.abort_on_infrastructure_error:
            return False
        if consecutive_infrastructure_failures >= self.config.inference.infrastructure_failure_threshold:
            return True
        if len(recent_infrastructure) < recent_infrastructure.maxlen:
            return False
        return (sum(recent_infrastructure) / len(recent_infrastructure)) >= self.config.inference.infrastructure_failure_rate

    def get_client(self, model: ModelSpec) -> CompletionClient:
        if model.id not in self.clients:
            self.clients[model.id] = build_client(model, self.config.inference)
        return self.clients[model.id]

    async def _run_one(
        self,
        client: CompletionClient,
        model: ModelSpec,
        sample: DatasetSample,
        condition: Condition,
        committed: dict[str, int] | None,
    ) -> RawRecord:
        candidate_index = self.prompt_builder.candidate_index(condition, sample)
        committed_answer = committed.get(sample.id) if committed else None
        prompt, template_name, template_hash = self.prompt_builder.build(
            condition,
            sample,
            committed_answer_index=committed_answer,
        )
        candidate_type = "none" if condition == Condition.N else ("gold" if condition == Condition.CC else "distractor")
        request_timestamp = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        try:
            completion = await client.complete(prompt)
            status, parsed, parse_error = parse_output(completion.text)
            if parsed is not None and parsed.condition != condition:
                parse_error = (
                    f"condition mismatch: expected {condition.value}, "
                    f"got {parsed.condition.value}"
                )
                status = ParseStatus.FAILED
                parsed = None
            error_type = ErrorType.PARSE if status == ParseStatus.FAILED else ErrorType.NONE
            return RawRecord(
                run_id=self.config.run.run_id,
                model_id=model.id,
                provider=model.provider,
                sample_id=sample.id,
                dataset=sample.dataset,
                condition=condition,
                candidate_index=candidate_index,
                committed_answer_index=committed_answer,
                candidate_text=None if candidate_index is None else sample.answers[candidate_index],
                candidate_type=candidate_type,
                prompt_template=template_name,
                prompt_template_hash=template_hash,
                dataset_hash=self.dataset_hash,
                request_timestamp=completion.request_timestamp,
                response_timestamp=completion.response_timestamp,
                raw_response=completion.text,
                raw_provider_response=completion.raw_response,
                usage_tokens=completion.usage_tokens,
                latency_ms=completion.latency_ms,
                api_request_id=completion.api_request_id,
                prompt_chars=len(prompt),
                attempt_count=completion.attempt_count,
                attempts=completion.attempts or [],
                error_type=error_type,
                is_infrastructure_error=False,
                parse_status=status,
                parsed=parsed,
                error=parse_error,
            )
        except CompletionError as exc:
            return RawRecord(
                run_id=self.config.run.run_id,
                model_id=model.id,
                provider=model.provider,
                sample_id=sample.id,
                dataset=sample.dataset,
                condition=condition,
                candidate_index=candidate_index,
                committed_answer_index=committed_answer,
                candidate_text=None if candidate_index is None else sample.answers[candidate_index],
                candidate_type=candidate_type,
                prompt_template=template_name,
                prompt_template_hash=template_hash,
                dataset_hash=self.dataset_hash,
                request_timestamp=exc.request_timestamp,
                response_timestamp=exc.response_timestamp,
                raw_response="",
                raw_provider_response=None,
                usage_tokens={},
                latency_ms=exc.latency_ms,
                api_request_id=None,
                prompt_chars=len(prompt),
                attempt_count=exc.attempt_count,
                attempts=exc.attempts,
                error_type=exc.error_type,
                is_infrastructure_error=True,
                parse_status=ParseStatus.FAILED,
                parsed=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            return RawRecord(
                run_id=self.config.run.run_id,
                model_id=model.id,
                provider=model.provider,
                sample_id=sample.id,
                dataset=sample.dataset,
                condition=condition,
                candidate_index=candidate_index,
                committed_answer_index=committed_answer,
                candidate_text=None if candidate_index is None else sample.answers[candidate_index],
                candidate_type=candidate_type,
                prompt_template=template_name,
                prompt_template_hash=template_hash,
                dataset_hash=self.dataset_hash,
                request_timestamp=request_timestamp,
                response_timestamp=datetime.now(timezone.utc).isoformat(),
                raw_response="",
                raw_provider_response=None,
                usage_tokens={},
                latency_ms=(time.perf_counter() - start) * 1000,
                api_request_id=None,
                prompt_chars=len(prompt),
                attempt_count=0,
                attempts=[],
                error_type=ErrorType.UNEXPECTED,
                is_infrastructure_error=False,
                parse_status=ParseStatus.FAILED,
                parsed=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def preflight_model(self, model: ModelSpec) -> None:
        if not self.config.inference.preflight:
            return
        await self.wait_for_server_ready(model)
        sample = DatasetSample(
            id="CCVF_PREFLIGHT",
            dataset="preflight",
            context="",
            question="What is 1 + 1?",
            answers=["1", "2", "3", "4"],
            label=1,
            selected_distractor=0,
        )
        client = self.get_client(model)
        prompt, template_name, template_hash = self.prompt_builder.build(Condition.N, sample)
        request_timestamp = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        preflight_max_output_tokens = max(1, self.config.inference.preflight_max_output_tokens)
        try:
            completion = await client.complete(
                prompt,
                max_output_tokens=preflight_max_output_tokens,
            )
        except CompletionError as exc:
            health = await client.health_check()
            self.log_event(
                "preflight_failed",
                model_id=model.id,
                provider=model.provider,
                model=model.model,
                base_url=model.base_url,
                error_type=exc.error_type.value,
                error=f"{type(exc).__name__}: {exc}",
                attempts=exc.attempts,
                server_health=health,
            )
            health_detail = ""
            if model.provider == "local_openai":
                health_detail = f" Tried base_url={model.base_url}; health={health}."
            raise InfrastructureAbort(
                f"Preflight failed for {model.id}: {type(exc).__name__}: {exc}. "
                f"{health_detail} Fix or restart the model server before running the full benchmark."
            ) from exc
        status, parsed, parse_error = parse_output(completion.text)
        record = RawRecord(
            run_id=self.config.run.run_id,
            model_id=model.id,
            provider=model.provider,
            sample_id=sample.id,
            dataset=sample.dataset,
            condition=Condition.N,
            candidate_index=None,
            committed_answer_index=None,
            candidate_text=None,
            candidate_type="none",
            prompt_template=template_name,
            prompt_template_hash=template_hash,
            dataset_hash=self.dataset_hash,
            request_timestamp=completion.request_timestamp,
            response_timestamp=completion.response_timestamp,
            raw_response=completion.text,
            raw_provider_response=completion.raw_response,
            usage_tokens=completion.usage_tokens,
            latency_ms=completion.latency_ms,
            api_request_id=completion.api_request_id,
            prompt_chars=len(prompt),
            attempt_count=completion.attempt_count,
            attempts=completion.attempts or [],
            error_type=ErrorType.PARSE if status == ParseStatus.FAILED else ErrorType.NONE,
            is_infrastructure_error=False,
            parse_status=status,
            parsed=parsed,
            error=parse_error,
        )
        failure = self.generation_failure_reason(record, max_output_tokens=preflight_max_output_tokens)
        if failure:
            self.log_event(
                "preflight_failed",
                model_id=model.id,
                error_type="generation",
                error=failure,
                raw_response_preview=completion.text[:500],
                usage_tokens=completion.usage_tokens,
                finish_reason=_finish_reason(record),
                latency_ms=completion.latency_ms,
            )
            raise InfrastructureAbort(
                f"Preflight failed for {model.id}: {failure}. "
                "The server returned an unusable completion, so the 10k benchmark was not started."
            )
        if status == ParseStatus.FAILED:
            self.log_event(
                "preflight_failed",
                model_id=model.id,
                error_type=ErrorType.PARSE.value,
                error=parse_error,
                raw_response_preview=completion.text[:500],
                usage_tokens=completion.usage_tokens,
                finish_reason=_finish_reason(record),
                latency_ms=completion.latency_ms,
            )
            raise InfrastructureAbort(
                f"Preflight failed for {model.id}: model output did not parse as CCVF JSON "
                f"({parse_error}). Raw output begins: {completion.text[:200]!r}"
            )
        self.log_event(
            "preflight_complete",
            model_id=model.id,
            parse_status=status.value,
            selected_answer_index=parsed.selected_answer_index if parsed else None,
            confidence=parsed.confidence if parsed else None,
            usage_tokens=completion.usage_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_timestamp=request_timestamp,
        )

    async def wait_for_server_ready(self, model: ModelSpec) -> None:
        if model.provider != "local_openai":
            return
        client = self.get_client(model)
        timeout = max(0.0, self.config.inference.server_ready_timeout_seconds)
        poll = max(0.25, self.config.inference.server_ready_poll_seconds)
        start = time.perf_counter()
        last_health: dict | None = None
        logged_waiting = False
        while True:
            health = await client.health_check()
            last_health = health
            if health.get("ok") is True:
                if logged_waiting:
                    self.log_event(
                        "server_ready",
                        model_id=model.id,
                        base_url=model.base_url,
                        waited_seconds=round(time.perf_counter() - start, 3),
                        server_health=health,
                    )
                return
            elapsed = time.perf_counter() - start
            if not logged_waiting:
                self.log_event(
                    "server_waiting",
                    model_id=model.id,
                    base_url=model.base_url,
                    server_health=health,
                    timeout_seconds=timeout,
                )
                logged_waiting = True
            if elapsed >= timeout:
                self.log_event(
                    "server_not_ready",
                    model_id=model.id,
                    base_url=model.base_url,
                    waited_seconds=round(elapsed, 3),
                    server_health=last_health,
                )
                raise InfrastructureAbort(
                    f"Preflight failed for {model.id}: local server at {model.base_url} "
                    f"was not ready after {timeout:.0f}s. Last health={last_health}. "
                    "Wait for vLLM to print 'Application startup complete' or restart the model server."
                )
            await asyncio.sleep(poll)

    def generation_failure_reason(self, record: RawRecord, *, max_output_tokens: int | None = None) -> str | None:
        if record.is_infrastructure_error:
            return None
        finish_reason = _finish_reason(record)
        output_tokens = int(record.usage_tokens.get("output_tokens") or 0)
        output_limit = max_output_tokens or self.config.inference.max_output_tokens
        if (
            finish_reason == "length"
            and output_tokens >= output_limit
            and not record.raw_response.strip()
        ):
            return (
                "empty completion reached max_output_tokens with finish_reason=length. "
                "This usually indicates a serving/model-template incompatibility, such as "
                "a backend repeatedly generating skipped special tokens."
            )
        return None

    def validate_condition_records(self, model: ModelSpec, condition: Condition, records: list[RawRecord]) -> None:
        infrastructure = [rec for rec in records if self.is_infrastructure_record(rec)]
        if infrastructure:
            examples = ", ".join(f"{rec.sample_id}:{rec.error_type.value}" for rec in infrastructure[:5])
            self.log_event(
                "condition_validation_failed",
                model_id=model.id,
                condition=condition.value,
                reason="infrastructure_failures",
                infrastructure_failures=len(infrastructure),
                examples=examples,
            )
            raise InfrastructureAbort(
                f"{model.id} {condition.value} contains {len(infrastructure)} infrastructure failures "
                f"after retries ({examples}). Metrics would be invalid; fix/restart the server and rerun "
                "with resume enabled."
            )
        generation_failures = [rec for rec in records if self.generation_failure_reason(rec)]
        if generation_failures:
            examples = ", ".join(rec.sample_id for rec in generation_failures[:5])
            reason = self.generation_failure_reason(generation_failures[0])
            self.log_event(
                "condition_validation_failed",
                model_id=model.id,
                condition=condition.value,
                reason="generation_failures",
                generation_failures=len(generation_failures),
                examples=examples,
                diagnostic=reason,
            )
            raise InfrastructureAbort(
                f"{model.id} {condition.value} has {len(generation_failures)} unusable empty max-token "
                f"generations ({examples}). {reason}"
            )
        parseable = sum(1 for rec in records if rec.parse_status in {ParseStatus.OK, ParseStatus.REPAIRED})
        if records and parseable == 0:
            self.log_event(
                "condition_validation_failed",
                model_id=model.id,
                condition=condition.value,
                reason="zero_parseable_records",
                total_records=len(records),
            )
            raise InfrastructureAbort(
                f"{model.id} {condition.value} produced zero parseable records. "
                "Check the prompt template, chat template, and serving backend before continuing."
            )

    async def run(self) -> Path:
        self.log_event("run_start", run_id=self.config.run.run_id, config_path=str(self.config_path))
        try:
            samples = load_dataset(
                self.dataset_path,
                allow_non_json_lines=self.config.dataset.allow_non_json_lines,
                limit=self.config.dataset.limit,
            )
            manifest = {
                "path": str(self.dataset_path),
                "sha256": self.dataset_hash,
                "num_samples": len(samples),
            }
            (self.run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            shutil.copy2(self.config_path, self.run_dir / "config.yaml")
            all_records: list[RawRecord] = []
            rng = random.Random(self.config.run.seed)

            for model in self.config.models:
                await self.preflight_model(model)
                n_records = await self.run_condition_tasks(model=model, samples=samples, condition=Condition.N)
                self.validate_condition_records(model, Condition.N, n_records)
                all_records.extend(n_records)
                committed: dict[str, int] = {}
                ics_samples: list[DatasetSample] = []
                sample_map = {s.id: s for s in samples}
                for rec in n_records:
                    if rec.parsed and rec.parsed.selected_answer_index == sample_map[rec.sample_id].label:
                        committed[rec.sample_id] = rec.parsed.selected_answer_index
                        ics_samples.append(sample_map[rec.sample_id])
                w_naive_records = await self.run_condition_tasks(
                    model=model,
                    samples=samples,
                    condition=Condition.W_NAIVE,
                )
                self.validate_condition_records(model, Condition.W_NAIVE, w_naive_records)
                all_records.extend(w_naive_records)
                rng.shuffle(ics_samples)
                for cond in (Condition.W, Condition.SF, Condition.CC):
                    records = await self.run_condition_tasks(
                        model=model,
                        samples=ics_samples,
                        condition=cond,
                        committed=committed,
                    )
                    self.validate_condition_records(model, cond, records)
                    all_records.extend(records)

            df = records_dataframe(all_records, samples)
            if "is_infrastructure_error" in df.columns and df["is_infrastructure_error"].fillna(False).any():
                count = int(df["is_infrastructure_error"].fillna(False).sum())
                raise InfrastructureAbort(
                    f"Run contains {count} infrastructure failures; metrics were not written because they would be invalid."
                )
            metrics, stats = [], []
            for model in self.config.models:
                model_metrics, model_stats = compute_grouped_metrics_and_stats(df, model.id, samples)
                metrics.extend(model_metrics)
                stats.extend(model_stats)
            # Apply Holm correction once across the complete family of
            # model × dataset comparisons (e.g., 9 × 3 = 27 rows).
            stats = apply_holm_correction(stats)
            write_tables(self.run_dir, df, metrics, stats)
            write_report(self.run_dir, manifest, metrics, stats)
            self.log_event("run_complete", run_id=self.config.run.run_id, run_dir=str(self.run_dir))
            return self.run_dir
        except InfrastructureAbort as exc:
            self.log_event("run_aborted", run_id=self.config.run.run_id, error=str(exc))
            raise
        finally:
            await self.close_clients()

    async def close_clients(self) -> None:
        for client in self.clients.values():
            await client.aclose()
