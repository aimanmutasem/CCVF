from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import yaml

DEFAULT_TRANSFORMERS_OVERLAYS = [
    Path("/transformers513"),
    Path("/transformers513_min"),
]
KNOWN_UNSUPPORTED_VLLM085_ARCHITECTURES = {
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
}
KNOWN_UNSUPPORTED_TRANSFORMERS_FALLBACK_ARCHITECTURES = {
    "Gemma4UnifiedForConditionalGeneration",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve a local CCVF model with vLLM.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--dtype", default=None, help="vLLM dtype, e.g. half, bfloat16, or auto")
    parser.add_argument("--gpu-memory-utilization", default=None)
    parser.add_argument("--max-model-len", default=None)
    parser.add_argument("--max-num-seqs", default=None)
    parser.add_argument(
        "--generation-config",
        default=None,
        help="vLLM generation config source. Use 'vllm' for vLLM defaults or 'auto' for model generation_config.json.",
    )
    parser.add_argument("--load-format", default=None, help="Optional vLLM load format, useful for diagnostics")
    parser.add_argument("--device", default=None, help="Optional vLLM device override")
    parser.add_argument("--transformers-path", default=os.environ.get("CCVF_TRANSFORMERS_PATH"))
    parser.add_argument("--no-transformers-overlay", action="store_true")
    parser.add_argument(
        "--skip-transformers-probe",
        action="store_true",
        help="Skip the launcher's AutoConfig probe and let vLLM use its installed Transformers directly.",
    )
    parser.add_argument(
        "--config-probe-timeout",
        type=float,
        default=float(os.environ.get("CCVF_CONFIG_PROBE_TIMEOUT", "180")),
        help="Seconds to wait for each AutoConfig probe before trying an overlay.",
    )
    parser.add_argument(
        "--allow-unsupported-architecture",
        action="store_true",
        help="Skip the vLLM 0.8.5 architecture preflight and launch anyway.",
    )
    parser.add_argument(
        "--skip-gpu-memory-preflight",
        action="store_true",
        help="Launch vLLM without checking free GPU memory first. Use only for diagnostics.",
    )
    parser.add_argument("--vllm-bin", default=os.environ.get("VLLM_BIN", "vllm"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    models = raw.get("models") or []
    if len(models) != 1:
        raise SystemExit("serve_model.py expects a config with exactly one model")
    model = models[0]
    if model.get("provider") != "local_openai":
        raise SystemExit("serve_model.py is only for provider=local_openai")
    model_path = Path(model["model"])
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise SystemExit(
            f"Model checkpoint is incomplete or missing config.json: {model_path}\n"
            "Download the full checkpoint before serving it, or fix the model path in the config."
        )
    model_config = _read_model_config(config_path)
    serve_config = raw.get("serve") or {}
    requested_dtype = str(args.dtype or serve_config.get("dtype") or "half")
    dtype = _resolve_dtype_for_visible_gpus(requested_dtype, model_config)
    _preflight_model_dtype(model_config, dtype)
    gpu_memory_utilization = str(args.gpu_memory_utilization or serve_config.get("gpu_memory_utilization") or "0.80")
    max_model_len = str(args.max_model_len or serve_config.get("max_model_len") or "4096")
    max_num_seqs = str(args.max_num_seqs or serve_config.get("max_num_seqs") or "32")
    generation_config = str(args.generation_config or serve_config.get("generation_config") or "vllm")
    cmd = [
        args.vllm_bin,
        "serve",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(args.tp),
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--max-model-len",
        max_model_len,
        "--max-num-seqs",
        max_num_seqs,
        "--generation-config",
        generation_config,
        "--trust-remote-code",
    ]
    extra_args = serve_config.get("extra_args") or []
    if isinstance(extra_args, str):
        extra_args = [extra_args]
    if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
        raise SystemExit("serve.extra_args must be a string or a list of strings")
    cmd.extend(extra_args)
    if args.load_format:
        cmd.extend(["--load-format", args.load_format])
    if args.device:
        cmd.extend(["--device", args.device])
    env = os.environ.copy()
    serve_env = serve_config.get("env") or {}
    if not isinstance(serve_env, dict) or not all(isinstance(key, str) for key in serve_env):
        raise SystemExit("serve.env must be a mapping from environment-variable names to values")
    for key, value in serve_env.items():
        env[key] = str(value)
    overlay = _select_transformers_overlay(
        args.transformers_path,
        model_path,
        args.no_transformers_overlay,
        skip=args.skip_transformers_probe,
        timeout=args.config_probe_timeout,
    )
    if overlay is not None:
        env["PYTHONPATH"] = f"{overlay}:{env.get('PYTHONPATH', '')}".rstrip(":")
        print(f"Using Transformers overlay: {overlay}", flush=True)
    _preflight_vllm_architecture(model_config, args.vllm_bin, args.allow_unsupported_architecture)
    if serve_env:
        formatted_env = " ".join(f"{key}={env[key]}" for key in sorted(serve_env))
        print(f"vLLM environment overrides: {formatted_env}", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    existing_server = _check_existing_openai_server(args.port, model_path)
    if existing_server.get("ok"):
        model_ids = existing_server.get("model_ids") or []
        if _model_id_matches(model_ids, model_path):
            run_id = (raw.get("run") or {}).get("run_id") or model.get("id") or args.config.stem
            print(
                "\nA vLLM OpenAI server for this model is already running on "
                f"http://127.0.0.1:{args.port}/v1.",
                flush=True,
            )
            print(
                "Do not start serve_model again in this tmux. Use a second tmux for the benchmark:\n"
                "  export LOCAL_OPENAI_API_KEY=EMPTY\n"
                f"  python scripts/run_ccvf.py run --config {args.config} --run-id {run_id}",
                flush=True,
            )
            return 0
        raise SystemExit(
            "A vLLM OpenAI server is already running on the requested port, but it does not "
            "appear to be serving the requested model.\n"
            f"Requested model: {model_path}\n"
            f"Port: http://127.0.0.1:{args.port}/v1\n"
            f"Models reported by server: {model_ids}\n\n"
            "Use the existing server's matching config/run command, choose a different port, "
            "or stop only your own stale server before starting a new one."
        )
    already_loaded = _preflight_gpu_memory(
        gpu_memory_utilization,
        args.tp,
        serve_config,
        skip=args.skip_gpu_memory_preflight,
        model_path=model_path,
        port=args.port,
        config_path=args.config,
        run_id=(raw.get("run") or {}).get("run_id") or model.get("id") or args.config.stem,
    )
    if already_loaded:
        return 0
    return subprocess.call(cmd, env=env)


def _read_model_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid model config JSON: {path}: {exc}") from exc


def _resolve_dtype_for_visible_gpus(dtype: str, model_config: dict | None = None) -> str:
    normalized = dtype.lower()
    if normalized not in {"bfloat16", "bf16"}:
        return dtype
    try:
        import torch
    except Exception:
        return dtype
    try:
        if not torch.cuda.is_available():
            return dtype
        capabilities = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return dtype
    unsupported = [cap for cap in capabilities if cap < (8, 0)]
    if unsupported:
        if (model_config or {}).get("model_type") == "gemma3":
            names = []
            for i, cap in enumerate(capabilities):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = f"cuda:{i}"
                names.append(f"{name} cc={cap[0]}.{cap[1]}")
            raise SystemExit(
                "Gemma 3 cannot be served as float16 in vLLM 0.19.1, and the visible GPU(s) "
                "do not support bfloat16 "
                f"({', '.join(names)}). Use serve.dtype: float32 on V100, or run Gemma 3 "
                "with serve.dtype: bfloat16 on Ampere/Hopper GPUs."
            )
        names = []
        for i, cap in enumerate(capabilities):
            try:
                name = torch.cuda.get_device_name(i)
            except Exception:
                name = f"cuda:{i}"
            names.append(f"{name} cc={cap[0]}.{cap[1]}")
        print(
            "Warning: requested dtype bfloat16, but at least one visible GPU does not support BF16 "
            f"({', '.join(names)}). Using dtype=half for V100/pre-Ampere compatibility.",
            file=sys.stderr,
            flush=True,
        )
        return "half"
    return dtype


def _preflight_model_dtype(model_config: dict, dtype: str) -> None:
    if model_config.get("model_type") != "gemma3":
        return
    if dtype.lower() in {"half", "float16", "fp16"}:
        raise SystemExit(
            "Gemma 3 is configured with dtype=float16/half, but vLLM 0.19.1 rejects this "
            "dtype for Gemma 3 because of numerical instability. Use serve.dtype: float32 "
            "on V100, or serve.dtype: bfloat16 on GPUs with BF16 support."
        )


def _check_existing_openai_server(port: int, model_path: Path, timeout_seconds: float = 2.0) -> dict:
    models_url = f"http://127.0.0.1:{port}/v1/models"
    health_url = f"http://127.0.0.1:{port}/health"
    models = _fetch_json(models_url, timeout_seconds)
    if models.get("ok"):
        payload = models.get("payload") or {}
        model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]
        return {"ok": True, "url": models_url, "model_ids": model_ids}
    health = _fetch_text(health_url, timeout_seconds)
    if health.get("ok"):
        return {
            "ok": True,
            "url": health_url,
            "model_ids": [],
            "warning": f"health endpoint responded, but {models_url} did not return model IDs",
        }
    return {"ok": False, "models_error": models.get("error"), "health_error": health.get("error")}


def _fetch_json(url: str, timeout_seconds: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
        return {"ok": True, "payload": json.loads(body)}
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _fetch_text(url: str, timeout_seconds: float) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
        return {"ok": True, "body": body}
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _model_id_matches(model_ids: list[str], model_path: Path) -> bool:
    requested = str(model_path)
    requested_resolved = str(model_path.resolve())
    requested_name = model_path.name
    for model_id in model_ids:
        if model_id in {requested, requested_resolved, requested_name}:
            return True
        try:
            if Path(model_id).name == requested_name:
                return True
        except (OSError, ValueError):
            continue
    return False


def _preflight_gpu_memory(
    gpu_memory_utilization: str,
    tensor_parallel_size: int,
    serve_config: dict,
    *,
    skip: bool,
    model_path: Path,
    port: int,
    config_path: Path,
    run_id: str,
) -> bool:
    if skip or os.environ.get("CCVF_SKIP_GPU_MEMORY_PREFLIGHT") == "1" or serve_config.get("skip_gpu_memory_preflight"):
        return False
    try:
        utilization = float(gpu_memory_utilization)
    except ValueError:
        return False
    if not 0 < utilization <= 1:
        return False
    rows = _query_nvidia_smi_memory()
    if not rows:
        print("Warning: could not query nvidia-smi for GPU memory preflight.", file=sys.stderr, flush=True)
        return False
    selected = _select_gpu_rows(rows, tensor_parallel_size)
    if len(selected) < tensor_parallel_size:
        summary = _format_gpu_memory_rows(rows)
        raise SystemExit(
            "Not enough visible GPUs for the requested tensor parallel size.\n"
            f"Requested --tp {tensor_parallel_size}, but selected {len(selected)} GPU(s).\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n\n"
            f"GPU memory snapshot:\n{summary}"
        )
    failures = []
    for row in selected:
        required_mib = row["total_mib"] * utilization
        if row["free_mib"] < required_mib:
            failures.append((row, required_mib))
    if not failures:
        return False
    selected_summary = _format_gpu_memory_rows(selected)
    all_summary = _format_gpu_memory_rows(rows)
    detail = "\n".join(
        f"- GPU {row['index']} ({row['name']}): free {row['free_mib'] / 1024:.2f} GiB, "
        f"required {required_mib / 1024:.2f} GiB at gpu_memory_utilization={utilization:.2f}"
        for row, required_mib in failures
    )
    processes = _query_nvidia_smi_processes()
    same_model_processes = _find_same_model_gpu_processes(processes, selected, model_path)
    if same_model_processes:
        print(
            _format_same_model_already_loaded_message(
                matches=same_model_processes,
                model_path=model_path,
                port=port,
                config_path=config_path,
                run_id=run_id,
            ),
            flush=True,
        )
        return True
    process_summary = _format_gpu_processes(processes, selected)
    raise SystemExit(
        "Selected GPU(s) do not have enough free memory for vLLM startup.\n"
        f"{detail}\n\n"
        "This usually means another experiment is already occupying the selected GPU(s). "
        "No process was stopped. Use an idle node/GPU, wait for the current job to finish, "
        "or select different GPUs with CUDA_VISIBLE_DEVICES before running this wrapper.\n\n"
        "Selected GPU memory:\n"
        f"{selected_summary}\n\n"
        "All visible nvidia-smi GPUs:\n"
        f"{all_summary}\n\n"
        "Processes using selected GPU(s):\n"
        f"{process_summary}\n\n"
        "If you intentionally want to bypass only this preflight check, rerun with "
        "--skip-gpu-memory-preflight or set CCVF_SKIP_GPU_MEMORY_PREFLIGHT=1. "
        "That will not reduce memory use; vLLM may still fail if the GPUs are truly full."
    )


def _query_nvidia_smi_memory() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "total_mib": float(parts[3]),
                    "free_mib": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _query_nvidia_smi_processes() -> list[dict]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    processes = []
    pids = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            pid = int(parts[1])
            used_mib = float(parts[3])
        except ValueError:
            continue
        processes.append(
            {
                "gpu_uuid": parts[0],
                "pid": pid,
                "process_name": parts[2],
                "used_mib": used_mib,
            }
        )
        pids.append(pid)
    details = _query_process_details(pids)
    for process in processes:
        process.update(details.get(process["pid"], {}))
    return processes


def _query_process_details(pids: list[int]) -> dict[int, dict]:
    if not pids:
        return {}
    cmd = ["ps", "-o", "pid=,ppid=,user:32=,etime=,cmd=", "-p", ",".join(str(pid) for pid in sorted(set(pids)))]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    details: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        details[pid] = {"ppid": ppid, "user": parts[2], "etime": parts[3], "cmd": parts[4]}
    return details


def _query_process_tree_details() -> dict[int, dict]:
    cmd = ["ps", "-eo", "pid=,ppid=,user:32=,etime=,cmd="]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0:
        return {}
    details: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        details[pid] = {"ppid": ppid, "user": parts[2], "etime": parts[3], "cmd": parts[4]}
    return details


def _select_gpu_rows(rows: list[dict], tensor_parallel_size: int) -> list[dict]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.strip() and visible.strip() != "-1":
        tokens = [token.strip() for token in visible.split(",") if token.strip()]
        if all(token.isdigit() for token in tokens):
            visible_indices = [int(token) for token in tokens]
            by_index = {row["index"]: row for row in rows}
            return [by_index[index] for index in visible_indices[:tensor_parallel_size] if index in by_index]
        print(
            "Warning: CUDA_VISIBLE_DEVICES uses non-numeric GPU ids; GPU memory preflight will "
            "check the first nvidia-smi GPU rows.",
            file=sys.stderr,
            flush=True,
        )
    return rows[:tensor_parallel_size]


def _format_gpu_memory_rows(rows: list[dict]) -> str:
    if not rows:
        return "  <none>"
    return "\n".join(
        f"  GPU {row['index']}: {row['name']}, free {row['free_mib'] / 1024:.2f} / "
        f"{row['total_mib'] / 1024:.2f} GiB"
        for row in rows
    )


def _format_gpu_processes(processes: list[dict], selected_rows: list[dict]) -> str:
    selected_uuids = {row.get("uuid") for row in selected_rows}
    selected_indices = {row.get("uuid"): row.get("index") for row in selected_rows}
    selected = [process for process in processes if process.get("gpu_uuid") in selected_uuids]
    if not selected:
        return "  <none reported by nvidia-smi>"
    lines = []
    for process in selected:
        gpu_index = selected_indices.get(process.get("gpu_uuid"), "?")
        user = process.get("user", "?")
        etime = process.get("etime", "?")
        cmd = process.get("cmd") or process.get("process_name") or "?"
        if len(cmd) > 120:
            cmd = cmd[:117] + "..."
        lines.append(
            f"  GPU {gpu_index}: pid={process['pid']}, user={user}, used={process['used_mib'] / 1024:.2f} GiB, "
            f"elapsed={etime}, cmd={cmd}"
        )
    return "\n".join(lines)


def _find_same_model_gpu_processes(processes: list[dict], selected_rows: list[dict], model_path: Path) -> list[dict]:
    selected_uuids = {row.get("uuid") for row in selected_rows}
    selected_indices = {row.get("uuid"): row.get("index") for row in selected_rows}
    selected = [process for process in processes if process.get("gpu_uuid") in selected_uuids]
    if not selected:
        return []
    process_tree = _query_process_tree_details()
    requested_variants = _model_path_variants(model_path)
    matches = []
    matched_gpu_uuids = set()
    for process in selected:
        chain = _process_ancestor_chain(process["pid"], process_tree)
        command_text = "\n".join(item.get("cmd", "") for item in chain)
        if not any(variant in command_text for variant in requested_variants):
            continue
        gpu_uuid = process.get("gpu_uuid")
        matched_gpu_uuids.add(gpu_uuid)
        server = next((item for item in chain if "vllm" in item.get("cmd", "") and "serve" in item.get("cmd", "")), None)
        matches.append(
            {
                **process,
                "gpu_index": selected_indices.get(gpu_uuid, "?"),
                "server_pid": server.get("pid") if server else None,
                "server_cmd": server.get("cmd") if server else None,
            }
        )
    if matched_gpu_uuids != selected_uuids:
        return []
    return matches


def _process_ancestor_chain(pid: int, process_tree: dict[int, dict], max_depth: int = 16) -> list[dict]:
    chain = []
    seen = set()
    current = pid
    for _ in range(max_depth):
        if current in seen:
            break
        seen.add(current)
        detail = process_tree.get(current)
        if not detail:
            break
        chain.append({"pid": current, **detail})
        current = detail.get("ppid")
        if not current or current <= 1:
            break
    return chain


def _model_path_variants(model_path: Path) -> set[str]:
    variants = {str(model_path), model_path.name}
    try:
        variants.add(str(model_path.resolve()))
    except OSError:
        pass
    return {variant for variant in variants if variant}


def _format_same_model_already_loaded_message(
    *,
    matches: Sequence[dict],
    model_path: Path,
    port: int,
    config_path: Path,
    run_id: str,
) -> str:
    process_lines = []
    for match in matches:
        cmd = match.get("server_cmd") or match.get("cmd") or match.get("process_name") or "?"
        if len(cmd) > 140:
            cmd = cmd[:137] + "..."
        process_lines.append(
            f"  GPU {match.get('gpu_index', '?')}: worker_pid={match['pid']}, "
            f"server_pid={match.get('server_pid') or '?'}, used={match['used_mib'] / 1024:.2f} GiB, cmd={cmd}"
        )
    return (
        "The selected GPU(s) are already occupied by a vLLM process for this same model.\n"
        f"Model: {model_path}\n"
        f"Port requested: http://127.0.0.1:{port}/v1\n\n"
        "This is not a new memory problem. It means the model is already loaded, so starting "
        "serve_model again would try to load a second copy.\n\n"
        "Matching GPU processes:\n"
        + "\n".join(process_lines)
        + "\n\nUse a second tmux for the benchmark:\n"
        "  export LOCAL_OPENAI_API_KEY=EMPTY\n"
        f"  python scripts/run_ccvf.py run --config {config_path} --run-id {run_id}\n\n"
        "If the benchmark cannot connect, then the existing server is stale; stop only your own "
        "matching vLLM PIDs and start the server once."
    )


def _select_transformers_overlay(
    explicit_path: str | None,
    model_path: Path,
    disabled: bool,
    *,
    skip: bool,
    timeout: float,
) -> Path | None:
    if skip:
        return None
    ok, detail = _probe_transformers_config(model_path, overlay=None, timeout=timeout)
    if ok:
        return None
    original_detail = detail
    if disabled:
        raise SystemExit(
            "Installed Transformers cannot parse this checkpoint config, and the Transformers overlay "
            "was disabled with --no-transformers-overlay.\n"
            f"AutoConfig failure:\n{detail}"
        )
    candidates = [Path(explicit_path)] if explicit_path else []
    candidates.extend(DEFAULT_TRANSFORMERS_OVERLAYS)
    failures = []
    for path in candidates:
        if not (path / "transformers").exists():
            failures.append(f"{path}: missing transformers/ package")
            continue
        ok, detail = _probe_transformers_config(model_path, overlay=path, timeout=timeout)
        if ok:
            return path
        failures.append(f"{path}: {detail}")
    raise SystemExit(
        "Installed Transformers cannot parse this checkpoint config, and no usable Transformers "
        "overlay was found.\n"
        f"Original AutoConfig failure:\n{original_detail}\n\n"
        "Tried overlays:\n"
        + "\n".join(f"- {failure}" for failure in failures)
        + "\n\nSet CCVF_TRANSFORMERS_PATH or pass --transformers-path to a directory containing transformers/."
    )


def _probe_transformers_config(model_path: Path, overlay: Path | None, *, timeout: float) -> tuple[bool, str]:
    env = os.environ.copy()
    if overlay is not None:
        env["PYTHONPATH"] = f"{overlay}:{env.get('PYTHONPATH', '')}".rstrip(":")
    script = """
import sys
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print(type(cfg).__module__ + "." + type(cfg).__name__)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(model_path)],
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return False, f"AutoConfig probe timed out after {exc.timeout} seconds"
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    return proc.returncode == 0, output or f"AutoConfig probe exited with code {proc.returncode}"


def _preflight_vllm_architecture(model_config: dict, vllm_bin: str, allow_unsupported: bool) -> None:
    architectures = {str(item) for item in model_config.get("architectures") or []}
    unsupported_fallback = sorted(architectures & KNOWN_UNSUPPORTED_TRANSFORMERS_FALLBACK_ARCHITECTURES)
    if unsupported_fallback and not allow_unsupported:
        raise SystemExit(
            "This checkpoint uses Gemma4 Unified architecture:\n"
            f"  {', '.join(unsupported_fallback)}\n\n"
            "The installed vLLM 0.19.1 environment can parse this config, but its native model "
            "registry does not include Gemma4UnifiedForConditionalGeneration. vLLM then falls "
            "back to the Transformers multimodal path, which fails during engine warmup with:\n"
            "  ValueError: Expected hidden_size to be 3840, but found: 256\n\n"
            "This is a vLLM/Gemma4-Unified implementation incompatibility, not a GPU-memory, "
            "tensor-parallel, conda, or CCVF runner problem. Use a Gemma4 checkpoint whose "
            "architecture is Gemma4ForConditionalGeneration, such as configs/gemma_4_E2B_it.yaml, "
            "or use Gemma 3 / another supported checkpoint. Pass --allow-unsupported-architecture "
            "only for diagnostics if you intentionally want to reproduce the downstream vLLM failure."
        )
    unsupported = sorted(architectures & KNOWN_UNSUPPORTED_VLLM085_ARCHITECTURES)
    if not unsupported or allow_unsupported:
        return
    if not _is_vllm_085(vllm_bin):
        print(
            "Warning: this checkpoint uses a Gemma4 architecture. Make sure your vLLM build "
            "has explicit Gemma4 support before launching.",
            flush=True,
        )
        return
    raise SystemExit(
        "This checkpoint config can be parsed with the newer Transformers overlay, but vLLM 0.8.5 "
        "does not provide a model implementation for:\n"
        f"  {', '.join(unsupported)}\n\n"
        "That means the original AutoConfig error is not a GPU, tensor-parallel, or weight-loading "
        "problem; it happens before weights are loaded because the installed Transformers is too old. "
        "After fixing config parsing, vLLM 0.8.5 still cannot serve Gemma4/Gemma4Unified checkpoints. "
        "Use a newer vLLM release with Gemma4 support, a backend that supports this architecture, "
        "or a Gemma3/other supported checkpoint. Pass --allow-unsupported-architecture only for "
        "diagnostics if you intentionally want to see vLLM's downstream failure."
    )


def _is_vllm_085(vllm_bin: str) -> bool:
    if "vllm085" in str(vllm_bin):
        return True
    try:
        proc = subprocess.run(
            [vllm_bin, "--version"],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    output = f"{proc.stdout}\n{proc.stderr}"
    return "0.8.5" in output


if __name__ == "__main__":
    raise SystemExit(main())
