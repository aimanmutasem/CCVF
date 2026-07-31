from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ccvf.schemas import DatasetSample


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: str | Path, *, allow_non_json_lines: bool = False, limit: int | None = None) -> list[DatasetSample]:
    path = Path(path)
    samples: list[DatasetSample] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                if allow_non_json_lines:
                    continue
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            try:
                samples.append(DatasetSample.model_validate(raw))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}:{line_no}: {exc}")
            if limit is not None and len(samples) >= limit:
                break
    if errors:
        raise ValueError("\n".join(errors[:20]))
    return samples


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

