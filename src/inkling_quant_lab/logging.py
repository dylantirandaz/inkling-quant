"""Human-readable console and structured JSONL run events."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from inkling_quant_lab.security import Redactor


class EventLogger:
    """Append redacted events containing the required run identity fields."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        model_id: str,
        config_hash: str,
        redactor: Redactor | None = None,
        console: TextIO | None = sys.stderr,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.model_id = model_id
        self.config_hash = config_hash
        self.redactor = redactor or Redactor()
        self.console = console
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        *,
        stage: str,
        event: str,
        message: str,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append one structured event and an optional concise console line."""

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            "model_id": self.model_id,
            "config_hash": self.config_hash,
            "level": level,
            "event": event,
            "message": self.redactor.text(message),
            "data": self.redactor.value(data or {}),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
        if self.console is not None:
            print(f"[{level.upper()}] {stage}: {record['message']}", file=self.console)
