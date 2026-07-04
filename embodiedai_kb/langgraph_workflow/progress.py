from __future__ import annotations

import sys
from datetime import datetime
from textwrap import shorten
from typing import Any


def emit_progress(
    args: Any,
    label: str,
    message: str,
    **fields: Any,
) -> None:
    """Print one concise live progress line for long-running workflow steps."""

    if getattr(args, "quiet_workflow_progress", False):
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    field_text = " ".join(
        f"{key}={shorten(str(value), width=90, placeholder='...')}"
        for key, value in fields.items()
        if value is not None
    )
    suffix = f" | {field_text}" if field_text else ""
    print(f"[{timestamp}] [{label}] {message}{suffix}", file=sys.stderr, flush=True)
