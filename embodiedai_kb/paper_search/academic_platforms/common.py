from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from hashlib import sha1
from typing import Any

from embodiedai_kb.storage.database import normalize_title


USER_AGENT = "EmbodiedAI-KB/0.1 academic-paper-search"


def extract_doi(text: str | None) -> str:
    if not text:
        return ""
    match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    return match.group(0).rstrip(".,;)") if match else ""


def stable_id(prefix: str, *values: str | None) -> str:
    for value in values:
        clean = (value or "").strip()
        if clean:
            return f"{prefix}:{clean}"
    digest = sha1(" ".join(v or "" for v in values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:title:{digest}"


def parse_date(value: str | None) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt, width in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(value[:width], fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def arxiv_id_from_url(url: str | None) -> str:
    value = (url or "").strip()
    match = re.search(
        r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
        value,
        flags=re.I,
    )
    if match:
        return match.group(1).removesuffix(".pdf")
    return ""


def arxiv_pdf_url(arxiv_id: str | None) -> str:
    clean = (arxiv_id or "").strip().removesuffix(".pdf")
    return f"https://arxiv.org/pdf/{clean}" if clean else ""


def doi_from_url(value: str | None) -> str:
    value = (value or "").strip()
    if value.lower().startswith("https://doi.org/"):
        return value[len("https://doi.org/") :]
    return extract_doi(value)


def is_probable_pdf_url(value: str | None) -> bool:
    url = (value or "").strip().lower()
    if not url:
        return False
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if "doi.org" in domain:
        return False
    if path.endswith((".mp4", ".mov", ".avi", ".zip", ".tar.gz")):
        return False
    if path.endswith(".pdf"):
        return True
    if "arxiv.org" in domain and "/pdf/" in path:
        return True
    if "openreview.net" in domain and "/pdf" in path:
        return True
    if any(marker in path for marker in ("/pdf", "download", "fulltext")):
        return True
    return False


def open_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    delay: float = 1.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay * (attempt + 1)
            except ValueError:
                wait = delay * (attempt + 1)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(delay * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("request failed without an exception")


def open_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    delay: float = 1.0,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise
            time.sleep(delay * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(delay * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("request failed without an exception")


def env_first(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return ""


def title_key(title: str) -> str:
    return normalize_title(title)


def content_value(content: dict[str, Any], key: str) -> Any:
    value = content.get(key)
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value
