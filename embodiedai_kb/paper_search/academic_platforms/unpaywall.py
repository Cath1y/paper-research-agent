from __future__ import annotations

import urllib.parse

from .common import env_first, is_probable_pdf_url, open_json


class UnpaywallResolver:
    """Resolve open-access PDF URLs by DOI using Unpaywall."""

    base_url = "https://api.unpaywall.org/v2"

    def __init__(self, email: str | None = None) -> None:
        self.email = (email or env_first("PAPER_SEARCH_MCP_UNPAYWALL_EMAIL", "UNPAYWALL_EMAIL")).strip()

    def has_api_access(self) -> bool:
        return bool(self.email)

    def resolve_best_pdf_url(self, doi: str) -> str:
        clean = (doi or "").strip()
        if not clean or not self.email:
            return ""
        quoted = urllib.parse.quote(clean, safe="")
        payload = open_json(
            f"{self.base_url}/{quoted}?{urllib.parse.urlencode({'email': self.email})}",
            retries=1,
        )
        best = payload.get("best_oa_location") or {}
        for location in [best, *(payload.get("oa_locations") or [])]:
            if not isinstance(location, dict):
                continue
            pdf_url = location.get("url_for_pdf") or ""
            if not pdf_url and is_probable_pdf_url(location.get("url")):
                pdf_url = location.get("url") or ""
            if pdf_url:
                return str(pdf_url)
        return ""
