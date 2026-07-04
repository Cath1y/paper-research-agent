"""Source-specific academic paper connectors."""

from .arxiv import ArxivSearcher
from .crossref import CrossrefSearcher
from .openalex import OpenAlexSearcher
from .openreview import OpenReviewSearcher
from .unpaywall import UnpaywallResolver

__all__ = [
    "ArxivSearcher",
    "CrossrefSearcher",
    "OpenAlexSearcher",
    "OpenReviewSearcher",
    "UnpaywallResolver",
]
