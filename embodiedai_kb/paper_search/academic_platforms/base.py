from __future__ import annotations

from abc import ABC, abstractmethod

from embodiedai_kb.paper_search.models import AcademicPaper


class PaperSource(ABC):
    """Abstract base class for academic paper sources."""

    source_name: str

    @abstractmethod
    def search(self, query: str, max_results: int = 10, **kwargs: object) -> list[AcademicPaper]:
        """Search papers matching the query."""

    def download_pdf(self, paper: AcademicPaper, save_path: str) -> str:
        """Optional source-native PDF downloader."""

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support source-native PDF downloads."
        )
