"""Academic paper search connectors and OA-first PDF resolution."""

from .models import AcademicPaper
from .tool import run_academic_paper_search

__all__ = ["AcademicPaper", "run_academic_paper_search"]
