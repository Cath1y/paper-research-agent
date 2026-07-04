from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from textwrap import shorten
from typing import Any


def _clean_text(value: Any, *, width: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    return shorten(text, width=width, placeholder="...")


def _stable_hash(*parts: Any, length: int = 12) -> str:
    raw = "\n".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _safe_thread_id(value: Any) -> str:
    thread_id = re.sub(r"[^\w_.-]+", "_", str(value or "default")).strip("._-")
    return thread_id or "default"


def memory_path(args: Any) -> Path:
    memory_dir = Path(getattr(args, "memory_dir", "data/memory"))
    return memory_dir / f"{_safe_thread_id(getattr(args, 'thread_id', 'default'))}.jsonl"


def memory_root_dir(args: Any) -> Path:
    return Path(getattr(args, "memory_dir", "data/memory"))


def legacy_memory_store_dir(args: Any) -> Path:
    return memory_root_dir(args) / "store"


def global_memory_store_dir(args: Any) -> Path:
    return memory_root_dir(args) / "global"


def memory_store_dir(args: Any) -> Path:
    """Return the current thread's detailed memory store directory."""

    return memory_root_dir(args) / "threads" / _safe_thread_id(
        getattr(args, "thread_id", "default")
    ) / "store"


GLOBAL_STORE_FILE_TEMPLATES = {
    "IDENTITY.md": """# Identity Memory

- Role: Embodied AI literature research and learning assistant.
- Style: Evidence-driven, concise, and explicit about source types.
- Constraints:
  - Do not treat memory as PaperQA/Web evidence.
  - Prefer planner/LLM decisions over hard-coded heuristic shortcuts.
  - When evidence is missing, say so instead of filling gaps.
""",
    "MEMORY.md": """# Long-Term Semantic Memory

Use this file for stable user/project facts that should persist across threads.

## User Preferences

- The user prefers evidence-driven literature research.
- The user dislikes brittle heuristic shortcuts for search/routing decisions.

## Project Facts

- The project uses LangGraph for global plan-and-execute orchestration.
- PaperSearchAgent and PaperQA reader provide local ReAct-like search/evidence loops.
""",
    "LESSONS.md": """# Experience Memory

Use this file for durable lessons about what worked or failed. These are suggestions for the LLM, not hard-coded rules.

- Author/team research questions usually benefit from profile/publication pages before academic database fallback.
- Academic database results must be checked for author identity before sending PDFs to PaperQA.
""",
}

THREAD_STORE_FILE_TEMPLATES = {
    "PAPERS.md": """# Paper Memory

Each run appends one Paper Card Collection. A collection contains the papers
selected/read in that episode, plus matched PaperQA evidence when available.
""",
    "EPISODES.md": """# Episode Memory

Detailed summaries of completed runs are appended here. Thread JSONL files remain the richer debug log.
""",
}

INDEX_FILE_NAMES = ("EPISODE_INDEX.jsonl", "PAPER_CARDS.jsonl")


def _ensure_text_file(path: Path, template: str, *, legacy_path: Path | None = None) -> None:
    if path.exists():
        return
    if legacy_path and legacy_path.exists():
        text = legacy_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            path.write_text(text.rstrip() + "\n", encoding="utf-8")
            return
    path.write_text(template.rstrip() + "\n", encoding="utf-8")


def ensure_memory_store(args: Any) -> dict[str, str]:
    global_dir = global_memory_store_dir(args)
    thread_dir = memory_store_dir(args)
    legacy_dir = legacy_memory_store_dir(args)
    global_dir.mkdir(parents=True, exist_ok=True)
    thread_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for filename, template in GLOBAL_STORE_FILE_TEMPLATES.items():
        path = global_dir / filename
        paths[f"global/{filename}"] = str(path)
        _ensure_text_file(path, template, legacy_path=legacy_dir / filename)
    for filename, template in THREAD_STORE_FILE_TEMPLATES.items():
        path = thread_dir / filename
        paths[f"thread/{filename}"] = str(path)
        _ensure_text_file(path, template)
    for filename in INDEX_FILE_NAMES:
        path = thread_dir / filename
        paths[f"thread/{filename}"] = str(path)
        if not path.exists():
            path.write_text("", encoding="utf-8")
    return paths


def _read_store_file(path: Path, *, width: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return ""
    return shorten(text, width=width, placeholder="\n...")


def _read_jsonl_tail(path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records[-limit:]


def load_memory_store(args: Any) -> dict[str, Any]:
    paths = ensure_memory_store(args)
    global_dir = global_memory_store_dir(args)
    thread_dir = memory_store_dir(args)
    return {
        "kind": "global_plus_thread_memory_store",
        "global_store_dir": str(global_dir),
        "thread_store_dir": str(thread_dir),
        "store_dir": str(thread_dir),
        "paths": paths,
        "identity": _read_store_file(global_dir / "IDENTITY.md", width=1400),
        "semantic_memory": _read_store_file(global_dir / "MEMORY.md", width=1800),
        "lessons": _read_store_file(global_dir / "LESSONS.md", width=1600),
        "paper_memory": _read_store_file(thread_dir / "PAPERS.md", width=2200),
        "episode_index": _read_jsonl_tail(thread_dir / "EPISODE_INDEX.jsonl", limit=20),
        "paper_cards": _read_jsonl_tail(thread_dir / "PAPER_CARDS.jsonl", limit=30),
    }


def _append_episode_markdown(record: dict[str, Any], args: Any) -> None:
    store_dir = memory_store_dir(args)
    path = store_dir / "EPISODES.md"
    episode_id = _episode_id(record)
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if f"episode_id: {episode_id}" in existing:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_episode_markdown_block(record) + "\n")


def _episode_id(record: dict[str, Any]) -> str:
    explicit = str(record.get("episode_id") or "").strip()
    if explicit:
        return explicit
    return _stable_hash(record.get("thread_id"), record.get("created_at"), record.get("question"))


def _episode_markdown_block(record: dict[str, Any]) -> str:
    created_at = _clean_text(record.get("created_at"), width=40)
    question = _clean_text(record.get("question"), width=240)
    summary = _clean_text(record.get("episode_summary"), width=900)
    answer_detail = _clean_text(record.get("answer_excerpt"), width=4500)
    task = record.get("task_type") or record.get("task_types") or "unknown"
    selected_count = len(record.get("selected_papers") or [])
    web_count = len(record.get("web_sources") or [])
    paperqa = record.get("paperqa") or {}
    queries = [str(query) for query in (record.get("queries") or [])[:8]]
    perspectives = record.get("research_perspectives") or []
    selected_papers = record.get("selected_papers") or []
    web_sources = record.get("web_sources") or []

    lines = [
        "",
        f"<!-- episode_id: {_episode_id(record)} -->",
        f"## {created_at} | thread={record.get('thread_id')} | episode={_episode_id(record)}",
        f"- question: {question}",
        f"- task: {task}",
        f"- selected_papers: {selected_count}",
        f"- web_sources: {web_count}",
        f"- paperqa: mode={paperqa.get('mode') or 'N/A'} evidence_count={paperqa.get('evidence_count')} status={_clean_text(paperqa.get('status'), width=180) or 'N/A'}",
        f"- answer_chars: {record.get('answer_chars') or len(str(record.get('answer_excerpt') or ''))}",
    ]
    if summary:
        lines.append(f"- episode_summary: {summary}")
    if perspectives:
        lines.append("- perspectives:")
        for item in perspectives[:8]:
            signal_text = ", ".join(str(x) for x in (item.get("signal_sources") or [])[:5])
            suffix = f" [signals={signal_text}]" if signal_text else ""
            lines.append(
                f"  - {_clean_text(item.get('perspective'), width=80)}: "
                f"{_clean_text(item.get('research_question'), width=220)}{suffix}"
            )
    if queries:
        lines.append("- queries:")
        lines.extend(f"  - {_clean_text(query, width=180)}" for query in queries)
    if selected_papers:
        lines.append("- selected_papers:")
        for paper in selected_papers[:10]:
            title = _clean_text(paper.get("title"), width=160)
            year = _clean_text(paper.get("year"), width=12) or "?"
            venue = _clean_text(paper.get("venue"), width=70) or "?"
            paper_url = _clean_text(paper.get("paper_url"), width=160)
            pdf_url = _clean_text(paper.get("pdf_url"), width=160)
            lines.append(f"  - {title} ({year}, {venue})")
            if paper_url:
                lines.append(f"    paper_url: {paper_url}")
            if pdf_url:
                lines.append(f"    pdf_url: {pdf_url}")
    if web_sources:
        lines.append("- web_sources:")
        for source in web_sources[:8]:
            lines.append(
                f"  - {_clean_text(source.get('title'), width=120)} | "
                f"{_clean_text(source.get('url'), width=180)}"
            )
    lines.extend(
        [
            "- answer_detail:",
            f"  {answer_detail}",
        ]
    )
    return "\n".join(lines)


def _append_papers_markdown(record: dict[str, Any], args: Any) -> int:
    papers = record.get("selected_papers") or []
    if not papers:
        return 0
    store_dir = memory_store_dir(args)
    path = store_dir / "PAPERS.md"
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    block, collection_id, card_count = _paper_card_collection_markdown(record)
    if not block or f"paper_collection_id: {collection_id}" in existing:
        return 0
    with path.open("a", encoding="utf-8") as handle:
        handle.write(block + "\n")
    return card_count


def _markdown_values(values: list[Any], *, limit: int = 8, width: int = 120) -> str:
    texts = [
        _clean_text(value, width=width)
        for value in values[:limit]
        if _clean_text(value, width=width)
    ]
    return ", ".join(texts) or "N/A"


def _paper_card_collection_markdown(record: dict[str, Any]) -> tuple[str, str, int]:
    papers = [paper for paper in (record.get("selected_papers") or []) if paper.get("title")]
    if not papers:
        return "", "", 0
    episode_id = _episode_id(record)
    collection_id = _stable_hash(episode_id, "paper_card_collection")
    created_at = _clean_text(record.get("created_at"), width=40)
    source_question = _clean_text(record.get("question"), width=260)
    episode_summary = _clean_text(record.get("episode_summary"), width=700)
    related_ids = [_paper_id(paper) for paper in papers]
    cards = [
        _build_paper_card(
            paper,
            record,
            index=index,
            related_paper_ids=related_ids,
        )
        for index, paper in enumerate(papers)
    ]

    lines = [
        "",
        f"<!-- paper_collection_id: {collection_id} -->",
        f"## Paper Card Collection | {created_at} | episode={episode_id}",
        f"- source_thread: {record.get('thread_id')}",
        f"- source_question: {source_question}",
        f"- paper_count: {len(cards)}",
    ]
    if episode_summary:
        lines.append(f"- episode_summary: {episode_summary}")
    for index, card in enumerate(cards, start=1):
        lines.extend(_paper_card_markdown_lines(card, index=index))
    return "\n".join(lines), collection_id, len(cards)


def _paper_card_markdown_lines(card: dict[str, Any], *, index: int) -> list[str]:
    evidence = card.get("evidence_snippets") or []
    evidence_ids = card.get("evidence_ids") or []
    lines = [
        "",
        f"### {index}. {_clean_text(card.get('title'), width=220) or 'Untitled'}",
        f"- card_id: {card.get('card_id')}",
        f"- paper_id: {card.get('paper_id')}",
        f"- short_name: {_clean_text(card.get('short_name'), width=120) or 'N/A'}",
        f"- authors: {_markdown_values(card.get('authors') or [], limit=10, width=90)}",
        f"- year: {_clean_text(card.get('year'), width=12) or 'N/A'}",
        f"- venue: {_clean_text(card.get('venue'), width=80) or 'N/A'}",
        f"- status: {card.get('status') or 'N/A'}",
        f"- role: {card.get('role') or 'N/A'}",
        f"- confidence: {card.get('confidence') or 'N/A'}",
        f"- source_question: {_clean_text(card.get('source_question'), width=240)}",
        f"- summary: {_clean_text(card.get('summary'), width=520) or 'N/A'}",
        f"- research_problem: {_clean_text(card.get('research_problem'), width=360) or 'N/A'}",
        f"- core_idea: {_clean_text(card.get('core_idea'), width=700) or 'N/A'}",
        f"- method_keywords: {_markdown_values(card.get('method_keywords') or [], limit=16, width=90)}",
        f"- key_results: {_markdown_values(card.get('key_results') or [], limit=5, width=180)}",
        f"- limitations: {_markdown_values(card.get('limitations') or [], limit=5, width=180)}",
        f"- paper_url: {_clean_text(card.get('paper_url'), width=220) or 'N/A'}",
        f"- pdf_url: {_clean_text(card.get('pdf_url'), width=220) or 'N/A'}",
        f"- cache_status: {_clean_text(card.get('cache_status'), width=60) or 'N/A'}",
        f"- cache_path: {_clean_text(card.get('cache_path'), width=260) or 'N/A'}",
        f"- sources: {_markdown_values(card.get('sources') or [], limit=8, width=90)}",
        f"- evidence_ids: {_markdown_values(evidence_ids, limit=12, width=120)}",
    ]
    if evidence:
        lines.append("- paperqa_evidence:")
        for item in evidence[:6]:
            lines.append(
                f"  - evidence_id: {item.get('evidence_id')}; "
                f"citation: {_clean_text(item.get('citation'), width=180) or 'N/A'}; "
                f"question: {_clean_text(item.get('question'), width=180) or 'N/A'}; "
                f"score: {item.get('score')}; "
                f"snippet: {_clean_text(item.get('snippet'), width=420) or 'N/A'}"
            )
    else:
        lines.append("- paperqa_evidence: N/A")
    return lines


def _jsonl_contains_id(path: Path, key: str, value: str) -> bool:
    if not path.exists() or not value:
        return False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and str(payload.get(key) or "") == value:
            return True
    return False


def _append_jsonl_unique(path: Path, payload: dict[str, Any], *, key: str) -> bool:
    value = str(payload.get(key) or "").strip()
    if not value:
        return False
    if _jsonl_contains_id(path, key, value):
        return False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return True


def _iso_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("T", 1)[0]


def _sentence_candidates(text: Any) -> list[str]:
    cleaned = _clean_text(text, width=1600)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+", cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]


def _first_sentence(text: Any, *, width: int = 260) -> str:
    sentences = _sentence_candidates(text)
    return _clean_text(sentences[0] if sentences else text, width=width)


def _result_sentences(text: Any, *, limit: int = 3) -> list[str]:
    markers = (
        "achieve",
        "achieves",
        "demonstrate",
        "demonstrates",
        "show",
        "shows",
        "outperform",
        "outperforms",
        "improve",
        "improves",
        "report",
        "reports",
        "提升",
        "优于",
        "证明",
        "结果",
    )
    results: list[str] = []
    for sentence in _sentence_candidates(text):
        lower = sentence.lower()
        if any(marker in lower for marker in markers):
            results.append(_clean_text(sentence, width=260))
        if len(results) >= limit:
            break
    return results


def _paper_short_name(title: Any) -> str:
    text = str(title or "").strip()
    if not text:
        return ""
    head = re.split(r"[:：]", text, maxsplit=1)[0].strip()
    return _clean_text(head or text, width=80)


def _safe_id_fragment(value: Any, *, fallback: str) -> str:
    text = str(value or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    fragment = "_".join(tokens[:5])
    return fragment or fallback


def _paper_id(paper: dict[str, Any]) -> str:
    explicit = str(paper.get("paper_id") or "").strip()
    if explicit:
        return _safe_id_fragment(explicit, fallback=f"paper_{_stable_hash(explicit)}")
    title = str(paper.get("title") or "").strip()
    url = str(paper.get("paper_url") or paper.get("pdf_url") or "").strip()
    fragment = _safe_id_fragment(_paper_short_name(title), fallback="paper")
    return f"paper_{fragment}_{_stable_hash(title, url, length=8)}"


def _paper_card_id(paper: dict[str, Any], record: dict[str, Any]) -> str:
    return f"card_{_stable_hash(_episode_id(record), _paper_id(paper), length=14)}"


def _paper_status(paper: dict[str, Any], evidence_ids: list[str]) -> str:
    if evidence_ids:
        return "read_pdf"
    cache_status = str(paper.get("cache_status") or "").strip().lower()
    if cache_status in {"downloaded", "cached"}:
        return "downloaded"
    if paper.get("pdf_url"):
        return "selected"
    return "candidate"


def _paper_confidence(paper: dict[str, Any], evidence_ids: list[str]) -> str:
    if evidence_ids:
        return "high"
    if paper.get("abstract") and paper.get("pdf_url"):
        return "medium"
    return "low"


def _paper_role(index: int, evidence_ids: list[str]) -> str:
    if index == 0 and evidence_ids:
        return "core_read_paper"
    if evidence_ids:
        return "supporting_read_paper"
    return "selected_candidate"


def _evidence_id(record: dict[str, Any], evidence: dict[str, Any]) -> str:
    raw = evidence.get("id")
    if raw:
        return f"ev_{_safe_id_fragment(raw, fallback=_stable_hash(raw, length=10))}"
    return f"ev_{_stable_hash(_episode_id(record), evidence.get('citation'), evidence.get('context'), length=12)}"


def _paper_evidence_records(
    paper: dict[str, Any],
    record: dict[str, Any],
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    evidence_contexts = record.get("paperqa_evidence_contexts") or []
    if not evidence_contexts:
        return []
    title = str(paper.get("title") or "")
    short_name = _paper_short_name(title).lower()
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", title.lower())
        if len(token) >= 4
    }
    matched: list[dict[str, Any]] = []
    for evidence in evidence_contexts:
        haystack = " ".join(
            str(evidence.get(key) or "")
            for key in ("citation", "docname", "dockey", "text_name", "context", "question")
        ).lower()
        token_hits = sum(1 for token in title_tokens if token in haystack)
        if (short_name and short_name in haystack) or token_hits >= 2:
            evidence_id = _evidence_id(record, evidence)
            matched.append(
                {
                    "evidence_id": evidence_id,
                    "citation": _clean_text(evidence.get("citation"), width=220),
                    "question": _clean_text(evidence.get("question"), width=220),
                    "score": evidence.get("score"),
                    "snippet": _clean_text(evidence.get("context"), width=520),
                }
            )
        if len(matched) >= limit:
            break
    return matched


def _episode_topics(record: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for item in record.get("research_perspectives") or []:
        topic = _clean_text(item.get("perspective"), width=80)
        if topic and topic not in topics:
            topics.append(topic)
    if not topics:
        for task in record.get("task_types") or []:
            topic = _clean_text(task, width=80)
            if topic and topic not in topics:
                topics.append(topic)
    return topics[:12]


def _build_episode_index(record: dict[str, Any]) -> dict[str, Any]:
    episode_id = _episode_id(record)
    papers = record.get("selected_papers") or []
    paper_ids = [_paper_id(paper) for paper in papers if paper.get("title")]
    return {
        "episode_id": episode_id,
        "thread_id": record.get("thread_id"),
        "created_at": record.get("created_at"),
        "question": record.get("question"),
        "task_type": record.get("task_type"),
        "task_types": record.get("task_types") or [],
        "summary": _clean_text(
            record.get("episode_summary") or record.get("answer_excerpt"),
            width=900,
        ),
        "topics": _episode_topics(record),
        "queries": [str(query) for query in (record.get("queries") or [])[:12]],
        "selected_paper_ids": paper_ids,
        "selected_paper_titles": [
            _clean_text(paper.get("title"), width=180)
            for paper in papers[:12]
            if paper.get("title")
        ],
        "paper_card_ids": [_paper_card_id(paper, record) for paper in papers if paper.get("title")],
        "web_source_count": len(record.get("web_sources") or []),
        "paperqa": record.get("paperqa") or {},
        "answer_chars": record.get("answer_chars"),
        "trace_nodes": record.get("trace_nodes") or [],
        "detail_ref": {
            "path": "EPISODES.md",
            "episode_id": episode_id,
        },
    }


def _build_paper_card(
    paper: dict[str, Any],
    record: dict[str, Any],
    *,
    index: int,
    related_paper_ids: list[str],
) -> dict[str, Any]:
    evidence = _paper_evidence_records(paper, record)
    evidence_ids = [item["evidence_id"] for item in evidence]
    abstract = paper.get("abstract") or ""
    evidence_summary = " ".join(str(item.get("snippet") or "") for item in evidence[:2])
    summary = _clean_text(abstract, width=520) or _clean_text(evidence_summary, width=520)
    method_keywords = []
    for value in [
        *(paper.get("keywords") or []),
        *(paper.get("categories") or []),
        *(paper.get("quality_signals") or []),
        *(paper.get("sources") or []),
    ]:
        text = _clean_text(value, width=80)
        if text and text not in method_keywords:
            method_keywords.append(text)
    return {
        "card_id": _paper_card_id(paper, record),
        "paper_id": _paper_id(paper),
        "episode_id": _episode_id(record),
        "title": paper.get("title"),
        "short_name": _paper_short_name(paper.get("title")),
        "authors": paper.get("authors") or [],
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "status": _paper_status(paper, evidence_ids),
        "role": _paper_role(index, evidence_ids),
        "source_question": record.get("question"),
        "summary": summary,
        "research_problem": _first_sentence(abstract, width=320),
        "core_idea": _clean_text(abstract, width=620),
        "method_keywords": method_keywords[:16],
        "key_results": _result_sentences(abstract, limit=3),
        "limitations": [],
        "evidence_ids": evidence_ids,
        "evidence_snippets": evidence,
        "related_papers": [
            paper_id
            for paper_id in related_paper_ids
            if paper_id != _paper_id(paper)
        ][:8],
        "paper_url": paper.get("paper_url"),
        "pdf_url": paper.get("pdf_url"),
        "cache_status": paper.get("cache_status"),
        "cache_path": paper.get("cache_path"),
        "sources": paper.get("sources") or [],
        "last_updated": _iso_date(record.get("created_at")),
        "confidence": _paper_confidence(paper, evidence_ids),
    }


def _append_episode_index(record: dict[str, Any], args: Any) -> int:
    path = memory_store_dir(args) / "EPISODE_INDEX.jsonl"
    payload = _build_episode_index(record)
    return 1 if _append_jsonl_unique(path, payload, key="episode_id") else 0


def _append_paper_cards(record: dict[str, Any], args: Any) -> int:
    papers = [paper for paper in (record.get("selected_papers") or []) if paper.get("title")]
    if not papers:
        return 0
    path = memory_store_dir(args) / "PAPER_CARDS.jsonl"
    related_ids = [_paper_id(paper) for paper in papers]
    appended = 0
    for index, paper in enumerate(papers):
        card = _build_paper_card(
            paper,
            record,
            index=index,
            related_paper_ids=related_ids,
        )
        if _append_jsonl_unique(path, card, key="card_id"):
            appended += 1
    return appended


def append_memory_store_record(record: dict[str, Any], args: Any) -> dict[str, Any]:
    paths = ensure_memory_store(args)
    _append_episode_markdown(record, args)
    paper_count = _append_papers_markdown(record, args)
    episode_index_count = _append_episode_index(record, args)
    paper_card_count = _append_paper_cards(record, args)
    return {
        "mode": "markdown_store",
        "global_store_dir": str(global_memory_store_dir(args)),
        "thread_store_dir": str(memory_store_dir(args)),
        "store_dir": str(memory_store_dir(args)),
        "paths": paths,
        "episode_written": True,
        "paper_memory_appended": paper_count,
        "episode_index_appended": episode_index_count,
        "paper_cards_appended": paper_card_count,
    }


def sync_memory_store_from_records(records: list[dict[str, Any]], args: Any) -> dict[str, Any]:
    ensure_memory_store(args)
    episode_count = 0
    paper_count = 0
    episode_index_count = 0
    paper_card_count = 0
    for record in records:
        before_papers = _append_papers_markdown(record, args)
        _append_episode_markdown(record, args)
        episode_index_count += _append_episode_index(record, args)
        paper_card_count += _append_paper_cards(record, args)
        episode_count += 1
        paper_count += before_papers
    return {
        "mode": "markdown_store_sync",
        "global_store_dir": str(global_memory_store_dir(args)),
        "thread_store_dir": str(memory_store_dir(args)),
        "store_dir": str(memory_store_dir(args)),
        "record_count": len(records),
        "episode_seen": episode_count,
        "paper_memory_appended": paper_count,
        "episode_index_appended": episode_index_count,
        "paper_cards_appended": paper_card_count,
    }


def load_thread_memory(args: Any) -> tuple[list[dict[str, Any]], str, dict[str, Any], dict[str, Any]]:
    """Load recent thread records from local JSONL memory.

    Thread memory is intentionally compact when formatted for prompts. The JSONL
    file keeps richer debug records, while memory_context only helps resolve
    follow-ups such as "this paper" or "continue that direction".
    """

    path = memory_path(args)
    trace: dict[str, Any] = {
        "mode": "disabled" if getattr(args, "no_memory", False) else "jsonl",
        "thread_id": _safe_thread_id(getattr(args, "thread_id", "default")),
        "path": str(path),
        "loaded_count": 0,
        "recent_turns": int(getattr(args, "memory_recent_turns", 5) or 5),
        "reset": bool(getattr(args, "reset_memory", False)),
    }
    if getattr(args, "no_memory", False):
        return [], "", trace, {}

    store = load_memory_store(args)
    if getattr(args, "reset_memory", False) and path.exists():
        path.unlink()
        trace["reset_deleted"] = True

    if not path.exists():
        trace["status"] = "missing"
        packet = build_memory_packet([], store=store)
        trace["packet"] = {
            "recent_episode_count": len(packet.get("recent_episodes") or []),
            "latest_selected_paper_count": len(
                packet.get("latest_paper_episode_selected_papers") or []
            ),
            "recent_selected_paper_count": len(packet.get("recent_selected_papers") or []),
        }
        trace["store"] = {
            "mode": store.get("kind"),
            "global_store_dir": store.get("global_store_dir"),
            "thread_store_dir": store.get("thread_store_dir"),
        }
        return [], format_memory_context([], packet=packet), trace, packet

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)

    recent_count = max(0, int(getattr(args, "memory_recent_turns", 5) or 5))
    recent = records[-recent_count:] if recent_count else []
    sync_trace = sync_memory_store_from_records(records, args)
    store = load_memory_store(args)
    trace["loaded_count"] = len(records)
    trace["used_count"] = len(recent)
    trace["status"] = "loaded"
    packet = build_memory_packet(recent, store=store)
    trace["packet"] = {
        "recent_episode_count": len(packet.get("recent_episodes") or []),
        "latest_selected_paper_count": len(packet.get("latest_paper_episode_selected_papers") or []),
        "recent_selected_paper_count": len(packet.get("recent_selected_papers") or []),
    }
    trace["store"] = {
        "mode": store.get("kind"),
        "global_store_dir": store.get("global_store_dir"),
        "thread_store_dir": store.get("thread_store_dir"),
        "sync": sync_trace,
    }
    return recent, format_memory_context(recent, packet=packet), trace, packet


def _paper_briefs(items: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    briefs: list[dict[str, Any]] = []
    for item in items[:limit]:
        result = item.get("result", item)
        if not isinstance(result, dict):
            continue
        briefs.append(
            {
                "paper_id": result.get("paper_id"),
                "title": result.get("title"),
                "authors": (result.get("authors") or [])[:8],
                "year": result.get("year"),
                "venue": result.get("venue") or result.get("corpus"),
                "paper_url": result.get("paper_url"),
                "pdf_url": result.get("pdf_url"),
                "sources": result.get("sources") or [],
                "abstract": _clean_text(result.get("abstract"), width=1200),
                "keywords": result.get("keywords") or [],
                "categories": result.get("categories") or [],
                "quality_signals": result.get("quality_signals") or [],
                "cache_status": item.get("cache_status"),
                "cache_path": item.get("cache_path"),
                "selection_score": item.get("selection_score"),
                "decision": result.get("decision"),
                "citation_count": result.get("citation_count"),
                "influential_citation_count": result.get("influential_citation_count"),
            }
        )
    return briefs


def _web_briefs(items: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    briefs: list[dict[str, Any]] = []
    for item in items[:limit]:
        briefs.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": _clean_text(item.get("snippet"), width=280),
            }
        )
    return briefs


def _paperqa_evidence_briefs(
    paperqa_trace: dict[str, Any],
    *,
    limit: int = 32,
) -> list[dict[str, Any]]:
    evidence_contexts = paperqa_trace.get("evidence_contexts") or []
    briefs: list[dict[str, Any]] = []
    for item in evidence_contexts[:limit]:
        if not isinstance(item, dict):
            continue
        briefs.append(
            {
                "id": item.get("id"),
                "score": item.get("score"),
                "question": _clean_text(item.get("question"), width=260),
                "context": _clean_text(item.get("context"), width=900),
                "text_name": item.get("text_name"),
                "docname": item.get("docname"),
                "dockey": item.get("dockey"),
                "citation": _clean_text(item.get("citation"), width=320),
            }
        )
    return briefs


def build_memory_record(state: dict[str, Any], args: Any) -> dict[str, Any]:
    paperqa_trace = state.get("paperqa_trace") or {}
    route = state.get("route") or {}
    selected = state.get("selected") or []
    final_answer = state.get("final_answer") or ""
    synthesis_trace = state.get("synthesis_trace") or {}
    episode_summary = (
        state.get("episode_summary")
        or synthesis_trace.get("episode_summary")
        or ""
    )
    research_plan = state.get("research_plan") or []
    queries = state.get("queries") or []
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": _safe_thread_id(getattr(args, "thread_id", "default")),
        "question": state.get("question"),
        "task_type": route.get("task_type"),
        "task_types": route.get("task_types") or [],
        "episode_summary": _clean_text(episode_summary, width=1200),
        "answer_full": final_answer,
        "answer_excerpt": _clean_text(final_answer, width=4500),
        "answer_chars": len(final_answer),
        "research_perspectives": [
            {
                "perspective": item.get("perspective"),
                "research_question": item.get("research_question") or item.get("question"),
                "signal_sources": item.get("signal_sources") or [],
            }
            for item in research_plan[:8]
        ],
        "queries": [str(query) for query in queries[:12]],
        "selected_papers": _paper_briefs(selected, limit=10),
        "paperqa": {
            "status": paperqa_trace.get("status"),
            "mode": paperqa_trace.get("mode"),
            "evidence_count": paperqa_trace.get("evidence_count"),
            "relevant_paper_count": paperqa_trace.get("relevant_paper_count"),
            "sufficient": paperqa_trace.get("sufficient"),
            "adapter_mode": paperqa_trace.get("adapter_mode"),
            "answer_mode": paperqa_trace.get("answer_mode"),
        },
        "paperqa_evidence_contexts": _paperqa_evidence_briefs(paperqa_trace),
        "web_sources": _web_briefs(state.get("web_evidence") or [], limit=8),
        "trace_nodes": [step.get("node") for step in state.get("trace", []) if isinstance(step, dict)],
    }
    record["episode_id"] = _episode_id(record)
    return record


def append_thread_memory(state: dict[str, Any], args: Any) -> dict[str, Any]:
    path = memory_path(args)
    trace: dict[str, Any] = {
        "mode": "disabled" if getattr(args, "no_memory", False) else "jsonl",
        "thread_id": _safe_thread_id(getattr(args, "thread_id", "default")),
        "path": str(path),
        "written": False,
    }
    if getattr(args, "no_memory", False):
        return trace

    record = build_memory_record(state, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    store_trace = append_memory_store_record(record, args)
    trace.update(
        {
            "written": True,
            "question": record.get("question"),
            "selected_paper_count": len(record.get("selected_papers") or []),
            "answer_excerpt_chars": len(record.get("answer_excerpt") or ""),
            "store": store_trace,
        }
    )
    return trace


def _episode_brief(record: dict[str, Any]) -> dict[str, Any]:
    summary = record.get("episode_summary") or record.get("answer_excerpt")
    return {
        "created_at": record.get("created_at"),
        "question": record.get("question"),
        "task_type": record.get("task_type"),
        "task_types": record.get("task_types") or [],
        "summary": _clean_text(summary, width=320),
        "answer_hint": _clean_text(summary, width=220),
        "selected_paper_count": len(record.get("selected_papers") or []),
        "web_source_count": len(record.get("web_sources") or []),
        "paperqa": record.get("paperqa") or {},
    }


def _dedupe_papers_by_title(records: list[dict[str, Any]], *, limit: int = 12) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for record in reversed(records):
        for paper in record.get("selected_papers") or []:
            title = _clean_text(paper.get("title"), width=180)
            key = title.lower()
            if not title or key in seen_titles:
                continue
            seen_titles.add(key)
            item = dict(paper)
            item["source_question"] = record.get("question")
            item["source_created_at"] = record.get("created_at")
            papers.append(item)
            if len(papers) >= limit:
                return papers
    return papers


def build_memory_packet(
    records: list[dict[str, Any]],
    *,
    store: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a compact typed memory packet from recent JSONL episodes.

    This packet is deterministic and deliberately modest: it does not decide
    routing, does not judge evidence, and does not invent facts. It only
    separates thread episodes from paper/link memory so downstream LLM agents
    can reason over cleaner context.
    """

    latest_paper_episode: dict[str, Any] | None = None
    for record in reversed(records):
        if record.get("selected_papers"):
            latest_paper_episode = record
            break

    return {
        "kind": "thread_memory_packet",
        "policy": (
            "Use memory only for continuity, follow-up resolution, user preferences, "
            "and links from prior turns. Do not treat memory as paper/web evidence."
        ),
        "last_episode": _episode_brief(records[-1]) if records else None,
        "latest_paper_episode": _episode_brief(latest_paper_episode)
        if latest_paper_episode
        else None,
        "latest_paper_episode_question": latest_paper_episode.get("question")
        if latest_paper_episode
        else None,
        "latest_paper_episode_selected_papers": latest_paper_episode.get("selected_papers", [])
        if latest_paper_episode
        else [],
        "recent_selected_papers": _dedupe_papers_by_title(records),
        "recent_episodes": [_episode_brief(record) for record in records[-5:]],
        "episode_index": (store or {}).get("episode_index", []) if store else [],
        "paper_cards": (store or {}).get("paper_cards", []) if store else [],
        "store": store or {},
    }


def _format_paper_lines(papers: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    seen_titles: set[str] = set()
    for paper in papers:
        title = _clean_text(paper.get("title"), width=100)
        key = title.lower()
        if not title or key in seen_titles:
            continue
        seen_titles.add(key)
        year = _clean_text(paper.get("year"), width=12) or "?"
        venue = _clean_text(paper.get("venue"), width=50) or "?"
        paper_url = _clean_text(paper.get("paper_url"), width=130)
        pdf_url = _clean_text(paper.get("pdf_url"), width=130)
        source_question = _clean_text(paper.get("source_question"), width=120)
        link_bits = []
        if paper_url:
            link_bits.append(f"paper_url={paper_url}")
        if pdf_url:
            link_bits.append(f"pdf_url={pdf_url}")
        if source_question:
            link_bits.append(f"from_question={source_question}")
        suffix = " " + " ".join(link_bits) if link_bits else ""
        lines.append(f"- {title} ({year}, {venue}){suffix}")
        if len(lines) >= limit:
            break
    return lines


def _format_episode_index_lines(items: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    lines: list[str] = []
    for item in reversed(items[-limit:]):
        episode_id = _clean_text(item.get("episode_id"), width=30)
        question = _clean_text(item.get("question"), width=130)
        summary = _clean_text(item.get("summary"), width=220)
        topics = ", ".join(str(topic) for topic in (item.get("topics") or [])[:4])
        paper_count = len(item.get("selected_paper_ids") or [])
        lines.append(
            f"- episode_id={episode_id} question={question} "
            f"papers={paper_count} topics={_clean_text(topics, width=120) or 'N/A'}"
        )
        if summary:
            lines.append(f"  summary={summary}")
    return lines


def _format_paper_card_lines(items: list[dict[str, Any]], *, limit: int = 8) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for item in reversed(items):
        paper_id = _clean_text(item.get("paper_id"), width=70)
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        title = _clean_text(item.get("title"), width=100)
        status = _clean_text(item.get("status"), width=30)
        confidence = _clean_text(item.get("confidence"), width=20)
        source_question = _clean_text(item.get("source_question"), width=100)
        lines.append(
            f"- paper_id={paper_id} title={title} status={status} "
            f"confidence={confidence} from_question={source_question}"
        )
        if len(lines) >= limit:
            break
    return lines


def format_memory_context(
    records: list[dict[str, Any]],
    *,
    packet: dict[str, Any] | None = None,
) -> str:
    if not records and not (packet or {}).get("store"):
        return ""
    packet = packet or build_memory_packet(records)
    lines = [
        "Typed thread memory packet from previous runs. Use only for continuity, "
        "follow-up resolution, user preferences, and links already surfaced. "
        "Do not treat memory as PaperQA/Web evidence; fresh retrieval overrides it."
    ]

    store = packet.get("store") or {}
    if store:
        for label, key in (
            ("Identity memory", "identity"),
            ("Long-term semantic memory", "semantic_memory"),
            ("Experience memory", "lessons"),
        ):
            text = str(store.get(key) or "").strip()
            if text:
                lines.append(f"{label}:\n{text}")

    latest_paper_episode = packet.get("latest_paper_episode") or {}
    latest_papers = packet.get("latest_paper_episode_selected_papers") or []
    if latest_paper_episode:
        lines.append(
            "\n".join(
                [
                    "Latest episode with selected papers:",
                    f"question: {_clean_text(packet.get('latest_paper_episode_question'), width=180)}",
                    "selected_papers:",
                    *(
                        _format_paper_lines(latest_papers, limit=8)
                        or ["- N/A"]
                    ),
                ]
            )
        )

    recent_selected = packet.get("recent_selected_papers") or []
    if recent_selected:
        lines.append(
            "\n".join(
                [
                    "Recent selected papers across memory:",
                    *(_format_paper_lines(recent_selected, limit=10) or ["- N/A"]),
                ]
            )
        )

    episode_index = packet.get("episode_index") or []
    if episode_index:
        lines.append(
            "\n".join(
                [
                    "Recent episode index:",
                    *(_format_episode_index_lines(episode_index, limit=5) or ["- N/A"]),
                ]
            )
        )

    paper_cards = packet.get("paper_cards") or []
    if paper_cards:
        lines.append(
            "\n".join(
                [
                    "Recent paper cards:",
                    *(_format_paper_card_lines(paper_cards, limit=8) or ["- N/A"]),
                ]
            )
        )

    for idx, record in enumerate(records, start=1):
        perspectives = record.get("research_perspectives") or []
        perspective_names = "; ".join(
            str(item.get("perspective") or "")
            for item in perspectives[:3]
            if item.get("perspective")
        )
        selected_count = len(record.get("selected_papers") or [])
        web_count = len(record.get("web_sources") or [])
        lines.append(
            "\n".join(
                [
                    f"Turn {idx}:",
                    f"question: {_clean_text(record.get('question'), width=160)}",
                    f"task: {record.get('task_type') or record.get('task_types')}",
                    f"selected_paper_count: {selected_count}",
                    f"web_source_count: {web_count}",
                    f"perspectives: {perspective_names or 'N/A'}",
                    f"answer_hint: {_clean_text(record.get('episode_summary') or record.get('answer_excerpt'), width=180)}",
                ]
            )
        )
    return "\n\n".join(lines)
