from __future__ import annotations

import json
import re
from pathlib import Path
from textwrap import shorten
from typing import Any

from scripts.ask_literature import (
    normalize_openai_compatible_model,
    openai_compatible_api_key,
)

from .memory import memory_store_dir


def _clean_text(value: Any, *, width: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    return shorten(text, width=width, placeholder="...")


def _memory_model(args: Any) -> str | None:
    model = (
        getattr(args, "memory_llm", None)
        or getattr(args, "router_llm", None)
        or getattr(args, "llm", None)
        or getattr(args, "agent_llm", None)
    )
    if not model:
        return None
    if args.openai_base_url and not args.disable_openai_compatible_config:
        return normalize_openai_compatible_model(model, args, args.openai_base_url)
    return model


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else {}


def _episode_index_text(memory_packet: dict[str, Any], *, limit: int = 20) -> str:
    items = (memory_packet.get("episode_index") or [])[-limit:]
    lines: list[str] = []
    for item in reversed(items):
        lines.append(
            "\n".join(
                [
                    f"- episode_id={_clean_text(item.get('episode_id'), width=40)}",
                    f"  question={_clean_text(item.get('question'), width=180)}",
                    f"  summary={_clean_text(item.get('summary'), width=300)}",
                    f"  selected_paper_ids={', '.join(str(x) for x in (item.get('selected_paper_ids') or [])[:8])}",
                    f"  paper_card_ids={', '.join(str(x) for x in (item.get('paper_card_ids') or [])[:8])}",
                ]
            )
        )
    return "\n".join(lines) or "N/A"


def _paper_card_index_text(memory_packet: dict[str, Any], *, limit: int = 30) -> str:
    items = (memory_packet.get("paper_cards") or [])[-limit:]
    lines: list[str] = []
    for item in reversed(items):
        lines.append(
            "\n".join(
                [
                    f"- card_id={_clean_text(item.get('card_id'), width=48)}",
                    f"  paper_id={_clean_text(item.get('paper_id'), width=80)}",
                    f"  episode_id={_clean_text(item.get('episode_id'), width=48)}",
                    f"  title={_clean_text(item.get('title'), width=180)}",
                    f"  status={_clean_text(item.get('status'), width=40)} confidence={_clean_text(item.get('confidence'), width=30)}",
                    f"  source_question={_clean_text(item.get('source_question'), width=180)}",
                    f"  summary={_clean_text(item.get('summary') or item.get('core_idea'), width=280)}",
                    f"  evidence_ids={', '.join(str(x) for x in (item.get('evidence_ids') or [])[:8])}",
                ]
            )
        )
    return "\n".join(lines) or "N/A"


def _collection_blocks(text: str) -> list[str]:
    blocks = re.split(r"(?=<!-- paper_collection_id:)", text)
    return [block.strip() for block in blocks if block.strip()]


def _episode_blocks(text: str) -> list[str]:
    blocks = re.split(r"(?=<!-- episode_id:)", text)
    return [block.strip() for block in blocks if block.strip()]


def _extract_episode_details(store_dir: Path, episode_ids: list[str], *, max_chars: int) -> list[str]:
    path = store_dir / "EPISODES.md"
    if not path.exists() or not episode_ids:
        return []
    wanted = {str(item).strip() for item in episode_ids if str(item).strip()}
    details: list[str] = []
    for block in _episode_blocks(path.read_text(encoding="utf-8", errors="replace")):
        if any(f"episode_id: {episode_id}" in block for episode_id in wanted):
            details.append(shorten(block, width=max_chars, placeholder="\n..."))
    return details


def _card_blocks_from_collection(collection: str) -> tuple[str, list[str]]:
    parts = re.split(r"(?=\n### \d+\. )", collection)
    if not parts:
        return collection, []
    header = parts[0].strip()
    cards = [part.strip() for part in parts[1:] if part.strip()]
    return header, cards


def _extract_paper_details(
    store_dir: Path,
    *,
    episode_ids: list[str],
    card_ids: list[str],
    paper_ids: list[str],
    max_chars: int,
) -> list[str]:
    path = store_dir / "PAPERS.md"
    if not path.exists():
        return []
    wanted_episodes = {str(item).strip() for item in episode_ids if str(item).strip()}
    wanted_cards = {str(item).strip() for item in card_ids if str(item).strip()}
    wanted_papers = {str(item).strip() for item in paper_ids if str(item).strip()}
    if not (wanted_episodes or wanted_cards or wanted_papers):
        return []

    details: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for collection in _collection_blocks(text):
        header, cards = _card_blocks_from_collection(collection)
        if wanted_episodes and any(f"episode={episode_id}" in header for episode_id in wanted_episodes):
            details.append(shorten(collection, width=max_chars, placeholder="\n..."))
            continue
        for card in cards:
            card_hit = any(f"- card_id: {card_id}" in card for card_id in wanted_cards)
            paper_hit = any(f"- paper_id: {paper_id}" in card for paper_id in wanted_papers)
            if card_hit or paper_hit:
                details.append(shorten(f"{header}\n{card}", width=max_chars, placeholder="\n..."))
    return details


def memory_get_details(
    args: Any,
    *,
    episode_ids: list[str] | None = None,
    card_ids: list[str] | None = None,
    paper_ids: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Local memory_get tool: fetch detailed Markdown blocks by ids."""

    store_dir = memory_store_dir(args)
    max_chars = int(getattr(args, "memory_detail_max_chars", 9000) or 9000)
    per_block_chars = max(1200, max_chars // 4)
    episode_details = _extract_episode_details(
        store_dir,
        episode_ids or [],
        max_chars=per_block_chars,
    )
    paper_details = _extract_paper_details(
        store_dir,
        episode_ids=episode_ids or [],
        card_ids=card_ids or [],
        paper_ids=paper_ids or [],
        max_chars=per_block_chars,
    )
    sections: list[str] = []
    if episode_details:
        sections.append("## Recalled Episode Details\n" + "\n\n".join(episode_details))
    if paper_details:
        sections.append("## Recalled Paper Card Details\n" + "\n\n".join(paper_details))
    detail_text = "\n\n".join(sections)
    if len(detail_text) > max_chars:
        detail_text = shorten(detail_text, width=max_chars, placeholder="\n...")
    return detail_text, {
        "episode_detail_count": len(episode_details),
        "paper_detail_count": len(paper_details),
        "max_chars": max_chars,
    }


def _empty_trace(reason: str) -> dict[str, Any]:
    return {
        "mode": "none",
        "need_memory_detail": False,
        "episode_ids": [],
        "paper_ids": [],
        "card_ids": [],
        "memory_sufficient_to_answer": False,
        "reason": reason,
    }


async def run_memory_recall_agent(
    *,
    question: str,
    memory_packet: dict[str, Any],
    args: Any,
) -> tuple[str, dict[str, Any]]:
    """Decide which indexed memories to fetch, then run the local memory_get tool."""

    if getattr(args, "no_memory", False):
        return "", _empty_trace("memory disabled")
    if not memory_packet:
        return "", _empty_trace("no memory packet")
    if not (memory_packet.get("episode_index") or memory_packet.get("paper_cards")):
        return "", _empty_trace("no memory index")

    model = _memory_model(args)
    if not model:
        return "", _empty_trace("no memory recall llm configured")

    prompt = (
        "You are MemoryRecallAgent for a literature assistant. Your job is NOT to answer the user. "
        "First inspect the lightweight memory index. Decide whether detailed memory should be retrieved "
        "before Planner/Supervisor decides the next workflow.\n\n"
        "Use memory details when the user refers to previous work, previous papers, '刚才/上次/之前/这篇/这些/那个方向', "
        "asks for links from prior results, or asks to continue/refine a previous analysis. "
        "Do not retrieve details for an unrelated fresh research task unless a previous episode/card is clearly useful.\n\n"
        "Return ONLY JSON with keys:\n"
        "need_memory_detail: boolean,\n"
        "episode_ids: array of episode_id strings,\n"
        "paper_ids: array of paper_id strings,\n"
        "card_ids: array of card_id strings,\n"
        "memory_sufficient_to_answer: boolean,\n"
        "reason: short Chinese explanation.\n\n"
        f"User question:\n{question}\n\n"
        f"Episode index:\n{_episode_index_text(memory_packet)}\n\n"
        f"Paper card index:\n{_paper_card_index_text(memory_packet)}"
    )
    try:
        import litellm

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You select relevant memory ids for a retrieval tool. "
                        "Be conservative: retrieve only when memory is clearly relevant."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": int(getattr(args, "memory_recall_max_tokens", 900) or 900),
            "timeout": float(getattr(args, "llm_timeout", 180.0)),
        }
        if args.openai_base_url and not args.disable_openai_compatible_config:
            kwargs["api_base"] = args.openai_base_url
            api_key = openai_compatible_api_key(
                args,
                args.openai_base_url,
                args.openai_api_key_env,
            )
            if api_key:
                kwargs["api_key"] = api_key
        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0]
        payload = _extract_json_object(choice.message.content or "")
    except Exception as exc:
        trace = _empty_trace("memory recall llm failed")
        trace.update({"mode": "error", "error": shorten(str(exc), width=500, placeholder="...")})
        return "", trace

    decision = {
        "mode": "llm",
        "need_memory_detail": bool(payload.get("need_memory_detail")),
        "episode_ids": [str(x) for x in (payload.get("episode_ids") or []) if str(x).strip()],
        "paper_ids": [str(x) for x in (payload.get("paper_ids") or []) if str(x).strip()],
        "card_ids": [str(x) for x in (payload.get("card_ids") or []) if str(x).strip()],
        "memory_sufficient_to_answer": bool(payload.get("memory_sufficient_to_answer")),
        "reason": _clean_text(payload.get("reason"), width=300),
    }
    if not decision["need_memory_detail"]:
        return "", decision

    details, get_trace = memory_get_details(
        args,
        episode_ids=decision["episode_ids"],
        paper_ids=decision["paper_ids"],
        card_ids=decision["card_ids"],
    )
    decision.update(
        {
            "memory_get": get_trace,
            "detail_chars": len(details),
        }
    )
    return details, decision
