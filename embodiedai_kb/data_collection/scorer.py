from __future__ import annotations

import re
from dataclasses import dataclass, field

from embodiedai_kb.storage.schemas import PaperMetadata


@dataclass(frozen=True, slots=True)
class TermRule:
    pattern: str
    title_weight: float
    abstract_weight: float
    category: str
    label: str
    regex: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "regex",
            re.compile(self.pattern, flags=re.IGNORECASE),
        )


POSITIVE_RULES: tuple[TermRule, ...] = (
    TermRule(r"\bvision[-\s]?language[-\s]?action\b|\bVLA\b", 5, 3, "vla", "VLA"),
    TermRule(r"\bvision[-\s]?language[-\s]?navigation\b|\bVLN\b", 5, 3, "navigation", "VLN"),
    TermRule(r"\bembodied\s+(ai|agent|agents|intelligence)\b", 5, 3, "embodied_agent", "embodied AI"),
    TermRule(r"\b(agentic|generalist)\s+robot", 4, 2.5, "robot_foundation_model", "agentic/generalist robot"),
    TermRule(r"\brobot(ic)?\s+foundation\s+model", 4, 2.5, "robot_foundation_model", "robot foundation model"),
    TermRule(r"\brobot\s+learning\b|\brobotic\s+learning\b", 3, 1.5, "robot_learning", "robot learning"),
    TermRule(r"\brobot(ic)?\s+manipulation\b", 3.5, 1.8, "robot_manipulation", "robot manipulation"),
    TermRule(r"\bmobile\s+manipulation\b", 3.5, 1.8, "robot_manipulation", "mobile manipulation"),
    TermRule(r"\blanguage[-\s]?guided\s+robot", 4, 2.5, "language_guided_robotics", "language-guided robotics"),
    TermRule(r"\bopen[-\s]?vocabulary\s+robot", 3.5, 2, "language_guided_robotics", "open-vocabulary robotics"),
    TermRule(r"\blong[-\s]?horizon\s+(robot|manipulation|task)", 3, 1.5, "long_horizon", "long-horizon robotics"),
    TermRule(r"\bworld\s+model(s)?\b.*\brobot|\brobot.*\bworld\s+model(s)?\b", 3, 1.5, "world_model", "robot world model"),
    TermRule(r"\bspatial\s+intelligence\b", 2.5, 1.3, "spatial_intelligence", "spatial intelligence"),
    TermRule(r"\bsim[-\s]?to[-\s]?real\b|\breal[-\s]?robot", 2.5, 1.2, "sim2real", "sim-to-real / real robot"),
    TermRule(r"\bpolicy\s+learning\b.*\brobot|\brobot.*\bpolicy\s+learning\b", 2.5, 1.2, "robot_learning", "robot policy learning"),
    TermRule(r"\bmultimodal\b.*\brobot|\brobot.*\bmultimodal\b", 2.5, 1.2, "multimodal_robotics", "multimodal robotics"),
    TermRule(r"\bhabitat\b|\bAI2[-\s]?THOR\b|\bALFRED\b|\bManiSkill\b|\bMineDojo\b", 2.5, 1.2, "benchmark_environment", "embodied benchmark"),
    TermRule(r"\bRT[-\s]?1\b|\bRT[-\s]?2\b|\bOpenVLA\b|\bPaLM[-\s]?E\b|\bOcto\b|\bRoboCat\b|\bCLIPort\b|\bSayCan\b|\bVoxPoser\b", 5, 3, "seed_model", "seed model"),
)

NEGATIVE_RULES: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\bfinancial\s+agent|\btrading\s+agent", re.I), 4, "financial agent"),
    (re.compile(r"\bmedical\s+agent|\bclinical\s+agent", re.I), 3, "medical agent"),
    (re.compile(r"\brecommendation\s+agent|\brecommender", re.I), 3, "recommender agent"),
    (re.compile(r"\bsocial\s+agent|\bchatbot|\bconversational\s+agent", re.I), 2.5, "social/chat agent"),
    (re.compile(r"\bsoftware\s+agent|\bprogramming\s+agent|\bcoding\s+agent", re.I), 3, "software agent"),
    (re.compile(r"\bmolecular\s+agent|\bchemical\s+agent", re.I), 3, "molecular agent"),
)

ROBOT_ANCHOR = re.compile(
    r"\brobot|\bembodied|\bmanipulation|\bnavigation|\bphysical\s+world|\bsim[-\s]?to[-\s]?real|\bhabitat\b|\bAI2[-\s]?THOR\b|\bManiSkill\b",
    re.IGNORECASE,
)


def score_paper(paper: PaperMetadata) -> PaperMetadata:
    title = paper.title or ""
    abstract = paper.abstract or ""
    searchable = f"{title}\n{abstract}"
    score = 0.0
    categories: list[str] = []
    keywords: list[str] = []
    reasons: list[str] = []

    for rule in POSITIVE_RULES:
        title_hits = len(rule.regex.findall(title))
        abstract_hits = len(rule.regex.findall(abstract))
        if title_hits:
            added = title_hits * rule.title_weight
            score += added
            reasons.append(f"title:{rule.label}+{added:g}")
        if abstract_hits:
            added = min(abstract_hits, 3) * rule.abstract_weight
            score += added
            reasons.append(f"abstract:{rule.label}+{added:g}")
        if title_hits or abstract_hits:
            categories.append(rule.category)
            keywords.append(rule.label)

    if ROBOT_ANCHOR.search(searchable) and re.search(
        r"\blanguage\b|\bvision\b|\bmultimodal\b|\bfoundation\b|\bLLM\b|\bVLM\b",
        searchable,
        re.IGNORECASE,
    ):
        score += 1.5
        reasons.append("anchor:robot+vision/language/foundation+1.5")

    if paper.year and paper.year >= 2022:
        score += 0.5
        reasons.append("recent:year>=2022+0.5")

    if paper.pdf_url:
        score += 0.25
        reasons.append("has_pdf+0.25")

    if paper.code_url or paper.project_url:
        score += 0.5
        reasons.append("has_code_or_project+0.5")

    for pattern, penalty, label in NEGATIVE_RULES:
        if pattern.search(searchable) and not ROBOT_ANCHOR.search(searchable):
            score -= penalty
            reasons.append(f"negative:{label}-{penalty:g}")

    paper.relevance_score = round(max(score, 0.0), 2)
    paper.categories = sorted(set(paper.categories + categories))
    paper.keywords = sorted(set(paper.keywords + keywords))
    paper.relevance_reasons = reasons
    paper.decision = (
        "likely_embodied_ai" if paper.relevance_score >= 8 else
        "maybe_related" if paper.relevance_score >= 4 else
        "rejected"
    )
    return paper

