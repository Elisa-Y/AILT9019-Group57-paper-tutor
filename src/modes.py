"""The four user-facing capabilities: overview, terms, cited Q&A, Socratic mode."""

from __future__ import annotations

from . import prompts
from .llm import chat_json
from .pdf_parser import EXCLUDED_FROM_INDEX, Paper
from .retrieval import BM25Index, Chunk, section_digest

LEVELS = {
    1: "identify / recall",
    2: "explain / connect",
    3: "evidence / justify",
    4: "critique / evaluate",
    5: "transfer / create",
}


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #

def summarize(paper: Paper) -> dict:
    data = chat_json(
        prompts.SYSTEM_TUTOR,
        prompts.SUMMARY.format(digest=section_digest(paper, 1400)),
        temperature=0.2,
        max_tokens=1800,
    )
    data.setdefault("contributions", [])
    data.setdefault("findings", [])
    data.setdefault("limitations", [])
    return data


def key_terms(paper: Paper, n: int = 6) -> list[dict]:
    data = chat_json(
        prompts.SYSTEM_TUTOR,
        prompts.KEY_TERMS.format(digest=section_digest(paper, 1000), n=n),
        temperature=0.3,
        max_tokens=1200,
    )
    return data.get("terms", [])


# --------------------------------------------------------------------------- #
# Cited question answering
# --------------------------------------------------------------------------- #

def _sources_block(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i + 1}] ({c.label})\n{c.text}" for i, c in enumerate(chunks)
    )


def answer_question(index: BM25Index, question: str, k: int = 5) -> dict:
    hits = index.search(question, k=k)
    if not hits:
        return {
            "answer": "I could not find any passage in this paper that matches your question. "
                      "Try using a term that appears in the paper, or ask about a specific section.",
            "key_points": [], "quote": "", "quote_source": 0,
            "confidence": "low", "not_in_paper": True, "_hits": [],
        }
    data = chat_json(
        prompts.SYSTEM_TUTOR,
        prompts.QA.format(sources=_sources_block(hits), question=question),
        temperature=0.1,
        max_tokens=900,
    )
    data.setdefault("key_points", [])
    data["_hits"] = hits
    return data


# --------------------------------------------------------------------------- #
# Socratic guided reading
# --------------------------------------------------------------------------- #

def _section_excerpt(paper: Paper, section: str, limit: int = 6000) -> str:
    s = paper.section(section)
    if not s:
        return ""
    out, used = [], 0
    for para in s.paragraphs:
        if used >= limit:
            out.append("... [truncated]")
            break
        text = para.text[: limit - used]
        out.append(f"[p.{para.page}] {text}")
        used += len(text)
    return "\n".join(out)


def socratic_question(paper: Paper, section: str, level: int, history: list[str]) -> dict:
    s = paper.section(section)
    pages = f"{s.start_page}-{s.end_page}" if s else "?"
    return chat_json(
        prompts.SYSTEM_TUTOR,
        prompts.SOCRATIC_QUESTION.format(
            section=section,
            pages=pages,
            level=level,
            level_name=LEVELS.get(level, "explain"),
            history="\n".join(history[-4:]) or "(no answers yet)",
            excerpt=_section_excerpt(paper, section),
        ),
        temperature=0.5,
        max_tokens=700,
    )


def socratic_evaluate(paper: Paper, section: str, question: str,
                      expected: list[str], student_answer: str) -> dict:
    return chat_json(
        prompts.SYSTEM_TUTOR,
        prompts.SOCRATIC_EVALUATE.format(
            question=question,
            expected="; ".join(expected) or "(not provided)",
            answer=student_answer,
            excerpt=_section_excerpt(paper, section, 3000),
        ),
        temperature=0.2,
        max_tokens=800,
    )


# --------------------------------------------------------------------------- #
# Mind map
# --------------------------------------------------------------------------- #

VERDICT_CLASS = {
    "correct": "ok",
    "partial": "review",
    "missing": "todo",
    "misconception": "wrong",
}


def dialogue_to_mindmap(paper: Paper, turns: list[dict]) -> str:
    """Deterministic Mermaid builder: every node traces back to a real dialogue turn."""
    def esc(text: str, limit: int = 46) -> str:
        text = " ".join(text.split()).replace('"', "'").replace("[", "(").replace("]", ")")
        return text[:limit] + ("..." if len(text) > limit else "")

    lines = [
        "graph LR",
        f'  root(["{esc(paper.title, 60)}"])',
        "  classDef root fill:#e8e4f8,stroke:#6b4fbb,stroke-width:2px,color:#22102f;",
        "  classDef ok fill:#d8f5e3,stroke:#2e9e6b,color:#123;",
        "  classDef review fill:#fff3cd,stroke:#c9a227,color:#3d2f00;",
        "  classDef wrong fill:#fde2e1,stroke:#c0392b,color:#3d0f0c;",
        "  classDef todo fill:#e3efff,stroke:#2f6fb5,color:#0d2742;",
        "  class root root;",
    ]
    sections: dict[str, str] = {}
    for i, turn in enumerate(turns):
        sec = turn.get("section", "General")
        if sec not in sections:
            sid = f"S{len(sections) + 1}"
            sections[sec] = sid
            lines.append(f'  {sid}["{esc(sec, 30)}"]')
            lines.append(f'  root --> {sid}')
        sid = sections[sec]
        verdict = turn.get("verdict", "missing")
        cls = VERDICT_CLASS.get(verdict, "todo")
        nid = f"Q{i + 1}"
        lines.append(f'  {nid}["{esc(turn.get("question", ""))}"]')
        lines.append(f'  {sid} -- asks --> {nid}')
        cid = f"C{i + 1}"
        concept = turn.get("focus") or "concept"
        lines.append(f'  {cid}("{esc(concept, 28)}")')
        lines.append(f'  {nid} -- {verdict} --> {cid}')
        lines.append(f"  class {cid} {cls}")
        if turn.get("followup"):
            fid = f"F{i + 1}"
            lines.append(f'  {fid}["{esc(turn["followup"])}"]')
            lines.append(f'  {cid} -. follow-up .-> {fid}')
            lines.append(f"  class {fid} todo")
    return "\n".join(lines)


def sections_for_index(paper: Paper) -> list[str]:
    """Sections a student can pick for guided reading: substantive, non-front-matter."""
    return [
        s.title for s in paper.sections
        if s.title not in EXCLUDED_FROM_INDEX
        and s.title != "Front Matter"
        and len(s.text) >= 300
    ]
