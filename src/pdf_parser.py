"""PDF parsing and section segmentation for academic papers.

Core path step 1-2 of the pipeline:
    PDF -> text with page numbers -> logical sections (Abstract / Method / ...)
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz


# Canonical section names we try to recognise. Order matters: first match wins.
CANONICAL = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "related works": "Related Work",
    "literature review": "Related Work",
    "preliminaries": "Preliminaries",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "proposed method": "Method",
    "materials and methods": "Method",
    "experiment": "Experiments",
    "experiments": "Experiments",
    "experimental setup": "Experiments",
    "evaluation": "Results",
    "results": "Results",
    "results and discussion": "Results",
    "analysis": "Discussion",
    "discussion": "Discussion",
    "ablation study": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "conclusion and future work": "Conclusion",
    "future work": "Conclusion",
    "references": "References",
    "bibliography": "References",
    "appendix": "Appendix",
}

# Sections that should not enter the retrieval index (noise for question answering).
EXCLUDED_FROM_INDEX = {"References", "Appendix"}

_HEADING_NUM = re.compile(r"^(?:(\d+(?:\.\d+)*)\.?|[IVX]{1,5}\.)\s+(.+)$")
_WS = re.compile(r"[ \t]+")


@dataclass
class Paragraph:
    text: str
    page: int  # 1-based


@dataclass
class Section:
    title: str
    paragraphs: list[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def start_page(self) -> int:
        return self.paragraphs[0].page if self.paragraphs else 1

    @property
    def end_page(self) -> int:
        return self.paragraphs[-1].page if self.paragraphs else 1


@dataclass
class Paper:
    title: str
    sections: list[Section]
    n_pages: int

    @property
    def section_titles(self) -> list[str]:
        return [s.title for s in self.sections]

    def section(self, title: str) -> Section | None:
        for s in self.sections:
            if s.title == title:
                return s
        return None

    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


# --------------------------------------------------------------------------- #
# Low level text extraction
# --------------------------------------------------------------------------- #

def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")            # soft hyphen
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = _WS.sub(" ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)     # de-hyphenate line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ordered_blocks(page: "fitz.Page") -> list[tuple[str, float]]:
    """Return (text, font_size) blocks in human reading order.

    Two-column academic layouts need column-major order, otherwise the parser
    interleaves the left and right column line by line.
    """
    blocks = page.get_text("dict")["blocks"]
    items: list[dict] = []
    for b in blocks:
        if b.get("type") != 0:  # 0 = text block
            continue
        lines, size = [], 0.0
        for line in b["lines"]:
            span_text = "".join(s["text"] for s in line["spans"])
            if span_text.strip():
                lines.append(span_text)
                size = max(size, max(s["size"] for s in line["spans"]))
        if lines:
            text = "\n".join(lines).strip()
            if text:
                items.append({"text": text, "x0": b["bbox"][0], "x1": b["bbox"][2],
                              "y0": b["bbox"][1], "size": size})
    if not items:
        return []

    width = page.rect.width
    left = [i for i in items if i["x1"] <= width * 0.52]
    right = [i for i in items if i["x0"] >= width * 0.48]
    two_column = (
        len(items) >= 6
        and len(left) >= len(items) * 0.3
        and len(right) >= len(items) * 0.3
    )
    if two_column:
        items = sorted(left, key=lambda i: i["y0"]) + sorted(right, key=lambda i: i["y0"])
    else:
        items = sorted(items, key=lambda i: (round(i["y0"], 1), i["x0"]))
    return [(i["text"], i["size"]) for i in items]


def _body_size(pages_blocks: list[list[tuple[str, float]]]) -> float:
    sizes = [s for page in pages_blocks for (_, s) in page if s > 0]
    return statistics.median(sizes) if sizes else 10.0


def _heading_title(line: str, body_size: float, size: float) -> str | None:
    """Return a section title if this block looks like a heading, else None.

    Two cases:
      1. named headings ("Abstract", "Related Work") -> normalised to a canonical name
      2. numbered headings ("3 Model Architecture", "3.1 Encoder and Decoder Stacks")
    """
    text = " ".join(l.strip() for l in line.strip().split("\n") if l.strip())
    if not text or len(text) > 75 or text.endswith("."):
        return None
    m = _HEADING_NUM.match(text)
    if m:
        num, name = m.group(1), m.group(2).strip(" .:")
        if not name or not name[0].isupper() or len(name.split()) > 9:
            return None
        if size < body_size * 0.98:  # body text is not a heading
            return None
        return CANONICAL.get(re.sub(r"\s+", " ", name.lower())) or f"{num} {name}"
    canon = CANONICAL.get(re.sub(r"\s+", " ", text.lower().strip(" .:")))
    return canon


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def parse_pdf(path: str) -> Paper:
    doc = fitz.open(path)
    pages_blocks = [_ordered_blocks(p) for p in doc]
    body = _body_size(pages_blocks)

    # Drop running headers / footers: lines repeated on most pages.
    counts: dict[str, int] = {}
    for page in pages_blocks:
        for text, _ in page:
            first = text.split("\n")[0].strip()
            if len(first) < 90:
                counts[first] = counts.get(first, 0) + 1
    threshold = max(3, int(len(pages_blocks) * 0.5))
    running = {t for t, c in counts.items() if c >= threshold}

    sections: list[Section] = []
    current = Section(title="Front Matter")
    title = ""

    for page_no, page in enumerate(pages_blocks, start=1):
        for text, size in page:
            first_line = text.split("\n")[0].strip()
            if first_line in running:
                continue
            if not title and page_no == 1 and size >= body * 1.3 and len(text) < 200:
                title = _clean(text).replace("\n", " ")

            canonical = _heading_title(text, body, size)
            if canonical:
                if current.paragraphs:  # keep any accumulated content
                    sections.append(current)
                current = Section(title=canonical)
                continue

            para = _clean(text)
            if len(para) >= 20:
                current.paragraphs.append(Paragraph(text=para, page=page_no))

    if current.paragraphs:
        sections.append(current)

    sections = [s for s in sections if s.paragraphs]
    if not sections:  # scanned / unparseable PDF: fall back to page blocks
        sections = [
            Section(title=f"Page {i}", paragraphs=[Paragraph(" ".join(t for t, _ in p), i)])
            for i, p in enumerate(pages_blocks, start=1) if p
        ]

    if not title:
        title = sections[0].paragraphs[0].text[:120] if sections else "Untitled paper"

    return Paper(title=_clean(title), sections=sections, n_pages=len(pages_blocks))
