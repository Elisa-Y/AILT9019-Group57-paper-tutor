"""PDF parsing and section segmentation for academic papers.

Core path step 1-2 of the pipeline:
    PDF -> text with page numbers -> logical sections (Abstract / Method / ...)
"""

from __future__ import annotations

import re
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
    "general discussion": "Discussion",
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

# Chinese section names, mapped onto the same canonical set so downstream code
# (search, Socratic mode, mind map) does not need to know which language it is in.
CANONICAL_ZH = {
    "摘要": "Abstract",
    "中文摘要": "Abstract",
    "引言": "Introduction",
    "绪论": "Introduction",
    "前言": "Introduction",
    "研究背景": "Background",
    "背景": "Background",
    "相关工作": "Related Work",
    "相关研究": "Related Work",
    "文献综述": "Related Work",
    "预备知识": "Preliminaries",
    "基础知识": "Preliminaries",
    "方法": "Method",
    "研究方法": "Method",
    "方法介绍": "Method",
    "模型与方法": "Method",
    "本文方法": "Method",
    "实验": "Experiments",
    "实验设置": "Experiments",
    "实验与分析": "Experiments",
    "实验结果": "Results",
    "结果": "Results",
    "结果与分析": "Results",
    "分析": "Discussion",
    "讨论": "Discussion",
    "结论": "Conclusion",
    "总结": "Conclusion",
    "结论与展望": "Conclusion",
    "结语": "Conclusion",
    "参考文献": "References",
    "參考文獻": "References",
    "附录": "Appendix",
    "附錄": "Appendix",
    # Traditional Chinese (common in Hong Kong / Taiwan course readings)
    "緒論": "Introduction",
    "簡介": "Introduction",
    "研究背景": "Background",
    "相關工作": "Related Work",
    "文獻探討": "Related Work",
    "文獻回顧": "Related Work",
    "預備知識": "Preliminaries",
    "研究方法": "Method",
    "實驗": "Experiments",
    "實驗設置": "Experiments",
    "實驗結果": "Results",
    "結果": "Results",
    "討論": "Discussion",
    "結論": "Conclusion",
    "總結": "Conclusion",
    "結論與展望": "Conclusion",
}

# Sections that should not enter the retrieval index (noise for question answering).
EXCLUDED_FROM_INDEX = {"References", "Appendix"}

_HEADING_NUM_EN = re.compile(r"^(?:(\d+(?:\.\d+)*)\.?|[IVX]{1,5}\.)\s+(.+)$")
# Chinese headings often have no space after the number: "3.2注意力机制"
_HEADING_NUM_ZH = re.compile(r"^(\d+(?:\.\d+)*)\s*[\.、]?\s*([\u3400-\u9fff].*)$")
# Chinese numbering: "第三章 ...", "三、...", "（二）...", "3.2 ..."
_ZH_NUM_PREFIX = re.compile(
    r"^(?:第\s*[一二三四五六七八九十百]+\s*[章节部分]|（\s*[一二三四五六七八九十]+\s*）"
    r"|[一二三四五六七八九十]+\s*[、\.]|\d+(?:\.\d+)*\s*[\.、]?)\s*"
)
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")
_ZH_ABSTRACT = re.compile(r"^(?:中文)?摘要\s*[:：]\s*(.+)$", re.S)
_WS = re.compile(r"[ \t]+")
# Standalone headings like "Study 1" / "Experiment 2a" (common in psych/management papers)
_STUDY_LIKE = re.compile(
    r"^(?:Study|Studies|Experiment|Experiments|Exercise|Session|Phase|Part)\s+\d{1,3}[A-Za-z]?$"
)
_APPENDIX_LIKE = re.compile(r"^Appendix\s+[A-Z]{1,2}$")
# Text blocks that are never the paper title (cover pages, banners, citation lines)
_NOT_TITLE = re.compile(
    r"arxiv|vol\.\s*\d|http|doi\.org|issn|accessibility support|downloaded from"
    r"|copyright|©|provided proper attribution",
    re.I,
)
# JSTOR-style "remediated" PDFs duplicate every line with an invisible OCR layer
_OCR_FONT = "GlyphLess"


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


def _line_meta(line: dict) -> tuple[str, float, bool] | None:
    """(raw text, visible font size, is the whole line bold) for one PDF line.

    Size ignores invisible OCR spans (JSTOR 'remediated' files duplicate every
    line with GlyphLessFont spans whose size differs from the real text).
    """
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    visible = [s for s in spans if _OCR_FONT not in s["font"]]
    use = visible or spans
    text = "".join(s["text"] for s in line["spans"])
    size = max(s["size"] for s in use)
    bold = all((s["flags"] & 16) or "Bold" in s["font"] for s in use)
    return text, size, bold


def _split_block(b: dict) -> list[dict]:
    """Split one PDF block into (heading | body) segments.

    Journals often emit 'Results\\nChoosing to Use the Model. As predicted...'
    as a single block; joining the whole block would swallow the heading into
    the paragraph that follows it. A short bold line starts a new segment.
    """
    segments: list[dict] = []
    for line in b["lines"]:
        meta = _line_meta(line)
        if meta is None:
            continue
        text, size, bold = meta
        cls = "head" if (bold and len(text.strip()) <= 75) else "body"
        merge = False
        if segments and segments[-1]["cls"] == cls:
            # Merge multi-line headings only when the font size matches closely:
            # "Study 1" (11.7pt) followed by "Methods" (9.8pt) are two headings,
            # not one; a two-line title keeps the same size.
            merge = cls == "body" or abs(size - segments[-1]["size"]) <= 0.5
        if merge:
            seg = segments[-1]
            seg["lines"].append(text)
            seg["size"] = max(seg["size"], size)
        else:
            segments.append({"cls": cls, "lines": [text], "size": size,
                             "y0": line["bbox"][1]})
    out = []
    for seg in segments:
        joined = _join_lines(seg["lines"]).strip()
        if joined:
            out.append({"text": joined, "x0": b["bbox"][0], "x1": b["bbox"][2],
                        "y0": seg["y0"], "size": seg["size"]})
    return out


def _ordered_blocks(page: "fitz.Page") -> list[tuple[str, float]]:
    """Return (text, font_size) items in human reading order.

    Two-column academic layouts need column-major order, otherwise the parser
    interleaves the left and right column line by line.
    """
    items: list[dict] = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:  # 0 = text block
            continue
        items.extend(_split_block(b))
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
    """The dominant body font size, weighted by how much text uses it.

    A plain median fails on figure-heavy pages: dozens of tiny axis labels
    outnumber the paragraphs and drag the 'body size' down to the label size,
    which then breaks every size-based heading threshold.
    """
    weights: dict[float, int] = {}
    for page in pages_blocks:
        for text, size in page:
            if size > 0:
                key = round(size, 1)
                weights[key] = weights.get(key, 0) + len(text)
    if not weights:
        return 10.0
    return max(weights.items(), key=lambda kv: kv[1])[0]


def _join_lines(lines: list[str]) -> str:
    """Join the lines of one PDF block.

    Latin text keeps its line breaks (a heading split as "1\\nIntroduction" needs
    them). Chinese text is joined without a separator, because a wrapped Chinese
    sentence would otherwise get a gap in the middle of a word.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _CJK_CHAR.match(out[-1]) and _CJK_CHAR.match(line[0]):
            out += line
        else:
            out += "\n" + line
    return out


def _looks_cjk(text: str) -> bool:
    return bool(_CJK_CHAR.search(text))


def _heading_title(line: str, body_size: float, size: float) -> str | None:
    """Return a section title if this block looks like a heading, else None.

    Handles four shapes:
      1. named English headings ("Abstract", "Related Work")
      2. numbered English headings ("3 Model Architecture", "3.1 Encoder ...")
      3. named Chinese headings ("摘要", "相关工作")
      4. numbered Chinese headings ("3.2 注意力机制", "第三章 方法", "（二）实验")
    """
    text = " ".join(l.strip() for l in line.strip().split("\n") if l.strip())
    if not text or len(text) > 75:
        return None
    text = text.rstrip(".").rstrip()          # "Results." is still a heading
    if not text or text[-1] in "，。；：！？、,;:!?":
        return None

    m = _HEADING_NUM_EN.match(text) or _HEADING_NUM_ZH.match(text)
    if m:
        num, name = m.group(1), m.group(2).strip(" .:、：")
        if not name:
            return None
        # a heading starts with a capital letter, or with a Chinese character
        if not (name[0].isupper() or _looks_cjk(name[0])):
            return None
        if not _looks_cjk(name) and len(name.split()) > 9:
            return None
        # "2017年，..." is body text, not a section: sections never number past 20
        first = int(re.match(r"\d+", num).group(0))
        if first > 20:
            return None
        # Chinese headings are short; this keeps long wrapped lines out
        if _looks_cjk(name) and len(name) > 30:
            return None
        if size < body_size * (1.02 if _looks_cjk(name) else 0.98):
            return None
        key = re.sub(r"\s+", " ", name.lower())
        return CANONICAL.get(key) or CANONICAL_ZH.get(name) or f"{num} {name}"

    # Named heading (no number). Chinese headings are short and set larger than body text.
    stripped = _ZH_NUM_PREFIX.sub("", text).strip(" .:、：")
    if _STUDY_LIKE.match(stripped) or _APPENDIX_LIKE.match(stripped):
        return stripped          # "Study 1", "Experiment 2a", "Appendix A"
    canon = CANONICAL.get(re.sub(r"\s+", " ", stripped.lower())) or CANONICAL_ZH.get(stripped)
    if canon and _looks_cjk(stripped) and size < body_size * 1.02:
        return None
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
    best_title: tuple[float, str] | None = None
    pending_parent: str | None = None

    for page_no, page in enumerate(pages_blocks, start=1):
        for text, size in page:
            first_line = text.split("\n")[0].strip()
            if first_line in running:
                continue
            # The title is the largest text on the first two pages that does not
            # look like a cover-page notice or citation banner. Page 1 is often a
            # JSTOR / course-platform cover rather than the paper itself.
            if (page_no <= 2 and size >= body * 1.25
                    and 15 <= len(text) <= 250 and not _NOT_TITLE.search(text)):
                if best_title is None or size > best_title[0]:
                    best_title = (size, text)

            canonical = _heading_title(text, body, size)
            if canonical:
                if current.paragraphs:
                    sections.append(current)
                    pending_parent = None
                elif current.title != "Front Matter":
                    # "Study 1" followed directly by "Methods": keep the empty
                    # heading as a parent label instead of dropping it.
                    pending_parent = current.title
                # Journals reuse names ("Results" per study); keep titles unique
                # so the section picker and lookups stay unambiguous.
                combined = f"{pending_parent} — {canonical}" if pending_parent else canonical
                if any(s.title == combined for s in sections):
                    combined = f"{combined} (p.{page_no})"
                current = Section(title=combined)
                continue

            para = _clean(text)
            # Chinese papers usually write the abstract inline: "摘要：本文提出……"
            # instead of putting it on its own line, so split it into a section here.
            m_abs = _ZH_ABSTRACT.match(para) if page_no == 1 else None
            if m_abs and not any(s.title == "Abstract" for s in sections):
                if current.paragraphs:
                    sections.append(current)
                rest = m_abs.group(1).strip()
                current = Section(title="Abstract")
                if len(rest) >= 20:
                    current.paragraphs.append(Paragraph(text=rest, page=page_no))
                continue
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

    if best_title:
        title = best_title[1]
    if not title:
        title = sections[0].paragraphs[0].text[:120] if sections else "Untitled paper"

    return Paper(title=_clean(title), sections=sections, n_pages=len(pages_blocks))
