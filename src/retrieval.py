"""Chunking + lexical retrieval (BM25).

Why BM25 instead of a vector database for v0.1?
  - Zero extra dependencies, no embedding API cost, no index build time.
  - Academic papers are full of exact terms (BLEU, p-value, ResNet-50), which
    lexical matching handles very well.
Embeddings can be added later as a second retriever and merged (hybrid), which
is exactly what the proposal promises.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .pdf_parser import EXCLUDED_FROM_INDEX, Paper

_WORD = re.compile(r"[a-z0-9][a-z0-9\-\.']*")
# Chinese/Japanese/Korean text has no spaces, so BM25 needs its own segmentation.
# Character bigrams are the standard cheap substitute for a word segmenter: they need
# no dictionary, handle unknown terms (专有名词, 模型名), and work well for retrieval.
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_CJK_STOP_BIGRAMS = {
    "我们", "本文", "本章", "可以", "这个", "那个", "这些", "那些", "以及", "但是",
    "因为", "所以", "如果", "因此", "由于", "并且", "或者", "而是", "就是", "没有",
    "不是", "一个", "一种", "进行", "通过", "对于", "其中", "为了", "根据", "例如",
}
_CJK_STOP_CHARS = set("的了在是和与及对从把被就都也很更最这那其之而或以为我们你他她它")
_STOP = set(
    """a an the of in on for to and or is are was were be been being with by as at
    from that this these those it its we our they their he she his her which who
    not no can could may might will would should do does did done have has had
    than then there here such into about over under between within using used use
    also more most other some any all both each""".split()
)


def _cjk_tokens(text: str) -> list[str]:
    """Bigrams for CJK runs, single characters when the run is one character long."""
    out: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            if run not in _CJK_STOP_CHARS:
                out.append(run)
            continue
        for i in range(len(run) - 1):
            bigram = run[i:i + 2]
            if bigram not in _CJK_STOP_BIGRAMS:
                out.append(bigram)
    return out


def tokenize(text: str) -> list[str]:
    out = _cjk_tokens(text)
    for w in _WORD.findall(text.lower()):
        w = w.strip(".'-")
        if len(w) < 2 or w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural stemming: findings -> finding
        out.append(w)
    return out


def detect_lang(text: str) -> str:
    """'zh' when the text is mostly Chinese, else 'en'. Used to localise UI messages."""
    sample = text[:4000]
    cjk = len(_CJK_RUN.findall(sample))
    letters = len(re.findall(r"[A-Za-z]", sample))
    return "zh" if cjk * 3 > letters else "en"


@dataclass
class Chunk:
    id: int
    text: str
    page: int
    section: str

    @property
    def label(self) -> str:
        return f"p.{self.page} §{self.section}"


def build_chunks(paper: Paper, max_chars: int = 1100, overlap: int = 120) -> list[Chunk]:
    """Merge paragraphs into ~max_chars chunks, keeping page + section metadata."""
    chunks: list[Chunk] = []
    for section in paper.sections:
        if section.title in EXCLUDED_FROM_INDEX:
            continue
        buf, buf_page = "", section.start_page
        for para in section.paragraphs:
            if not buf:
                buf_page = para.page
            if len(buf) + len(para.text) <= max_chars:
                buf = f"{buf}\n{para.text}".strip()
                continue
            if buf:
                chunks.append(Chunk(len(chunks), buf, buf_page, section.title))
            tail = buf[-overlap:] if len(buf) > overlap else ""
            buf = f"{tail}\n{para.text}".strip()
            buf_page = para.page
        if buf:
            chunks.append(Chunk(len(chunks), buf, buf_page, section.title))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = sum(self.dl) / max(1, len(self.dl))
        df: Counter = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        q = tokenize(query)
        if not q:
            return []
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                score += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            # small boost when the query term appears in the section heading
            heading = set(tokenize(self.chunks[i].section))
section_matches = len(heading & set(q))

score += 1.2 * section_matches

# extra boost for important sections
important = {
    "Introduction",
    "Method",
    "Experiments",
    "Results",
    "Discussion",
    "Conclusion",
}

if self.chunks[i].section in important:
    score += 0.5 * section_matches
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


def section_digest(paper: Paper, max_chars_per_section: int = 1400) -> str:
    """Compact view of the paper for whole-paper prompts.

    Every paragraph is prefixed with its page number so the model can cite
    `[p.7]` and the reader can verify the claim in the PDF.
    """
    parts = []
    for s in paper.sections:
        if s.title in EXCLUDED_FROM_INDEX:
            continue
        lines, used = [], 0
        for para in s.paragraphs:
            if used >= max_chars_per_section:
                break
            text = para.text[: max_chars_per_section - used]
            lines.append(f"[p.{para.page}] {text}")
            used += len(text)
        parts.append(f"### {s.title} (p.{s.start_page}-{s.end_page})\n" + "\n".join(lines))
    return "\n\n".join(parts)


def selectable_sections(paper: Paper) -> list[str]:
    return [s.title for s in paper.sections if s.title not in EXCLUDED_FROM_INDEX]
