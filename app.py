"""Paper Tutor - a Socratic reading assistant for research papers.

Run:  streamlit run app.py
Core path: upload PDF -> parse & segment -> overview -> cited Q&A -> guided dialogue -> mind map
"""

from __future__ import annotations

import os
import tempfile

import streamlit as st

from src import modes
from src.llm import LLMError, settings, status
from src.pdf_parser import parse_pdf
from src.retrieval import BM25Index, build_chunks

st.set_page_config(page_title="Paper Tutor", page_icon="📖", layout="wide")

VERDICT_STYLE = {
    "correct": ("Understood", "#d8f5e3", "#2e9e6b"),
    "partial": ("Partly right", "#fff3cd", "#c9a227"),
    "missing": ("Needs another look", "#e3efff", "#2f6fb5"),
    "misconception": ("Misconception", "#fde2e1", "#c0392b"),
}


def init_state() -> None:
    defaults = {
        "paper": None, "chunks": [], "index": None, "filename": "",
        "summary": None, "terms": None, "chat": [], "turns": [],
        "question": None, "section": None, "level": 1, "history": [], "q_id": 0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_pdf(uploaded) -> None:
    """Parse in a temp file that is deleted immediately (no paper leaves the machine)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        path = tmp.name
    try:
        paper = parse_pdf(path)
    finally:
        os.unlink(path)
    st.session_state.paper = paper
    st.session_state.chunks = build_chunks(paper)
    st.session_state.index = BM25Index(st.session_state.chunks)
    st.session_state.filename = uploaded.name
    for k in ("summary", "terms", "chat", "turns", "question", "history"):
        st.session_state[k] = None if k in ("summary", "terms", "question") else []
    st.session_state.level = 1


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Paper Tutor")
        st.caption("Read a paper with a tutor that asks you questions instead of answering for you.")
        cfg = status()
        (st.success if settings()["ready"] else st.warning)(f"Model: {cfg}")
        if not settings()["ready"]:
            st.caption("Add `LLM_API_KEY` to `.env`, then run `python check_key.py`.")

        st.divider()
        up = st.file_uploader("Upload a paper (PDF)", type="pdf")
        if up is not None and up.name != st.session_state.filename:
            with st.spinner("Parsing and indexing…"):
                try:
                    load_pdf(up)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse this PDF: {exc}")

        if st.session_state.paper:
            p = st.session_state.paper
            st.success(f"**{p.title[:70]}**\n\n{p.n_pages} pages · {len(st.session_state.chunks)} chunks")
            st.caption("Sections: " + ", ".join(p.section_titles))
            if st.button("Start over", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        st.divider()
        st.caption("Nothing is stored on a server. The PDF is parsed locally "
                   "and deleted right after indexing.")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_overview() -> None:
    paper = st.session_state.paper
    if not paper:
        st.info("Upload a paper in the sidebar to begin.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Analyse this paper", type="primary", use_container_width=True):
            with st.spinner("Reading the paper…"):
                try:
                    st.session_state.summary = modes.summarize(paper)
                    st.session_state.terms = modes.key_terms(paper, 6)
                except LLMError as exc:
                    st.error(str(exc))

    s = st.session_state.summary
    if not s:
        st.caption("Generates: one-line claim, research problem, contributions, method, "
                   "findings, limitations and key terms — each grounded in the PDF.")
        return

    st.markdown(f"#### {s.get('one_line', '')}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Research problem**")
        st.write(s.get("problem", ""))
        st.markdown("**Method**")
        st.write(s.get("method", ""))
        st.markdown("**Contributions**")
        for c in s.get("contributions", []):
            st.markdown(f"- {c}")
    with b:
        st.markdown("**Main findings**")
        for f in s.get("findings", []):
            st.markdown(f"- {f}")
        st.markdown("**Limitations**")
        for l in s.get("limitations", []):  # noqa: E741
            st.markdown(f"- {l}")
    if s.get("cite"):
        st.caption(f"Pages to check: {s['cite']}")

    if st.session_state.terms:
        st.divider()
        st.markdown("**Terms you probably need**")
        for t in st.session_state.terms:
            with st.expander(f"{t.get('term', '')} — {t.get('plain', '')}"):
                st.write(t.get("in_paper", ""))
                st.caption(f"used on p.{t.get('page', '?')}")


def tab_ask() -> None:
    paper = st.session_state.paper
    if not paper:
        st.info("Upload a paper in the sidebar to begin.")
        return

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("not_in_paper"):
                    st.caption("The paper does not cover this — the tutor refused to guess.")
                if msg.get("quote"):
                    st.caption(f"“{msg['quote']}” — source [{msg.get('quote_source', '?')}]")
                if msg.get("sources"):
                    with st.expander(f"Retrieved passages ({len(msg['sources'])})"):
                        for i, c in enumerate(msg["sources"], start=1):
                            st.markdown(f"**[{i}] {c.label}**")
                            st.caption(c.text[:600] + "…")

    q = st.chat_input("Ask something the paper can answer…")
    if not q:
        return
    st.session_state.chat.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching the paper…"):
            try:
                res = modes.answer_question(st.session_state.index, q)
            except LLMError as exc:
                st.error(str(exc))
                return
        st.markdown(res.get("answer", ""))
        if res.get("key_points"):
            for p in res["key_points"]:
                st.markdown(f"- {p}")
        st.session_state.chat.append({
            "role": "assistant",
            "content": res.get("answer", ""),
            "quote": res.get("quote", ""),
            "quote_source": res.get("quote_source", 0),
            "not_in_paper": res.get("not_in_paper", False),
            "sources": res.get("_hits", []),
        })


def tab_guided() -> None:
    paper = st.session_state.paper
    if not paper:
        st.info("Upload a paper in the sidebar to begin.")
        return

    sections = modes.sections_for_index(paper)
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        section = st.selectbox("Section", sections, index=min(1, len(sections) - 1))
    with c2:
        level = st.select_slider(
            "Depth", options=[1, 2, 3, 4, 5], value=st.session_state.level,
            format_func=lambda x: f"L{x} · {modes.LEVELS[x]}",
        )
    with c3:
        st.write("")
        ask = st.button("New question" if not st.session_state.question else "Skip",
                        use_container_width=True)

    if ask:
        with st.spinner("Choosing a question…"):
            try:
                q = modes.socratic_question(paper, section, level, st.session_state.history)
                st.session_state.question = q
                st.session_state.section = section
                st.session_state.level = level
                st.session_state.q_id += 1  # new widget key -> empty answer box
            except LLMError as exc:
                st.error(str(exc))

    q = st.session_state.question

    # Feedback on the previous answer survives the rerun.
    last = st.session_state.get("last_eval")
    if last:
        ev = last["ev"]
        label, bg, fg = VERDICT_STYLE.get(ev.get("verdict", "missing"), VERDICT_STYLE["missing"])
        st.markdown(
            f"<div style='border:1px solid {fg};background:{bg};border-radius:8px;"
            f"padding:14px 16px'><b style='color:{fg}'>{label}</b> · <i>{last['question']}</i><br><br>"
            f"{ev.get('feedback', '')}"
            + (f"<br><br><i>“{ev.get('evidence', '')}”</i> "
               f"<span style='color:{fg}'>[p.{ev.get('page', '?')}]</span>"
               if ev.get("evidence") else "")
            + "</div>",
            unsafe_allow_html=True,
        )
        if ev.get("missing_points"):
            st.markdown("**Still missing:** " + "; ".join(ev["missing_points"]))
        if ev.get("hint"):
            st.info(f"Hint: {ev['hint']}")
        st.divider()

    if not q:
        st.caption("The tutor asks, you answer. It gives hints, not answers — level L1 recalls, "
                   "L5 asks you to transfer the idea to a new setting.")
        return

    st.markdown(
        f"<div style='border-left:4px solid #6b4fbb;padding:12px 16px;background:#f6f4fd;"
        f"border-radius:6px'><b>L{st.session_state.level} · {q.get('focus', '')}</b><br>"
        f"{q.get('question', '')}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Hint"):
        st.write(q.get("hint", "—"))

    ans = st.text_area("Your answer", height=120, key=f"answer_{st.session_state.q_id}",
                       placeholder="Explain it in your own words. 'I don't know' is a valid answer.")
    submit = st.button("Submit", type="primary", disabled=not ans.strip())

    if submit:
        with st.spinner("Checking your answer…"):
            try:
                ev = modes.socratic_evaluate(
                    paper, st.session_state.section, q.get("question", ""),
                    q.get("expected_points", []), ans.strip(),
                )
            except LLMError as exc:
                st.error(str(exc))
                return

        st.session_state.turns.append({
            "section": st.session_state.section,
            "question": q.get("question", ""),
            "focus": q.get("focus", ""),
            "answer": ans.strip(),
            "verdict": ev.get("verdict", "missing"),
            "followup": ev.get("followup", ""),
        })
        st.session_state.history.append(f"Q: {q.get('question','')}\nA: {ans.strip()[:200]}")

        if ev.get("level_up") and st.session_state.level < 5:
            st.session_state.level += 1
        st.session_state.question = (
            {"question": ev["followup"], "focus": q.get("focus", ""),
             "hint": "", "expected_points": []}
            if ev.get("followup") else None
        )
        st.session_state.last_eval = {"ev": ev, "question": q.get("question", "")}
        st.session_state.q_id += 1  # fresh answer box for the next question
        st.rerun()


def tab_mindmap() -> None:
    paper = st.session_state.paper
    if not paper:
        st.info("Upload a paper in the sidebar to begin.")
        return
    turns = st.session_state.turns
    if not turns:
        st.caption("Finish at least one guided-reading turn and the map builds itself.")
        return

    code = modes.dialogue_to_mindmap(paper, turns)
    st.caption("Green = understood · yellow = needs review · red = misconception · blue = to explore")
    st.html(
        f"""<div id="mm"></div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
          mermaid.initialize({{startOnLoad:true, theme:'base'}});
          mermaid.render('g', `{code}`).then(r => document.getElementById('mm').innerHTML = r.svg);
        </script>""",
        unsafe_allow_javascript=True,
    )
    c1, c2 = st.columns(2)
    c1.download_button("Download .mmd", code, file_name="mindmap.mmd", use_container_width=True)
    c2.download_button("Download .md", f"```mermaid\n{code}\n```", file_name="mindmap.md",
                       use_container_width=True)
    with st.expander("Mermaid source"):
        st.code(code, language="mermaid")


# --------------------------------------------------------------------------- #
init_state()
sidebar()

if st.session_state.paper:
    t1, t2, t3, t4 = st.tabs(["Overview", "Ask the paper", "Guided reading", "Mind map"])
    with t1:
        tab_overview()
    with t2:
        tab_ask()
    with t3:
        tab_guided()
    with t4:
        tab_mindmap()
else:
    st.title("📖 Paper Tutor")
    st.markdown(
        "Upload a research paper and read it **with** a tutor, not through one.\n\n"
        "1. **Overview** — problem, contributions, method, findings, limitations, key terms, all cited.\n"
        "2. **Ask the paper** — answers come only from retrieved passages, with page numbers.\n"
        "3. **Guided reading** — Socratic questions per section, L1 to L5, with diagnosis of your answer.\n"
        "4. **Mind map** — your dialogue becomes a map of what you understand and what you don't.\n\n"
        "Start in the sidebar."
    )
