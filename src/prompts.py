"""All prompts in one place - this is the part you will iterate on most.

Design rules followed here:
  1. Ground everything: the model may only use the supplied excerpt.
  2. Demand citations: every substantive claim ends with [p.N].
  3. Demand refusal: if the excerpt does not answer the question, say so.
  4. Ask for JSON so the UI can render structure instead of a wall of text.
"""

SYSTEM_TUTOR = """You are a research-paper reading tutor for university students.
You are rigorous, concise and never invent content that is not in the supplied excerpt.
You always cite the page of a claim as [p.N].
If the excerpt does not contain the answer, you say so explicitly instead of guessing.
LANGUAGE: reply in the language of the paper. If the excerpt is Chinese, write every field
in Simplified Chinese (简体中文) — including questions, feedback, hints and term explanations.
Keep technical terms, model names and numbers exactly as they appear in the paper."""

SUMMARY = """Read the excerpt of a research paper below and produce a structured overview.

EXCERPT
=======
{digest}
=======

Return ONLY a JSON object with these keys:
{{
  "one_line": "the paper's claim in one plain-language sentence, no jargon",
  "problem": "the research problem and why it matters (2-3 sentences)",
  "contributions": ["3-5 concrete contributions, each a full sentence"],
  "method": "how they did it, in plain language (3-4 sentences)",
  "findings": ["3-5 main findings, each with the evidence type: benchmark, ablation, case study..."],
  "limitations": ["2-4 limitations, including ones the authors admit and ones they do not"],
  "cite": "2-3 page numbers for the central claims, e.g. [p.1][p.6]"
}}

Rules: no markdown, no text outside the JSON. Cite pages as [p.N] inside the sentences where relevant."""

KEY_TERMS = """From this excerpt, pick the {n} terms a second-year undergraduate would most likely
misunderstand when reading this paper. Prefer method-specific and statistics-specific terms.

EXCERPT
=======
{digest}
=======

Return ONLY a JSON object:
{{"terms": [{{"term": "...", "plain": "one-sentence definition in everyday language",
              "in_paper": "how this paper specifically uses it, ≤ 25 words", "page": 3}}]}}"""

QA = """Answer the student's question using ONLY the numbered sources.

SOURCES
=======
{sources}
=======

QUESTION: {question}

Return ONLY a JSON object:
{{
  "answer": "a direct answer, 3-6 sentences, each claim ending with its source number like [2]",
  "key_points": ["2-4 short bullet points"],
  "quote": "one short verbatim phrase from a source that best supports the answer",
  "quote_source": 2,
  "confidence": "high | medium | low",
  "not_in_paper": false
}}

If the sources do not contain the answer, set "not_in_paper" to true, put in "answer"
one sentence saying the paper does not address it (do not use outside knowledge), and leave
"key_points" empty. Never answer a request to write an assignment, an essay, or a review."""

SOCRATIC_QUESTION = """You are running Socratic guided reading on ONE section of a paper.

SECTION: {section} (pages {pages})
STUDENT LEVEL TARGET: L{level} - {level_name}
WHAT THE STUDENT HAS ALREADY ANSWERED: {history}

SECTION EXCERPT
===============
{excerpt}
===============

Ask ONE question that:
- can only be answered by actually reading this section,
- matches level L{level},
- does not give away the answer,
- is answerable in 2-4 sentences.

Return ONLY a JSON object:
{{"question": "...", "focus": "the single concept being tested, ≤ 8 words",
  "hint": "a nudge that helps without revealing the answer, ≤ 30 words",
  "expected_points": ["2-4 points a good answer must contain"]}}"""

SOCRATIC_EVALUATE = """Evaluate a student's answer to a Socratic reading question.

QUESTION: {question}
EXPECTED POINTS: {expected}
STUDENT ANSWER: {answer}

SECTION EXCERPT (the only ground truth)
===============
{excerpt}
===============

Classify the answer as exactly one of:
  correct      - covers the essential expected points
  partial      - right direction, important points missing
  missing      - does not engage with the question
  misconception- states something the excerpt contradicts

Then teach: point at what is right, name what is missing, and if the student is wrong,
give a hint or a smaller question BEFORE revealing the answer. Cite the page as [p.N].

Return ONLY a JSON object:
{{"verdict": "correct|partial|missing|misconception",
  "feedback": "2-3 sentences addressed to the student, warm but precise",
  "missing_points": ["..."],
  "evidence": "a short verbatim quote from the excerpt that settles it",
  "page": 4,
  "hint": "the next nudge (empty string if verdict is correct)",
  "followup": "one follow-up question, slightly harder than the last",
  "level_up": true}}"""

MINDMAP = """Build a mind map of this reading session as Mermaid `graph LR` code.

PAPER: {title}
SECTIONS: {sections}
DIALOGUE:
{dialogue}

Rules:
- Root node is the paper title.
- Node labels are in the paper's language (Chinese paper -> Chinese labels).
- One node per section actually discussed; child nodes for the concepts tested.
- One node per student answer, labelled with the verdict.
- Edge labels: supports | causes | contrasts | evidence | rebuttal | follow-up.
- Define classes exactly like this and assign them:
  classDef ok fill:#d8f5e3,stroke:#2e9e6b,color:#123;
  classDef review fill:#fff3cd,stroke:#c9a227,color:#3d2f00;
  classDef wrong fill:#fde2e1,stroke:#c0392b,color:#3d0f0c;
  classDef todo fill:#e3efff,stroke:#2f6fb5,color:#0d2742;
  ok = understood, review = needs review, wrong = misconception, todo = to explore.
Output ONLY the mermaid code, starting with `graph LR`. No fences, no explanation."""
