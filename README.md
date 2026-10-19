# Paper Tutor

A reading tutor for research papers. You upload a PDF, and instead of summarising it and letting you
skim, the app asks you questions about it and tells you what you actually understood.

Built for AILT9019 Group 57. It is a study aid: it does not write assignments, review papers, or
verify whether a finding is true.

---

## What it does

| Tab | What you get | Grounding |
|---|---|---|
| **Overview** | one-line claim, research problem, contributions, method, findings, limitations, 6 key terms | every claim carries `[p.N]` |
| **Ask the paper** | answers built only from retrieved passages | numbered sources shown under each answer; refuses when the paper does not cover it |
| **Guided reading** | Socratic questions per section, L1 recall → L5 transfer | diagnosis of your answer: correct / partial / missing / misconception, with the sentence that settles it |
| **Mind map** | your dialogue as a Mermaid graph, colour-coded by understanding | nodes come from real dialogue turns, not from a second LLM guess |

---

## Setup (macOS / Linux, 3 minutes)

```bash
cd paper-tutor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your key in .env
```

`.env` needs one of these. **Gemini API is not available in Hong Kong, Macao or mainland China**
(Google's supported-region list excludes them), so the default preset here is Groq:

| Preset | Cost | How to get a key |
|---|---|---|
| **OpenRouter** (default) | `:free` models, ~50 req/day, no payment | openrouter.ai → Keys → Create key |
| DeepSeek | ~¥1 / M tokens, no daily cap | platform.deepseek.com |
| Groq | free, ~1,000 req/day | console.groq.com → API Keys (console unreachable on some networks) |
| SiliconFlow | free quota on several models | cloud.siliconflow.cn |
| Google Gemini | free, ~1,500 req/day | aistudio.google.com (needs a non-HK account + IP) |
| OpenAI | paid | platform.openai.com |

> GitHub Models was retired on 2026-07-30 — a GitHub PAT no longer works for inference.

Any OpenAI-compatible endpoint works — the app only uses `chat.completions`.
After editing `.env`, verify the key before opening the app:

```bash
python check_key.py
```

It prints the endpoint, the model, and either `OK` or a specific reason (401 key rejected,
429 quota, 404 wrong model name). If a provider's website will not even open, find out whether
it is your network or the service:

```bash
python netcheck.py
```

## Run

```bash
streamlit run app.py
```

Opens on <http://localhost:8501>.

## How to check it works (the core path)

1. Upload `samples/attention.pdf` (Attention Is All You Need) in the sidebar.
   Sidebar should show **15 pages · ~45 chunks** and list the detected sections
   (`Abstract`, `Introduction`, `3 Model Architecture`, `3.2.2 Multi-Head Attention`, …).
2. **Overview → Analyse this paper.** You should get a one-line claim, five contributions,
   findings with `[p.N]` citations, and terms such as *self-attention* and *BLEU*.
3. **Ask the paper →** `how many attention heads does the base model use?`
   Expected: 8 heads, with source `[2]` pointing at §3.2.2, page 5.
4. **Ask the paper →** `what is the capital of France?`
   Expected: a refusal saying the paper does not address it (`not_in_paper`).
5. **Guided reading →** pick `Introduction`, L2, **New question**, answer it badly on purpose.
   Expected: verdict *partial* or *misconception*, a missing-points list, and one follow-up question.
6. **Mind map →** after two turns, a coloured graph with your questions, concepts and verdicts,
   plus `.mmd` / `.md` downloads.

If step 2 fails with `LLM call failed: ... 401`, the API key in `.env` is wrong.
If step 1 shows `0 pages`, the PDF is a scan without a text layer.

## Project layout

```
paper-tutor/
├── app.py                 # Streamlit UI, 4 tabs, session state
├── src/
│   ├── pdf_parser.py      # PyMuPDF extraction, two-column order, heading detection, segmentation
│   ├── retrieval.py       # chunking + BM25 + page/section metadata + grounded digest
│   ├── llm.py             # OpenAI-compatible client, JSON parsing, on-disk response cache
│   ├── prompts.py         # every prompt, in one file so they can be tuned
│   └── modes.py           # overview / terms / cited QA / Socratic / mind map
├── samples/attention.pdf  # test paper
└── .env.example
```

## Design decisions

- **BM25 instead of a vector store.** For a single paper (~45 chunks) lexical retrieval is fast,
  free, deterministic, and strong on exact terms (`BLEU`, `p-value`, `ResNet-50`). The hybrid
  vector + keyword retriever promised in the proposal is the next step, not the first one.
- **Page-level citations.** Every chunk stores the page and section it came from, so `[p.7]` can be
  checked by a human in seconds. That is what makes the "grounded" claim testable.
- **Refusal is a feature.** `not_in_paper` is a normal output, tested in step 4 above.
- **Response cache** in `.cache/llm/`. Repeating a question during the demo costs nothing and works
  if the API is rate limited mid-recording.

## Known limitations

- Section detection is regex + font-size based. Papers with unusual heading styles fall back to
  page-level sections; nothing breaks, but citations get coarser.
- Table content is indexed as plain text, so numeric questions sometimes retrieve table junk.
- BM25 misses paraphrases ("how big is the model" vs "number of parameters").
- No OCR: scanned PDFs are rejected rather than silently mis-parsed.
- The mind map needs `cdn.jsdelivr.net` to render the graph; the `.mmd` download works offline.

## Next steps

1. Hybrid retrieval: add embeddings, fuse with BM25 via reciprocal rank fusion.
2. Learner model that persists across sessions (what is mastered vs needs review).
3. Five-student user test with pre/post comprehension checks.
