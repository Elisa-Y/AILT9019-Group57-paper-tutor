"""Check that your .env key works before you start the app.

    python check_key.py

Prints one line per problem and exits non-zero if anything is wrong, so you can
run it in 10 seconds instead of debugging inside Streamlit.
"""

from __future__ import annotations

import sys

from src.llm import LLMError, chat, settings, status

PROBE = 'Reply with exactly this JSON and nothing else: {"ok": true}'


def main() -> int:
    cfg = settings()
    print(f"endpoint : {cfg['base_url'] or 'https://api.openai.com/v1'}")
    print(f"model    : {cfg['model']}")
    if cfg["style"] == "vertex":
        print("style    : Vertex AI generateContent (HKU portal)")
        print("auth     : api-key + Ocp-Apim-Subscription-Key headers")
    else:
        print("auth     : Bearer + api-key + subscription-key headers")
    if not cfg["ready"]:
        print("\nFAIL  no key found.")
        print("      cp .env.example .env  and put LLM_API_KEY=... in it")
        return 1
    if ("REPLACE-WITH" in (cfg["base_url"] or "") or "REPLACE-WITH" in cfg["model"]
            or "在这里" in (cfg["base_url"] or "") or "在这里" in cfg["model"]
            or "在这里" in cfg["api_key"]):
        print("\nFAIL  .env still contains the placeholder text.")
        print("      Open .env and replace the three lines with the key and URL from the portal.")
        return 1
    if cfg["base_url"] and not cfg["base_url"].startswith("http"):
        print("\nFAIL  LLM_BASE_URL must start with https:// — it is currently:")
        print(f"      {cfg['base_url']}")
        return 1
    print(f"key      : {cfg['api_key'][:6]}...{cfg['api_key'][-4:]}  ({len(cfg['api_key'])} chars)")

    try:
        reply = chat("You answer with JSON only.", PROBE, temperature=0, max_tokens=50)
    except LLMError as exc:
        print(f"\nFAIL  {exc}")
        return 1

    print(f"reply    : {reply[:80]}")
    print("\nOK    the key works. Now run:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
