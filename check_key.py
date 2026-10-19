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
    if not cfg["ready"]:
        print("\nFAIL  no key found.")
        print("      cp .env.example .env  and put LLM_API_KEY=... in it")
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
