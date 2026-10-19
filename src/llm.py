"""Minimal LLM wrapper: any OpenAI-compatible endpoint + on-disk response cache.

Supported out of the box (set these in .env):
  Gemini   base_url=https://generativelanguage.googleapis.com/v1beta/openai/  model=gemini-2.5-flash
  OpenAI   (leave base_url empty)                                             model=gpt-4o-mini
  DeepSeek base_url=https://api.deepseek.com                                  model=deepseek-chat

The cache matters for the demo: if the API is rate limited during the recording,
the same prompt replays instantly from disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class LLMError(RuntimeError):
    pass


def settings() -> dict:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    base = os.getenv("LLM_BASE_URL", "").strip() or None
    model = os.getenv("LLM_MODEL", "").strip() or ("gpt-4o-mini" if base is None else "gemini-2.5-flash")
    return {"api_key": key, "base_url": base, "model": model, "ready": bool(key)}


def _client() -> OpenAI:
    cfg = settings()
    if not cfg["ready"]:
        raise LLMError(
            "No API key found. Create a .env file in the project root with LLM_API_KEY=... "
            "(copy .env.example)."
        )
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def _cache_path(model: str, system: str, user: str, temperature: float) -> Path:
    raw = f"{model}|{temperature}|{system}|{user}".encode("utf-8")
    return CACHE_DIR / (hashlib.sha256(raw).hexdigest()[:20] + ".json")


def _explain(exc: Exception) -> str:
    """Turn SDK exceptions into something a student can act on."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return ("401 - the API key was rejected. Check .env: LLM_API_KEY must be the raw key, "
                "with no quotes and no 'Bearer ' prefix.")
    if status == 403:
        return ("403 - the key is valid but not allowed to use this model or region. "
                "Gemini API is not available in Hong Kong; use the GitHub Models preset instead.")
    if status == 404:
        return ("404 - the model name does not exist at this endpoint. Check LLM_MODEL "
                "(GitHub Models needs the provider prefix, e.g. openai/gpt-4o-mini).")
    if status == 429:
        return ("429 - rate limit or daily quota reached. Free tiers allow ~50 requests/day; "
                "wait a minute, or switch to another preset in .env.")
    if "Connection" in type(exc).__name__ or "connect" in str(exc).lower():
        return ("Network error - the endpoint could not be reached. If you are on a campus or "
                "hotel network, try another network or the GitHub Models preset.")
    if isinstance(exc, AttributeError):
        return ("The endpoint replied with something that is not an OpenAI-style completion. "
                "Most often LLM_BASE_URL is missing the path, e.g. it should be "
                "https://generativelanguage.googleapis.com/v1beta/openai/ or "
                "https://models.github.ai/inference — note the trailing part.")
    return f"{type(exc).__name__}: {exc}"


def chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Single-turn call with a disk cache. Returns plain text."""
    cfg = settings()
    key = _cache_path(cfg["model"], system, user, temperature)
    if key.exists():
        return json.loads(key.read_text())["text"]
    try:
        resp = _client().chat.completions.create(
            model=cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        text = (resp.choices[0].message.content or "").strip()
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a readable message in the UI
        raise LLMError(_explain(exc)) from exc
    if not text:
        raise LLMError("The endpoint returned an empty reply. Try another model in .env.")
    key.write_text(json.dumps({"text": text, "model": cfg["model"]}, ensure_ascii=False))
    return text


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"Model did not return JSON. Raw output:\n{text[:400]}")
    return json.loads(text[start:end + 1])


def chat_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 2000) -> dict:
    """Same as chat() but parses the reply as a JSON object."""
    return _extract_json(chat(system, user, temperature, max_tokens))


def status() -> str:
    cfg = settings()
    if not cfg["ready"]:
        return "offline (no API key)"
    return f"{cfg['model']} @ {cfg['base_url'] or 'api.openai.com'}"
