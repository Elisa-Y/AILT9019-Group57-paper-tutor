"""Minimal LLM wrapper: two API styles + on-disk response cache.

Supported out of the box (set these in .env):
  HKU ITS  base_url=<everything before /{deployment-id}:generateContent>
           model=gemini-3.5-flash        LLM_API_STYLE=vertex
           (Vertex AI generateContent format, api-key + subscription-key headers)
  Gemini   base_url=https://generativelanguage.googleapis.com/v1beta/openai/  model=gemini-2.5-flash
  OpenAI   (leave base_url empty)                                             model=gpt-4o-mini
  DeepSeek base_url=https://api.deepseek.com                                  model=deepseek-chat

Set LLM_API_STYLE=vertex for gateways that expose Google's generateContent shape
(the HKU ITS Developer Portal does); the default `openai` style sends the key as
`Authorization: Bearer`, and LLM_AUTH_HEADER adds a second copy under another
header name for API-gateway style servers.

The cache matters for the demo: if the API is rate limited during the recording,
the same prompt replays instantly from disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
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
    header = os.getenv("LLM_AUTH_HEADER", "").strip() or None
    style = os.getenv("LLM_API_STYLE", "openai").strip().lower() or "openai"
    return {"api_key": key, "base_url": base, "model": model,
            "auth_header": header, "style": style, "ready": bool(key)}


def _client() -> OpenAI:
    cfg = settings()
    if not cfg["ready"]:
        raise LLMError(
            "No API key found. Create a .env file in the project root with LLM_API_KEY=... "
            "(copy .env.example)."
        )
    # Send the key three ways: `Authorization: Bearer` for OpenAI-style servers, plus
    # `api-key` and the API-gateway subscription key. Extra headers are ignored by
    # servers that do not use them, so this stays safe for any OpenAI-compatible host.
    headers = {"api-key": cfg["api_key"], "Ocp-Apim-Subscription-Key": cfg["api_key"]}
    if cfg["auth_header"]:
        headers[cfg["auth_header"]] = cfg["api_key"]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], default_headers=headers)


def _cache_path(model: str, system: str, user: str, temperature: float) -> Path:
    raw = f"{model}|{temperature}|{system}|{user}".encode("utf-8")
    return CACHE_DIR / (hashlib.sha256(raw).hexdigest()[:20] + ".json")


def _explain(exc: Exception) -> str:
    """Turn SDK exceptions into something a student can act on."""
    status = getattr(exc, "status_code", None)
    if status == 401:
        return ("401 - the API key was rejected. Check .env: LLM_API_KEY must be the raw key, "
                "with no quotes and no 'Bearer ' prefix. On the HKU portal the subscription "
                "must also be approved, not just requested.")
    if status == 403:
        return ("403 - the key is valid but not allowed to use this model. The subscription may "
                "not include that model yet, or the quota is exhausted. Check the model list in "
                "the portal, then update LLM_MODEL.")
    if status == 404:
        return ("404 - that model id does not exist at this endpoint (or the endpoint path is "
                "wrong). Copy LLM_BASE_URL and LLM_MODEL character for character from the "
                "portal's API documentation page; gateway paths often end in /v1 or /openai/v1.")
    if status == 429:
        return ("429 - rate limit or quota reached. The HKU portal shares one monthly AI-credit "
                "budget between the web app and the API, so heavy testing depletes it; wait a "
                "minute, or top up / switch provider.")
    if "Connection" in type(exc).__name__ or "connect" in str(exc).lower():
        return ("Network error - the endpoint could not be reached. If you are on a campus or "
                "hotel network, try another network, or run: python netcheck.py")
    if isinstance(exc, AttributeError):
        return ("The endpoint replied with something that is not an OpenAI-style completion. "
                "Most often LLM_BASE_URL is missing the path, e.g. it should be "
                "https://generativelanguage.googleapis.com/v1beta/openai/ — note the trailing part.")
    return f"{type(exc).__name__}: {exc}"


def _body_hint(exc: Exception) -> str:
    """Gateways often explain themselves in the body; the SDK hides it behind a status code."""
    resp = getattr(exc, "response", None)
    body = getattr(resp, "text", "") or ""
    body = body.strip().replace("\n", " ")
    return f"\n       server said: {body[:300]}" if body and body != "OK" else ""


def _vertex_url_and_headers(cfg: dict) -> tuple[str, dict]:
    base = (cfg["base_url"] or "").rstrip("/")
    # Portal shows: POST /{deployment-id}:generateContent — base URL is everything before it.
    url = f"{base}/{cfg['model']}:generateContent"
    headers = {
        "Content-Type": "application/json",
        # the try-it panel sends the subscription key both ways; extra headers are harmless
        "api-key": cfg["api_key"],
        "Ocp-Apim-Subscription-Key": cfg["api_key"],
    }
    if cfg["auth_header"]:
        headers[cfg["auth_header"]] = cfg["api_key"]
    return url, headers


def _vertex_text(data: dict) -> str:
    fb = (data.get("promptFeedback") or {}).get("blockReason")
    if fb:
        raise LLMError(f"The gateway blocked this prompt (blockReason={fb}).")
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMError(f"No candidates in reply. Raw: {json.dumps(data)[:300]}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        reason = candidates[0].get("finishReason", "?")
        raise LLMError(f"Empty reply (finishReason={reason}). Raw: {json.dumps(data)[:300]}")
    return text


def _vertex_chat(cfg: dict, system: str, user: str, temperature: float, max_tokens: int) -> str:
    import urllib.error
    import urllib.request

    url, headers = _vertex_url_and_headers(cfg)
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", "replace")
                status_code = resp.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            status_code = exc.code
        except Exception as exc:  # noqa: BLE001 - network level
            raise LLMError(_explain(exc)) from exc
        if status_code == 429 and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        if status_code >= 400:
            err = RuntimeError(f"HTTP {status_code}")
            err.status_code = status_code  # type: ignore[attr-defined]
            err.response_body = body  # type: ignore[attr-defined]
            raise LLMError(_explain(err) + f"\n       server said: {body.strip()[:300]}")
        return _vertex_text(json.loads(body))
    raise LLMError("Unreachable")  # pragma: no cover


def chat(system: str, user: str, temperature: float = 0.2, max_tokens: int = 2000) -> str:
    """Single-turn call with a disk cache. Returns plain text."""
    cfg = settings()
    key = _cache_path(cfg["model"], system, user, temperature)
    if key.exists():
        return json.loads(key.read_text())["text"]

    if cfg["style"] == "vertex":
        text = _vertex_chat(cfg, system, user, temperature, max_tokens)
    else:
        text = _openai_chat(cfg, system, user, temperature, max_tokens)

    if not text:
        raise LLMError("The endpoint returned an empty reply. Try another model in .env.")
    key.write_text(json.dumps({"text": text, "model": cfg["model"]}, ensure_ascii=False))
    return text


def _openai_chat(cfg: dict, system: str, user: str, temperature: float, max_tokens: int) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(3):
        try:
            resp = _client().chat.completions.create(
                model=cfg["model"], temperature=temperature,
                max_tokens=max_tokens, messages=messages,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            # free tiers are rate limited per minute; one short wait usually clears a 429
            if getattr(exc, "status_code", None) == 429 and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise LLMError(_explain(exc) + _body_hint(exc)) from exc
    raise LLMError("Unreachable")  # pragma: no cover


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
