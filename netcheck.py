"""Which provider endpoints can this machine actually reach?

    python netcheck.py

Browser "this site can't be reached" is usually DNS or a network filter, not the
service being down. This separates the two: it resolves the name, opens a TLS
connection, and prints the HTTP status. 401/403 from an API host is GOOD news --
it means the service answered (curl just has no key and no browser headers).
"""

from __future__ import annotations

import socket
import ssl
import time
from urllib.parse import urlparse

HOSTS = [
    ("OpenRouter", "https://openrouter.ai/"),
    ("OpenRouter API", "https://openrouter.ai/api/v1/models"),
    ("DeepSeek API", "https://api.deepseek.com/"),
    ("Groq console", "https://console.groq.com/"),
    ("Groq API", "https://api.groq.com/openai/v1/models"),
    ("SiliconFlow", "https://api.siliconflow.cn/v1/models"),
    ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models"),
]


def probe(url: str, timeout: float = 6.0) -> str:
    host = urlparse(url).netloc
    t0 = time.time()
    try:
        socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return f"DNS FAIL ({exc})"
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host):
                pass
    except Exception as exc:  # noqa: BLE001
        return f"TLS FAIL ({type(exc).__name__})"
    return f"reachable in {time.time() - t0:.1f}s"


def main() -> None:
    print(f"{'service':<18}{'result':<34}endpoint")
    print("-" * 92)
    for name, url in HOSTS:
        print(f"{name:<18}{probe(url):<34}{url}")
    print("\nDNS FAIL  -> your network cannot resolve the name: try another network (phone hotspot).")
    print("TLS FAIL  -> the connection is being reset or filtered: try another network.")
    print("reachable -> the host is up. If the browser still fails, clear cache or try another browser.")


if __name__ == "__main__":
    main()
