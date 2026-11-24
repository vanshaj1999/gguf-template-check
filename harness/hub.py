"""Small, polite Hugging Face Hub client with an on-disk cache."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "results" / "cache"
UA = "hf-template-scan/0.1 (research script)"
MIN_GAP_S = 0.8
_last = [0.0]


def _get(url: str) -> tuple[int, bytes, dict]:
    wait = MIN_GAP_S - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    headers = {"User-Agent": UA}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        _last[0] = time.time()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                time.sleep(30 * (attempt + 1))
                continue
            return e.code, b"", dict(e.headers or {})
        except (urllib.error.URLError, TimeoutError):
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def fetch(url: str, cache: bool = True) -> tuple[int, bytes, dict]:
    if not cache:
        status, body, headers = _get(url)
        return status, body, {k: v for k, v in headers.items() if k.lower() == "link"}
    CACHE.mkdir(parents=True, exist_ok=True)
    key = CACHE / hashlib.sha256(url.encode()).hexdigest()
    if key.exists():
        meta = json.loads(key.with_suffix(".json").read_text())
        return meta["status"], key.read_bytes(), meta["headers"]
    status, body, headers = _get(url)
    key.write_bytes(body)
    keep = {k: v for k, v in headers.items() if k.lower() == "link"}
    key.with_suffix(".json").write_text(json.dumps({"url": url, "status": status, "headers": keep}))
    return status, body, keep


def fetch_json(url: str, cache: bool = True):
    status, body, headers = fetch(url, cache)
    if status != 200:
        return None, headers
    return json.loads(body), headers


def list_quantized_gguf(base: str) -> list[dict]:
    q = urllib.parse.urlencode(
        [("filter", f"base_model:quantized:{base}"), ("filter", "gguf"), ("limit", "1000"),
         ("sort", "downloads"), ("direction", "-1"), ("expand[]", "gguf"),
         ("expand[]", "downloads"), ("expand[]", "createdAt"), ("expand[]", "lastModified")]
    )
    url = f"https://huggingface.co/api/models?{q}"
    out = []
    while url:
        page, headers = fetch_json(url)
        out.extend(page or [])
        url = None
        link = headers.get("Link") or headers.get("link") or ""
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return out


def commits(repo: str, cache: bool = True) -> list[dict]:
    out, seen = [], set()
    for p in range(50):
        page, _ = fetch_json(f"https://huggingface.co/api/models/{repo}/commits/main?p={p}", cache)
        if not page or page[0]["id"] in seen:
            break
        seen.update(c["id"] for c in page)
        out.extend(page)
    return out


def raw_file(repo: str, rev: str, path: str, cache: bool = True) -> str | None:
    status, body, _ = fetch(f"https://huggingface.co/{repo}/raw/{rev}/{path}", cache)
    return body.decode("utf-8", "replace") if status == 200 else None
