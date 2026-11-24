# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2>=3.1"]
# ///
"""Render each unique Qwen template with llama.cpp (/apply-template) and with Python Jinja."""

import csv
import hashlib
import json
import subprocess
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from render import SCENARIOS, render_all

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "Qwen3-0.6B-Q8_0.gguf"
TPL_DIR = ROOT / "results" / "templates"
PORT = 8091
TINY_BOS, TINY_EOS = "<|endoftext|>", "<|im_end|>"


def to_openai(sc: dict) -> dict:
    msgs = []
    for m in sc["messages"]:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [{**c, "function": {**c["function"], "arguments": (
                c["function"]["arguments"] if isinstance(c["function"]["arguments"], str)
                else json.dumps(c["function"]["arguments"]))}} for c in m["tool_calls"]]
        msgs.append(m)
    body = {"messages": msgs}
    if "tools" in sc:
        body["tools"] = sc["tools"]
    if "enable_thinking" in sc:
        body["chat_template_kwargs"] = {"enable_thinking": sc["enable_thinking"]}
    return body


def _post(path: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def llama_render(tpl_path: Path) -> dict[str, str]:
    proc = subprocess.Popen(
        ["nice", "-n", "15", "llama-server", "-m", str(MODEL), "--jinja", "--chat-template-file",
         str(tpl_path), "-ngl", "0", "-t", "2", "-c", "2048", "--no-warmup", "--host",
         "127.0.0.1", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(60):
            if proc.poll() is not None:
                err = proc.stderr.read().decode("utf-8", "replace")[-300:]
                return {s: f"LOAD_ERROR: {err}" for s in SCENARIOS}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        out = {}
        for name, sc in SCENARIOS.items():
            status, data = _post("/apply-template", to_openai(sc))
            out[name] = data.get("prompt") if status == 200 else f"ERROR: {json.dumps(data)[:200]}"
        return out
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "results" / "scan.csv")))
    by_hash: dict[str, dict] = {}
    users = defaultdict(list)
    for r in rows:
        if r["pure_repack"] != "True" or not r["tpl_hash"]:
            continue
        users[r["tpl_hash"]].append(r)
        by_hash.setdefault(r["tpl_hash"], r)
    for fam_dir in sorted((ROOT / "results" / "official").iterdir()):
        for v in sorted(fam_dir.glob("v*.jinja")):
            h = hashlib.sha1(v.read_text().encode()).hexdigest()[:12]
            if not (TPL_DIR / f"{h}.jinja").exists():
                (TPL_DIR / f"{h}.jinja").write_text(v.read_text())
            by_hash.setdefault(h, {"family": fam_dir.name.replace("__", "/")})
    prompt_dir = ROOT / "results" / "engine_prompts"
    prompt_dir.mkdir(exist_ok=True)
    out_path = ROOT / "results" / "engine.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tpl_hash", "family", "uploads", "downloads", "scenario", "python", "llama",
                    "agree"])
        for h, r in by_hash.items():
            src = (TPL_DIR / f"{h}.jinja").read_text()
            cached = prompt_dir / f"{h}.json"
            if cached.exists():
                both = json.loads(cached.read_text())
                py, ll = both["python"], both["llama"]
            else:
                py = render_all(src, TINY_BOS, TINY_EOS)
                ll = llama_render(TPL_DIR / f"{h}.jinja")
                cached.write_text(json.dumps({"python": py, "llama": ll}))
            dl = sum(int(u["downloads"]) for u in users[h])
            for s in SCENARIOS:
                pys, lls = py[s], ll[s]
                pyc = "error" if pys.startswith(("ERROR", "PARSE")) else hashlib.sha1(
                    pys.encode()).hexdigest()[:10]
                llc = "error" if lls.startswith(("ERROR", "LOAD")) else hashlib.sha1(
                    lls.encode()).hexdigest()[:10]
                w.writerow([h, r["family"], len(users[h]), dl, s, pyc, llc, pyc == llc])
            f.flush()
            print(h, r["family"], len(users[h]), flush=True)


if __name__ == "__main__":
    main()
