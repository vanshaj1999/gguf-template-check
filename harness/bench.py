"""Test B/C runner: send every task to a llama-server and store raw replies (resumable)."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 23


def chat(base: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
                data["_latency_s"] = round(time.time() - t0, 3)
                return data
        except urllib.error.HTTPError as e:
            return {"_http_error": e.code, "_body": (e.read() or b"")[:500].decode("utf-8", "replace")}
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("server unreachable")


def _msg(resp: dict) -> dict:
    if "_http_error" in resp:
        return {"error": f"HTTP {resp['_http_error']}: {resp['_body']}"}
    c = resp["choices"][0]
    m = c.get("message", {})
    t = resp.get("timings", {})
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "finish": c.get("finish_reason"),
            "usage": resp.get("usage", {}), "tok_s": t.get("predicted_per_second"),
            "prompt_tok_s": t.get("prompt_per_second"), "latency_s": resp.get("_latency_s")}


def run_task(base: str, task: dict, extra: dict) -> dict:
    common = {"temperature": 0, "seed": SEED, "max_tokens": task["max_tokens"], **extra}
    if task["group"] != "multiturn":
        body = {**common, "messages": task["messages"]}
        if task.get("tools"):
            body["tools"] = task["tools"]
        return {"turns": [_msg(chat(base, body))]}
    msgs, turns = list(task["messages"]), []
    for user in task["user_turns"]:
        msgs.append({"role": "user", "content": user})
        out = _msg(chat(base, {**common, "messages": msgs}))
        turns.append(out)
        if "error" in out:
            break
        msgs.append({"role": "assistant", "content": out["content"]})
    return {"turns": turns}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--run", required=True, help="run name, e.g. qwen3-8b__unsloth__embedded")
    ap.add_argument("--groups", default="gsm8k,ifeval,bfcl,multiturn,toolresult")
    ap.add_argument("--limit", type=int, default=0, help="max tasks per group (0 = all)")
    ap.add_argument("--max-tokens-cap", type=int, default=0)
    ap.add_argument("--extra", default='{"chat_template_kwargs": {"enable_thinking": false}}')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    groups = a.groups.split(",")
    tasks = [json.loads(line) for line in (ROOT / "data" / "tasks.jsonl").read_text().splitlines()]
    tasks = [t for t in tasks if t["group"] in groups]
    if a.limit:
        kept, count = [], {}
        for t in tasks:
            count[t["group"]] = count.get(t["group"], 0) + 1
            if count[t["group"]] <= a.limit:
                kept.append(t)
        tasks = kept
    out = ROOT / "results" / "bench" / f"{a.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    with open(out, "a") as f:
        for i, t in enumerate(tasks):
            if t["id"] in done:
                continue
            if a.max_tokens_cap:
                t["max_tokens"] = min(t["max_tokens"], a.max_tokens_cap)
            rec = {"id": t["id"], "group": t["group"], "run": a.run, **run_task(a.base, t, extra)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"{a.run}: {i + 1}/{len(tasks)}", flush=True)
    print(f"{a.run}: done -> {out}")


if __name__ == "__main__":
    main()
