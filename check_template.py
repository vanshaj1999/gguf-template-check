# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2>=3.1"]
# ///
"""Check if a GGUF model's chat template still matches the official one from its base model.

    uv run check_template.py bartowski/Qwen_Qwen3-8B-GGUF
    uv run check_template.py ./Qwen3-8B-Q4_K_M.gguf --base Qwen/Qwen3-8B
"""

import argparse
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))

import hub  # noqa: E402
from gguf_header import NeedMore, Reader  # noqa: E402
from render import SCENARIOS, render_all, squash  # noqa: E402
from scan import classify, official_versions  # noqa: E402

LABELS = {
    "plain": "plain question",
    "system": "system prompt",
    "multi_turn": "multi-turn chat",
    "tools": "tool definitions",
    "system_tools": "system prompt and tools",
    "tool_result": "tool call and its result",
    "thinking_off": "thinking switched off",
    "history_thinking": "earlier reasoning in the chat",
    "reasoning_field": "reasoning_content field",
    "null_content_call": "tool call with null content",
    "string_args_call": "tool call, arguments as a string",
    "parallel_calls": "two tool calls at once",
}
TOOL_CHATS = {"tools", "system_tools", "tool_result", "null_content_call", "string_args_call", "parallel_calls"}


def is_error(text: str) -> bool:
    return text.startswith(("ERROR", "PARSE_ERROR"))


def local_template(path: Path) -> str | None:
    size = 8 << 20
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(size)
            r = Reader(buf)
            try:
                if r.take(4) != b"GGUF":
                    sys.exit(f"{path} is not a GGUF file")
                r.num("<I")
                r.num("<Q")
                for _ in range(r.num("<Q")):
                    key, kind = r.string(), r.num("<I")
                    value = r.value(kind)
                    if key == "tokenizer.chat_template":
                        return value
                return None
            except NeedMore:
                if len(buf) < size:
                    sys.exit(f"{path} ends inside its header")
                size *= 2


def hub_template(repo: str) -> tuple[str | None, str, str, list[str]]:
    query = urllib.parse.urlencode([("expand[]", "gguf"), ("expand[]", "baseModels")])
    info, _ = hub.fetch_json(f"https://huggingface.co/api/models/{repo}?{query}", cache=False)
    if info is None:
        sys.exit(f"Could not find {repo} on Hugging Face (private or gated repos need HF_TOKEN).")
    gguf = info.get("gguf") or {}
    bases = [m["id"] for m in (info.get("baseModels") or {}).get("models", [])]
    return gguf.get("chat_template"), gguf.get("bos_token") or "", gguf.get("eos_token") or "", bases


def compare(mine: str, ref: str) -> str:
    if mine == ref or (is_error(mine) and is_error(ref)):
        return "same"
    if is_error(mine):
        return "FAILS: " + mine.split(":", 1)[-1].strip()[:60]
    if is_error(ref):
        return "works (official one fails)"
    if squash(mine) == squash(ref):
        return "whitespace only"
    return "different prompt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="a Hugging Face GGUF repo (user/name) or a local .gguf file")
    ap.add_argument("--base", help="base model repo, if it can't be found automatically")
    ap.add_argument("--save", metavar="FILE", help="write the current official template to FILE")
    args = ap.parse_args()

    local = Path(args.model)
    if local.suffix == ".gguf" and local.exists():
        template, bos, eos, bases = local_template(local), "", "", []
    else:
        template, bos, eos, bases = hub_template(args.model)
    base = args.base or (bases[0] if bases else None)
    if not base:
        sys.exit("No base model listed for this upload. Pass it with --base, e.g. --base Qwen/Qwen3-8B")
    if not template:
        sys.exit(f"{args.model} has no chat template inside it, so your app will fall back to its own.")

    print(f"Checking  {args.model}\nBase      {base}")
    versions, base_bos, base_eos = official_versions(base, cache=False)
    if not versions:
        sys.exit(f"Could not read a chat template from {base} (gated repos need HF_TOKEN).")
    for v in versions:
        v["renders"] = render_all(v["template"], base_bos, base_eos)
    current = versions[-1]
    mine = render_all(template, bos or base_bos, eos or base_eos)
    verdict, _, _ = classify(mine, current["renders"], versions[:-1])
    print(f"Official  {len(versions)} template version(s), the current one from {current['date']}\n")

    if verdict == "same":
        print("Result    UP TO DATE. It builds the same prompt as the current official template.")
    elif verdict.startswith("stale:"):
        print(f"Result    OUT OF DATE. It carries the official template from {verdict[6:]}.")
        print(f"          {base} has changed it since then (latest change {current['date']}).")
    elif verdict == "whitespace":
        print("Result    WHITESPACE ONLY. Same prompt apart from spaces and line breaks.")
    else:
        print("Result    DIFFERENT. It matches no official version, so someone edited it.")

    print(f"\n{'Test conversation':36} This file vs current official")
    rows = {s: compare(mine[s], current["renders"][s]) for s in SCENARIOS}
    for s, status in rows.items():
        print(f"  {LABELS[s]:34} {status}")

    fails = [s for s, st in rows.items() if st.startswith("FAILS")]
    tool_diff = [s for s, st in rows.items() if s in TOOL_CHATS and st == "different prompt"]
    if fails or tool_diff:
        print("\nWhat it means")
        if fails:
            print("  Python apps (transformers, llama-cpp-python) will raise an error on the FAILS chats.")
        if tool_diff:
            print("  Tool calls get a different prompt. Test tool use before you rely on it.")
        print("  llama.cpp has its own template engine and patches some gaps, so check there with")
        print("  llama-server's /apply-template endpoint.")
    if verdict != "same":
        print("\nFix       Use the official template instead of the one inside the file:")
        print(f"            uv run check_template.py {args.model} --save chat_template.jinja")
        print("            llama-server -m model.gguf --jinja --chat-template-file chat_template.jinja")
    if args.save:
        Path(args.save).write_text(current["template"])
        print(f"\nSaved the current official template to {args.save}")
    return 0 if verdict in ("same", "whitespace") else 1


if __name__ == "__main__":
    sys.exit(main())
