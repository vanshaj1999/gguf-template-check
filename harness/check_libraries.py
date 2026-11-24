# /// script
# requires-python = ">=3.11"
# dependencies = ["transformers>=5", "jinja2>=3.1", "llama-cpp-python>=0.3"]
# ///
"""Render a tool call with null content through transformers and llama-cpp-python, old vs fixed Qwen3-8B template."""

import importlib.metadata as md
from pathlib import Path

from render import SCENARIOS

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = {"old (2025-04-28)": ROOT / "results/official/Qwen__Qwen3-8B/v0_2025-04-28.jinja",
             "fixed (2025-05-19)": ROOT / "results/official/Qwen__Qwen3-8B/v1_2025-05-19.jinja"}
CHAT = SCENARIOS["null_content_call"]


def with_transformers(template: str) -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return tok.apply_chat_template(CHAT["messages"], tools=CHAT["tools"], chat_template=template, tokenize=False)


def with_llama_cpp(template: str) -> str:
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    fmt = Jinja2ChatFormatter(template=template, eos_token="<|im_end|>", bos_token="<|endoftext|>")
    return fmt(messages=CHAT["messages"], tools=CHAT["tools"]).prompt


def main() -> None:
    for lib, fn in (("transformers", with_transformers), ("llama-cpp-python", with_llama_cpp)):
        print(f"{lib} {md.version(lib)}")
        for label, path in TEMPLATES.items():
            try:
                print(f"  {label:20} OK, {len(fn(path.read_text()))} characters")
            except Exception as e:  # noqa: BLE001
                print(f"  {label:20} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
