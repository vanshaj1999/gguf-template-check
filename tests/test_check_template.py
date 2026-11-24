"""Offline tests: run with `uv run --with pytest --with jinja2 pytest -q`."""

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import check_template as ct

OFFICIAL = ROOT / "results" / "official"


def history(family: str, bos: str = "", eos: str = "") -> list[dict]:
    versions = []
    for path in sorted((OFFICIAL / family).glob("v*.jinja"), key=lambda p: int(p.stem.split("_")[0][1:])):
        versions.append({"date": path.stem.split("_")[1],
                         "renders": ct.render_all(path.read_text(), bos, eos), "template": path.read_text()})
    return versions


def verdict(family: str, which: int, bos: str = "", eos: str = "") -> str:
    versions = history(family, bos, eos)
    return ct.classify(versions[which]["renders"], versions[-1]["renders"], versions[:-1])[0]


def test_old_qwen3_template_is_out_of_date():
    assert verdict("Qwen__Qwen3-8B", 0, eos="<|im_end|>") == "stale:2025-04-28"


def test_current_qwen3_template_is_up_to_date():
    assert verdict("Qwen__Qwen3-8B", -1, eos="<|im_end|>") == "same"


def test_first_mistral_template_is_out_of_date():
    assert verdict("mistralai__Mistral-7B-Instruct-v0.3", 0, "<s>", "</s>") == "stale:2024-05-22"


def test_old_qwen3_template_fails_on_null_content():
    old = history("Qwen__Qwen3-8B", eos="<|im_end|>")[0]["renders"]
    assert ct.is_error(old["null_content_call"])
    assert not ct.is_error(old["tool_result"])


def test_old_mistral_template_drops_tools():
    versions = history("mistralai__Mistral-7B-Instruct-v0.3", "<s>", "</s>")
    old, cur = versions[0]["renders"], versions[-1]["renders"]
    assert ct.compare(old["tools"], cur["tools"], "tools") == "TOOLS DROPPED"


def test_compare_labels():
    assert ct.compare("a b", "a b") == "same"
    assert ct.compare("ERROR: x", "ERROR: y") == "same"
    assert ct.compare("ERROR: TypeError: boom", "ok").startswith("FAILS")
    assert ct.compare("a\n\nb", "a b") == "whitespace only"
    assert ct.compare("a", "b") == "different prompt"


def _gguf_string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def test_reads_template_from_a_local_gguf(tmp_path):
    kvs = [("general.name", 8, _gguf_string("tiny")),
           ("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, 3) + struct.pack("<3i", 1, 1, 3)),
           ("tokenizer.chat_template", 8, _gguf_string("{{ messages[0].content }}"))]
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, kind, value in kvs:
        body += _gguf_string(key) + struct.pack("<I", kind) + value
    path = tmp_path / "tiny.gguf"
    path.write_bytes(body)
    assert ct.local_template(path) == "{{ messages[0].content }}"
