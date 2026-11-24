# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2>=3.1"]
# ///
"""Test A: compare the chat template inside every GGUF upload with the official one."""

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import hub
from render import SCENARIOS, render_all, squash

FAMILIES = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4-mini-instruct",
    "openai/gpt-oss-20b",
]
QUANT_WORDS = re.compile(
    r"gguf|imatrix|i1|i-?quants?|quant(ized|s)?|[iq]?q\d(_[a-z0-9]+)*|iq\d\w*|bf16|fp16|f16|f32|"
    r"[0-9.]+bits?|bits?|ud|static|weighted|llamacpp|llama\.cpp|ollama|lmstudio|gsq|rco|autoround|"
    r"awq|gptq|mxfp4|nvfp4|fp8|int[48]|w[48]a\d+", re.I)
OUT = Path(__file__).resolve().parent.parent / "results"


def pure_repack(repo: str, base: str) -> bool:
    """True when the repo name is only the base model name plus quantisation words."""
    name = repo.split("/")[1].lower()
    variants = {base.split("/")[1].lower(), base.lower().replace("/", "_"),
                base.lower().replace("/", "-")}
    for variant in sorted(variants, key=len, reverse=True):
        name = name.replace(variant, " ")
        name = name.replace(variant.replace("-", "_"), " ").replace(variant.replace(".", "_"), " ")
    tokens = [t.strip("_.") for t in re.split(r"[-\s]+", name) if t.strip("_.")]
    return all(QUANT_WORDS.fullmatch(t) for t in tokens)


def _token(v):
    return v.get("content", "") if isinstance(v, dict) else (v or "")


def _template_from(tok_cfg: str | None, jinja_file: str | None) -> str | None:
    if jinja_file:
        return jinja_file
    if not tok_cfg:
        return None
    ct = json.loads(tok_cfg).get("chat_template")
    if isinstance(ct, list):
        named = {t.get("name"): t.get("template") for t in ct}
        return named.get("default") or next(iter(named.values()), None)
    return ct


def official_versions(base: str, cache: bool = True) -> tuple[list[dict], str, str]:
    """All distinct official templates in the base repo history, oldest first."""
    history = sorted(hub.commits(base, cache), key=lambda c: c["date"])
    versions, seen = [], set()
    for c in history:
        tpl = _template_from(hub.raw_file(base, c["id"], "tokenizer_config.json"),
                             hub.raw_file(base, c["id"], "chat_template.jinja"))
        if tpl and tpl not in seen:
            seen.add(tpl)
            versions.append({"sha": c["id"], "date": c["date"][:10], "template": tpl})
    cfg = json.loads(hub.raw_file(base, "main", "tokenizer_config.json", cache) or "{}")
    return versions, _token(cfg.get("bos_token")), _token(cfg.get("eos_token"))


def _norm(r: dict) -> dict:
    return {s: "ERROR" if v.startswith(("ERROR", "PARSE_ERROR")) else v for s, v in r.items()}


def classify(up_renders: dict, current: dict, older: list[dict]) -> tuple[str, list[str], list[str]]:
    up_renders, current = _norm(up_renders), _norm(current)
    older = [{**v, "renders": _norm(v["renders"])} for v in older]
    diff = [s for s in SCENARIOS if up_renders[s] != current[s]]
    if not diff:
        return "same", [], []
    for v in older:
        if all(up_renders[s] == v["renders"][s] for s in SCENARIOS):
            return f"stale:{v['date']}", diff, []
    ws_only = [s for s in diff if squash(up_renders[s]) == squash(current[s])]
    if len(ws_only) == len(diff):
        return "whitespace", diff, ws_only
    return "differs", diff, ws_only


def scan_family(base: str, writer) -> None:
    versions, bos, eos = official_versions(base)
    fam_dir = OUT / "official" / base.replace("/", "__")
    fam_dir.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(versions):
        (fam_dir / f"v{i}_{v['date']}.jinja").write_text(v["template"])
        v["renders"] = render_all(v["template"], bos, eos)
    current, older = versions[-1]["renders"], versions[:-1]
    uploads = hub.list_quantized_gguf(base)
    print(f"{base}: {len(versions)} official versions, {len(uploads)} uploads", file=sys.stderr)
    for m in uploads:
        g = m.get("gguf") or {}
        tpl = g.get("chat_template") or ""
        row = {"family": base, "repo": m["id"], "author": m["id"].split("/")[0],
               "official": m["id"].split("/")[0] == base.split("/")[0],
               "pure_repack": pure_repack(m["id"], base),
               "downloads": m.get("downloads", 0), "created": (m.get("createdAt") or "")[:10],
               "eos": g.get("eos_token"), "eos_matches": g.get("eos_token") == eos,
               "ctx": g.get("context_length"), "tpl_len": len(tpl),
               "text_exact": tpl == versions[-1]["template"],
               "tpl_hash": hashlib.sha1(tpl.encode()).hexdigest()[:12] if tpl else ""}
        if tpl:
            tdir = OUT / "templates"
            tdir.mkdir(exist_ok=True)
            (tdir / f"{row['tpl_hash']}.jinja").write_text(tpl)
        if not tpl:
            row.update(cls="no_template", diff="", ws_only="", errors="")
        else:
            r = render_all(tpl, g.get("bos_token") or "", g.get("eos_token") or "")
            cls, diff, ws = classify(r, current, older)
            errs = [s for s in SCENARIOS if r[s].startswith(("ERROR", "PARSE_ERROR"))
                    and not current[s].startswith("ERROR")]
            row.update(cls=cls, diff="|".join(diff), ws_only="|".join(ws), errors="|".join(errs))
        writer.writerow(row)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fields = ["family", "repo", "author", "official", "pure_repack", "downloads",
              "created", "eos", "eos_matches", "ctx", "tpl_len", "text_exact", "tpl_hash", "cls",
              "diff",
              "ws_only", "errors"]
    with open(OUT / "scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for base in FAMILIES:
            scan_family(base, w)
            f.flush()


if __name__ == "__main__":
    main()
