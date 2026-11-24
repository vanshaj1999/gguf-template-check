"""Spot check: read tokenizer.chat_template from real GGUF headers (range requests only)."""

import csv
import json
import random
import struct
import urllib.request
from pathlib import Path

import hub

ROOT = Path(__file__).resolve().parent.parent
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q",
           12: "<d"}


class NeedMore(Exception):
    pass


class Reader:
    def __init__(self, buf: bytes):
        self.b, self.i = buf, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise NeedMore
        out = self.b[self.i:self.i + n]
        self.i += n
        return out

    def num(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.num("<Q")).decode("utf-8", "replace")

    def value(self, t: int):
        if t in SCALARS:
            return self.num(SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.num("<I"), self.num("<Q")
            if et in SCALARS:
                self.take(struct.calcsize(SCALARS[et]) * n)
            else:
                for _ in range(n):
                    self.value(et)
            return None
        raise ValueError(f"unknown gguf type {t}")


def chat_template(repo: str, path: str) -> str | None:
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    size = 8 << 20
    while size <= 128 << 20:
        req = urllib.request.Request(url, headers={"User-Agent": hub.UA, "Range": f"bytes=0-{size - 1}"})
        buf = urllib.request.urlopen(req, timeout=120).read()
        r = Reader(buf)
        try:
            assert r.take(4) == b"GGUF"
            r.num("<I")
            r.num("<Q")
            for _ in range(r.num("<Q")):
                key, t = r.string(), r.num("<I")
                v = r.value(t)
                if key == "tokenizer.chat_template":
                    return v
            return None
        except NeedMore:
            size *= 2
    raise RuntimeError("header larger than 128 MB")


def pick_file(repo: str) -> str | None:
    tree, _ = hub.fetch_json(f"https://huggingface.co/api/models/{repo}/tree/main")
    ggufs = sorted(t["path"] for t in tree or [] if t["path"].endswith(".gguf"))
    for pref in ("q4_k_m", "q8_0", "mxfp4", ""):
        for p in ggufs:
            if pref in p.lower() and ("-of-" not in p or "00001-of" in p):
                return p
    return None


def main() -> None:
    with open(ROOT / "results" / "scan.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["pure_repack"] == "True" and r["tpl_hash"]]
    must = {"Qwen/Qwen3-8B-GGUF", "unsloth/Qwen3-8B-GGUF", "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF", "ggml-org/gpt-oss-20b-GGUF",
            "unsloth/gpt-oss-20b-GGUF"}
    chosen = [r for r in rows if r["repo"] in must]
    rng = random.Random(23)
    rest = [r for r in rows if r["repo"] not in must]
    chosen += rng.sample(rest, 20 - len(chosen))
    out = []
    for r in chosen:
        path = pick_file(r["repo"])
        api_tpl = (ROOT / "results" / "templates" / f"{r['tpl_hash']}.jinja").read_text()
        try:
            file_tpl = chat_template(r["repo"], path) if path else None
        except Exception as e:  # noqa: BLE001
            file_tpl = f"ERROR {e}"
        same = file_tpl == api_tpl
        out.append({"repo": r["repo"], "file": path, "match": same})
        print(("MATCH " if same else "DIFF  ") + f"{r['repo']} / {path}", flush=True)
    (ROOT / "results" / "header_check.json").write_text(json.dumps(out, indent=1))
    print(f"{sum(o['match'] for o in out)}/{len(out)} file headers match the API template")


if __name__ == "__main__":
    main()
