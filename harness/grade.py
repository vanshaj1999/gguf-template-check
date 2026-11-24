# /// script
# requires-python = ">=3.11"
# dependencies = ["langdetect", "nltk>=3.9", "immutabledict", "absl-py"]
# ///
"""Grade results/bench/*.jsonl and write results/bench_summary.csv plus paired tests."""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = {json.loads(line)["id"]: json.loads(line) for line in open(ROOT / "data" / "tasks.jsonl")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def grade_gsm8k(task, turn) -> bool:
    text = turn.get("content", "")
    m = re.findall(r"Answer:\s*[<*$\\(\s]*(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return False
    try:
        return abs(float(nums[-1].replace(",", "")) - float(task["answer"])) < 1e-6
    except ValueError:
        return False


def grade_ifeval(task, turn) -> bool:
    from instruction_following_eval import instructions_registry
    resp = turn.get("content", "")
    if not resp.strip():
        return False
    for iid, kw in zip(task["instruction_id_list"], task["kwargs"]):
        inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**{k: v for k, v in kw.items() if v is not None})
        args = inst.get_instruction_args()
        if args and "prompt" in args:
            inst.build_description(prompt=task["prompt"])
        if not inst.check_following(resp):
            return False
    return True


def _std(v):
    if isinstance(v, str):
        return re.sub(r"[ ,./\-_*^]", "", v).lower().replace("'", '"')
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        return [_std(x) for x in v]
    if isinstance(v, dict):
        return {k: _std(x) for k, x in v.items()}
    return v


def grade_bfcl(task, turn) -> tuple[bool, bool]:
    """Returns (correct, format_error)."""
    calls = turn.get("tool_calls") or []
    if not calls:
        return False, "<tool_call>" in turn.get("content", "")
    if len(calls) != 1:
        return False, False
    fn = calls[0].get("function", {})
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except json.JSONDecodeError:
        return False, True
    gt = task["ground_truth"][0]
    name = next(iter(gt))
    if fn.get("name") != name:
        return False, False
    spec = gt[name]
    if any(k not in spec for k in args):
        return False, False
    for k, ok_values in spec.items():
        if k not in args:
            if "" not in ok_values:
                return False, False
            continue
        if _std(args[k]) not in [_std(x) for x in ok_values if x != ""]:
            return False, False
    return True, False


def _check(check: str, params: dict, text: str, last: bool) -> bool:
    t = text.strip()
    if check == "ends_with":
        return t.endswith(params["suffix"])
    if check == "max_words":
        return 0 < len(t.split()) < params["max"]
    if check == "language":
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        try:
            return detect(t) == params["lang"]
        except Exception:  # noqa: BLE001
            return False
    if check == "json_only":
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return False
        return isinstance(obj, dict) and all(k in obj for k in params["keys"])
    if check == "no_digits":
        return bool(t) and not re.search(r"\d", t)
    if check == "starts_with":
        return t.startswith(params["prefix"])
    if check == "lowercase":
        return bool(t) and t == t.lower()
    if check == "three_bullets":
        lines = [ln for ln in t.splitlines() if ln.strip()]
        return len(lines) == params["n"] and all(ln.startswith("- ") for ln in lines)
    if check == "secret":
        low = t.lower()
        return params["forbidden"] not in low and params["forbidden"][::-1] not in low
    if check == "persona":
        return (not last) or bool(re.search(rf"\b{params['required_last']}\b", t.lower()))
    raise ValueError(check)


def grade_multiturn(task, turns) -> bool:
    if len(turns) != len(task["user_turns"]) or any("error" in t for t in turns):
        return False
    return all(_check(task["check"], task["params"], t.get("content", ""), i == len(turns) - 1)
               for i, t in enumerate(turns))


def grade_record(rec) -> dict:
    task, turns = TASKS[rec["id"]], rec["turns"]
    first = turns[0] if turns else {}
    out = {"id": rec["id"], "group": rec["group"], "error": any("error" in t for t in turns),
           "runaway": any(t.get("finish") == "length" for t in turns),
           "tok_s": [t["tok_s"] for t in turns if t.get("tok_s")],
           "reasoning_chars": sum(len(t.get("reasoning", "")) for t in turns),
           "format_error": False, "content": first.get("content", "")}
    if rec["group"] == "gsm8k":
        out["correct"] = grade_gsm8k(task, first)
    elif rec["group"] == "ifeval":
        out["correct"] = grade_ifeval(task, first)
    elif rec["group"] == "toolresult":
        text = first.get("content", "").lower()
        out["correct"] = (not first.get("tool_calls")) and any(e in text for e in task["expect_any"])
    elif rec["group"] == "bfcl":
        out["correct"], out["format_error"] = grade_bfcl(task, first)
        out["content"] = json.dumps(first.get("tool_calls")) + first.get("content", "")
    else:
        out["correct"] = grade_multiturn(task, turns)
        out["content"] = "\n---\n".join(t.get("content", "") for t in turns)
    return out


def load(run: str) -> dict[str, dict]:
    path = ROOT / "results" / "bench" / f"{run}.jsonl"
    return {g["id"]: g for g in (grade_record(json.loads(line)) for line in open(path))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names (default: all files)")
    ap.add_argument("--ref", help="reference run for paired tests and agreement")
    ap.add_argument("--out", default="bench_summary.csv", help="file name inside results/")
    a = ap.parse_args()
    import nltk
    nltk.download("punkt_tab", quiet=True)
    runs = a.runs or sorted(p.stem for p in (ROOT / "results" / "bench").glob("*.jsonl"))
    graded = {r: load(r) for r in runs}
    ref = graded.get(a.ref) if a.ref else None
    rows = []
    for run, items in graded.items():
        by_group = defaultdict(list)
        for g in items.values():
            by_group[g["group"]].append(g)
            by_group["ALL"].append(g)
        for group, gs in sorted(by_group.items()):
            n, k = len(gs), sum(g["correct"] for g in gs)
            lo, hi = wilson(k, n)
            speeds = [s for g in gs for s in g["tok_s"]]
            row = {"run": run, "group": group, "n": n, "correct": k, "score": round(k / n, 4),
                   "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                   "runaway": sum(g["runaway"] for g in gs), "errors": sum(g["error"] for g in gs),
                   "format_errors": sum(g["format_error"] for g in gs),
                   "reasoning_items": sum(g["reasoning_chars"] > 0 for g in gs),
                   "tok_s_median": round(sorted(speeds)[len(speeds) // 2], 1) if speeds else ""}
            if ref is not None and run != a.ref:
                pairs = [(ref[g["id"]], g) for g in gs if g["id"] in ref]
                b = sum(1 for r, g in pairs if r["correct"] and not g["correct"])
                c = sum(1 for r, g in pairs if not r["correct"] and g["correct"])
                row.update(ref_worse=c, ref_better=b, mcnemar_p=round(mcnemar_p(b, c), 4),
                           same_output=round(sum(r["content"] == g["content"] for r, g in pairs)
                                             / max(1, len(pairs)), 4))
            rows.append(row)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k)
                    if k in rows[0] else 99)
    with open(ROOT / "results" / a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["group"] == "ALL" or len(graded) == 1:
            print({k: r[k] for k in r if k not in ("ci_low", "ci_high")})


if __name__ == "__main__":
    main()
