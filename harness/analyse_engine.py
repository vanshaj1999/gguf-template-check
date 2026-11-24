"""Per engine, how many downloads get the same prompt as the current official template."""

import csv
import difflib
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "results" / "engine_prompts"


def is_err(s: str) -> bool:
    return s.startswith(("ERROR", "PARSE", "LOAD"))


def main(show_diffs: bool) -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "scan.csv"))
            if r["pure_repack"] == "True"]
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["family"]][r["tpl_hash"]].append(r)
    summary = {}
    for fam, by_hash in groups.items():
        cur_file = sorted((ROOT / "results" / "official" / fam.replace("/", "__")).glob("v*.jinja"))[-1]
        cur = json.loads((PROMPTS / f"{hashlib.sha1(cur_file.read_text().encode()).hexdigest()[:12]}.json"
                          ).read_text())
        total = sum(int(r["downloads"]) for rs in by_hash.values() for r in rs) or 1
        tally = {e: defaultdict(int) for e in ("python", "llama")}
        for h, rs in by_hash.items():
            dl = sum(int(r["downloads"]) for r in rs)
            if not h:
                for e in tally:
                    tally[e]["no_template"] += dl
                continue
            got = json.loads((PROMPTS / f"{h}.json").read_text())
            for e in tally:
                crash = [s for s, v in got[e].items() if is_err(v) and not is_err(cur[e][s])]
                diff = [s for s, v in got[e].items() if not is_err(v) and v != cur[e][s]]
                state = "crash" if crash else "different" if diff else "same"
                tally[e][state] += dl
                if show_diffs and e == "llama" and diff and dl > 1000:
                    s = diff[0]
                    print(f"\n{fam} {h} dl={dl} llama differs in {diff}")
                    for line in list(difflib.unified_diff(cur[e][s].splitlines(), got[e][s].splitlines(),
                                                          lineterm="", n=0))[2:10]:
                        print("    ", line[:140])
        summary[fam] = {e: {k: v / total for k, v in t.items()} for e, t in tally.items()}
        summary[fam]["downloads_30d"] = total
    (ROOT / "results" / "engine_summary.json").write_text(json.dumps(summary, indent=1))
    print()
    for fam, s in summary.items():
        line = " | ".join(f"{e}: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(s[e].items()))
                          for e in ("python", "llama"))
        print(f"{fam:36} dl={s['downloads_30d']:>9,}  {line}")


if __name__ == "__main__":
    main("--diffs" in sys.argv)
