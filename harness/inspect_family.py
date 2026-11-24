# /// script
# requires-python = ">=3.11"
# dependencies = ["jinja2>=3.1"]
# ///
"""Show each unique template in a family: who uses it, and how its prompts differ from official."""

import csv
import difflib
import sys
from collections import defaultdict
from pathlib import Path

from render import render_all

ROOT = Path(__file__).resolve().parent.parent


def main(family: str, max_lines: int = 14) -> None:
    with open(ROOT / "results" / "scan.csv") as f:
        rows = [r for r in csv.DictReader(f)
                if r["family"] == family and r["pure_repack"] == "True" and r["tpl_hash"]]
    fam_dir = ROOT / "results" / "official" / family.replace("/", "__")
    official = max(fam_dir.glob("v*.jinja")).read_text()
    eos = rows[0]["eos"] if rows else ""
    cur = render_all(official, "", eos)
    groups = defaultdict(list)
    for r in rows:
        groups[r["tpl_hash"]].append(r)
    for h, rs in sorted(groups.items(), key=lambda kv: -sum(int(r["downloads"]) for r in kv[1])):
        dl = sum(int(r["downloads"]) for r in rs)
        top = sorted(rs, key=lambda r: -int(r["downloads"]))[:3]
        print(f"\n== {h} uploads={len(rs)} dl={dl} cls={rs[0]['cls']} diff={rs[0]['diff']}")
        print("   top:", ", ".join(f"{r['repo']} ({r['downloads']})" for r in top))
        if rs[0]["errors"]:
            r = render_all((ROOT / "results" / "templates" / f"{h}.jinja").read_text(), "", eos)
            print("   error:", r[rs[0]["errors"].split("|")[0]][:160])
        if rs[0]["diff"]:
            s = rs[0]["diff"].split("|")[0]
            up = render_all((ROOT / "results" / "templates" / f"{h}.jinja").read_text(), "", eos)[s]
            d = list(difflib.unified_diff(cur[s].splitlines(), up.splitlines(), "official", "upload",
                                          lineterm="", n=0))
            print(f"   first diff in '{s}':")
            for line in d[2:max_lines]:
                print("     ", line[:150])


if __name__ == "__main__":
    main(sys.argv[1])
