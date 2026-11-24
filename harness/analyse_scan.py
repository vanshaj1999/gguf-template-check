"""Summarise results/scan.csv into results/scan_summary.json and a printed table."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = list(csv.DictReader(open(ROOT / "results" / "scan.csv")))
    summary = {}
    for scope in ("all", "pure"):
        fams = defaultdict(list)
        for r in rows:
            if scope == "all" or r["pure_repack"] == "True":
                fams[r["family"]].append(r)
                fams["ALL"].append(r)
        out = {}
        for fam, rs in fams.items():
            n, dl = len(rs), sum(int(r["downloads"]) for r in rs) or 1
            cls, cls_dl, scen_err, scen_diff = Counter(), Counter(), Counter(), Counter()
            for r in rs:
                c = r["cls"].split(":")[0]
                cls[c] += 1
                cls_dl[c] += int(r["downloads"])
                for s in filter(None, r["errors"].split("|")):
                    scen_err[s] += 1
                for s in filter(None, r["diff"].split("|")):
                    scen_diff[s] += 1
            any_err = [r for r in rs if r["errors"]]
            out[fam] = {
                "uploads": n, "downloads_30d": dl,
                "class_count": dict(cls), "class_download_share": {k: v / dl for k, v in cls_dl.items()},
                "uploads_with_new_errors": len(any_err),
                "download_share_with_new_errors": sum(int(r["downloads"]) for r in any_err) / dl,
                "scenario_errors": dict(scen_err), "scenario_diffs": dict(scen_diff),
                "eos_mismatch": sum(1 for r in rs if r["eos_matches"] != "True" and r["cls"] != "no_template"),
                "unique_templates": len({r["tpl_hash"] for r in rs if r["tpl_hash"]}),
                "official_uploads": [(r["repo"], r["cls"], r["errors"]) for r in rs if r["official"] == "True"],
            }
        summary[scope] = out
    (ROOT / "results" / "scan_summary.json").write_text(json.dumps(summary, indent=1))
    for fam, s in summary["pure"].items():
        cc = s["class_count"]
        print(f"{fam:38} n={s['uploads']:4} same={cc.get('same', 0):4} stale={cc.get('stale', 0):4} "
              f"ws={cc.get('whitespace', 0):3} differs={cc.get('differs', 0):3} none={cc.get('no_template', 0):3} "
              f"| err_uploads={s['uploads_with_new_errors']:4} err_dl={s['download_share_with_new_errors']:.0%} "
              f"| uniq={s['unique_templates']} eos_mis={s['eos_mismatch']}")
        print(f"{'':38} dl share: " + ", ".join(f"{k} {v:.0%}" for k, v in s["class_download_share"].items()))
        if s["scenario_errors"]:
            print(f"{'':38} errors by chat: {s['scenario_errors']}")
        if s["official_uploads"]:
            print(f"{'':38} official: {s['official_uploads']}")


if __name__ == "__main__":
    main()
