"""SHA-256 of the Q4_K_M file in the ten most downloaded clean Qwen3-8B repacks, from Hub LFS metadata."""

import csv
import json
from pathlib import Path

import hub

ROOT = Path(__file__).resolve().parent.parent


def q4_file(repo: str) -> dict | None:
    tree, _ = hub.fetch_json(f"https://huggingface.co/api/models/{repo}/tree/main")
    for entry in sorted(tree or [], key=lambda e: e["path"]):
        path = entry["path"].lower()
        if path.endswith(".gguf") and "q4_k_m" in path and "-of-" not in path and entry.get("lfs"):
            return {"file": entry["path"], "sha256": entry["lfs"]["oid"], "bytes": entry["lfs"]["size"]}
    return None


def main() -> None:
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "scan.csv"))
            if r["family"] == "Qwen/Qwen3-8B" and r["pure_repack"] == "True"]
    rows.sort(key=lambda r: int(r["downloads"]), reverse=True)
    out = []
    for r in rows:
        f = q4_file(r["repo"])
        if f:
            out.append({"repo": r["repo"], "downloads": int(r["downloads"]), **f})
        if len(out) == 10:
            break
    (ROOT / "results" / "file_hashes.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(f"{o['sha256'][:16]}  {o['bytes'] / 1e9:5.2f} GB  {o['repo']}/{o['file']}")
    print(f"{len({o['sha256'] for o in out})} different hashes in {len(out)} files")


if __name__ == "__main__":
    main()
