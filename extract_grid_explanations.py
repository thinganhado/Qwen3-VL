#!/usr/bin/env python3
"""Extract Explanation text from captioner JSON files and save as txt.

Input structure:
  <src_root>/<sample_id>/<region_id>.json

Output structure:
  <dst_root>/<sample_id>/<region_id>.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXPLANATION_RE = re.compile(r"<Explanation>\s*(.*?)\s*</Explanation>", flags=re.S)


def extract_explanation(response: str) -> str | None:
    match = EXPLANATION_RE.search(response or "")
    if not match:
        return None
    return match.group(1).strip()


def replace_region_reference(text: str, region_id: int | str) -> str:
    pattern = re.compile(rf"\b[Rr]egion\s*{re.escape(str(region_id))}\b")
    return pattern.sub("this region", text)


def process_file(json_path: Path, dst_root: Path, overwrite: bool, verbose: bool) -> tuple[bool, str]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"error reading/parsing JSON: {json_path} -> {exc}"

    sample_id = data.get("sample_id")
    region_id = data.get("region_id")
    response = data.get("response", "")

    if sample_id is None or region_id is None:
        return False, f"missing sample_id/region_id: {json_path}"

    explanation = extract_explanation(response)
    if explanation is None:
        return False, f"no <Explanation> tag: {json_path}"

    cleaned = replace_region_reference(explanation, region_id)

    out_dir = dst_root / str(sample_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{region_id}.txt"

    if out_path.exists() and not overwrite:
        return False, f"exists (use --overwrite): {out_path}"

    out_path.write_text(cleaned + "\n", encoding="utf-8")
    if verbose:
        return True, f"wrote: {out_path}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and normalize Explanation text from captioner JSON files.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/captioner"),
        help="Root folder containing <sample_id>/<region_id>.json",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/En/grid_explanation"),
        help="Root output folder for <sample_id>/<region_id>.txt",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing txt files")
    parser.add_argument("--verbose", action="store_true", help="Print each written file")
    args = parser.parse_args()

    src_root = args.src_root
    dst_root = args.dst_root

    if not src_root.exists():
        print(f"source root does not exist: {src_root}")
        return 1

    json_files = sorted(src_root.glob("*/*.json"))
    if not json_files:
        print(f"no json files found under: {src_root}")
        return 1

    written = 0
    skipped = 0

    for jf in json_files:
        ok, msg = process_file(jf, dst_root, overwrite=args.overwrite, verbose=args.verbose)
        if ok:
            written += 1
            if msg:
                print(msg)
        else:
            skipped += 1
            if msg:
                print(f"skip: {msg}")

    print(f"done. written={written} skipped={skipped} total={len(json_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
