#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/region_phone_table_grid.csv"
DEFAULT_MFA_JSON_ROOT = "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned/"
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/"
DEFAULT_OUTPUT_DIR = "/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/outputs"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-30B-A3B-Thinking"


def _normalize_image_ref(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    p = Path(value).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image path does not exist: {p}")
    return str(p)


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_file is None:
        return _load_text_file(DEFAULT_SYSTEM_FILE, "--system-file")
    return _load_text_file(Path(args.system_file), "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    if args.user_template_file is None:
        return _load_text_file(DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")
    return _load_text_file(Path(args.user_template_file), "--user-template-file")


def _extract_transcript_word_tier(mfa_json_path: Path) -> str:
    if not mfa_json_path.exists():
        return ""
    try:
        obj = json.loads(mfa_json_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    entries = obj.get("tiers", {}).get("words", {}).get("entries", [])
    parts = []
    for e in entries:
        if not isinstance(e, (list, tuple)) or len(e) < 3:
            continue
        try:
            start = float(e[0])
            end = float(e[1])
            word = str(e[2]).strip()
        except Exception:
            continue
        if not word:
            continue
        parts.append(f"[{start:.2f}-{end:.2f}] {word}")
    return " ".join(parts)


def _discover_items(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    spec_root = Path(args.spec_root).expanduser().resolve()
    mfa_root = Path(args.mfa_json_root).expanduser().resolve()

    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")

    items = []
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id_raw = str(row.get("sample_id", "")).strip()
            rid_raw = str(row.get("region_id", "")).strip()
            if not sample_id_raw or not rid_raw:
                continue
            try:
                region_id = int(rid_raw)
            except ValueError:
                continue
            sample_id = Path(sample_id_raw).stem

            p1_path = spec_root / f"{sample_id}_grid_img_edge_number_axes.png"
            mfa_json_path = mfa_root / f"{sample_id}.json"
            if not p1_path.exists():
                continue

            items.append(
                {
                    "sample_id": sample_id,
                    "sample_id_raw": sample_id_raw,
                    "region_id": region_id,
                    "crop_method": "GRID",
                    "p1": str(p1_path),
                    "mfa_json": str(mfa_json_path),
                }
            )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid GRID items discovered from CSV + spec root.")

    return sorted(items, key=lambda x: (x["sample_id"], x["region_id"]))


def build_messages(args: argparse.Namespace, item: dict):
    p1 = _normalize_image_ref(item["p1"])
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)

    transcript_text = _extract_transcript_word_tier(Path(item["mfa_json"]))
    user_prompt = user_template.format_map(
        defaultdict(
            str,
            {
                "ID": item["region_id"],
                "id": item["region_id"],
                "region_id": item["region_id"],
                "sample_id": item["sample_id"],
                "sample_id_raw": item.get("sample_id_raw", item["sample_id"]),
                "transcript": transcript_text,
            },
        )
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Spectrogram (GRID with axes):"},
                {"type": "image", "image": p1},
                {"type": "text", "text": f"Transcript (word tier): {transcript_text}" if transcript_text else "Transcript (word tier): [missing]"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    md = {
        "sample_id": item["sample_id"],
        "region_id": item["region_id"],
        "crop_method": "GRID",
        "transcript": transcript_text,
    }
    return messages, md


def parse_args():
    parser = argparse.ArgumentParser(description="Run local HF Qwen-VL prompt for grid-only spectrogram artifact analysis.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")

    parser.add_argument("--meta-csv", default=DEFAULT_META_CSV, help="CSV path with sample_id and region_id entries.")
    parser.add_argument("--spec-root", default=DEFAULT_SPEC_ROOT, help="Root for GRID spectrograms with axes.")
    parser.add_argument("--mfa-json-root", default=DEFAULT_MFA_JSON_ROOT, help="Root for MFA JSON transcript files.")

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=(
            "Path to user prompt template txt. Supports placeholders: "
            "{ID}, {sample_id}, {transcript}. "
            f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}"
        ),
    )

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype, e.g., auto, float16, bfloat16.")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of items to generate per forward pass.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Per-sample output root.")
    parser.add_argument("--output-file", default=None, help="Single-item output file.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat batch output jsonl file.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if existing records exist.")
    parser.add_argument("--print-messages", action="store_true", help="Print built messages before generation.")
    return parser.parse_args()


def _resolve_torch_dtype(dtype_str: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported --dtype: {dtype_str}. Use one of: {list(mapping.keys())}")
    return mapping[dtype_str]


def _generate_one(model, processor, messages, max_new_tokens, do_sample, temperature, top_p):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    sample_flag = bool(do_sample and temperature > 0.0)
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sample_flag}
    if sample_flag:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def _generate_batch(model, processor, batch_messages, max_new_tokens, do_sample, temperature, top_p):
    try:
        inputs = processor.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(model.device)

        sample_flag = bool(do_sample and temperature > 0.0)
        generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sample_flag}
        if sample_flag:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        generated_ids = model.generate(**inputs, **generate_kwargs)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception as e:
        print(f"[batch-fallback] batch_size={len(batch_messages)} reason={e}")
        return [
            _generate_one(model, processor, m, max_new_tokens, do_sample, temperature, top_p)
            for m in batch_messages
        ]


def _write_sample_grouped_json(output_dir: Path, records_by_bucket: dict):
    for (method, sample_id), records in records_by_bucket.items():
        method_dir = output_dir / str(method).lower()
        sample_dir = method_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        by_region = {}
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            by_region[int(rid)] = rec

        payload = {
            "sample_id": sample_id,
            "num_regions": len(by_region),
            "regions": [by_region[rid] for rid in sorted(by_region.keys())],
        }
        (sample_dir / "json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_records_by_sample(output_dir: Path) -> dict:
    records_by_bucket = defaultdict(list)
    if not output_dir.exists():
        return records_by_bucket

    for p in output_dir.glob("*/*/json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = obj.get("sample_id")
        regions = obj.get("regions")
        method = p.parent.parent.name
        if not sample_id or not isinstance(regions, list):
            continue
        for rec in regions:
            if isinstance(rec, dict) and "region_id" in rec:
                rec.setdefault("crop_method", method.upper())
                records_by_bucket[(str(method).upper(), str(sample_id))].append(rec)
    return records_by_bucket


def _existing_done_keys(records_by_bucket: dict) -> set:
    done = set()
    for (method, sample_id), records in records_by_bucket.items():
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            done.add((str(sample_id), int(rid), str(method).upper()))
    return done


def main():
    args = parse_args()
    items = _discover_items(args)
    output_root = Path(args.output_dir).expanduser().resolve()

    existing_records = defaultdict(list)
    if not args.overwrite:
        existing_records = _load_existing_records_by_sample(output_root)
        done_keys = _existing_done_keys(existing_records)
        before = len(items)
        items = [it for it in items if (str(it["sample_id"]), int(it["region_id"]), "GRID") not in done_keys]
        skipped = before - len(items)
        if skipped > 0:
            print(f"[resume] skipped_existing_regions={skipped}")
        if len(items) == 0:
            print("[resume] no pending regions; nothing to generate.")
            return

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    if len(items) > 1 and args.output_file:
        raise ValueError("--output-file is only for single item. Use --output-dir for grouped outputs.")

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    jsonl_fp = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if args.overwrite else "a"
        jsonl_fp = out_path.open(mode, encoding="utf-8", buffering=1)

    records_by_bucket = defaultdict(list)
    if not args.overwrite:
        for bucket, recs in existing_records.items():
            records_by_bucket[bucket].extend(recs)

    try:
        for batch_start in range(0, len(items), args.batch_size):
            batch_items = items[batch_start: batch_start + args.batch_size]
            batch_built = [build_messages(args, item) for item in batch_items]
            batch_messages = [x[0] for x in batch_built]
            batch_md = [x[1] for x in batch_built]

            if args.print_messages:
                for m in batch_messages:
                    print(m)

            if len(batch_messages) == 1:
                batch_outputs = [
                    _generate_one(
                        model=model,
                        processor=processor,
                        messages=batch_messages[0],
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                ]
            else:
                batch_outputs = _generate_batch(
                    model=model,
                    processor=processor,
                    batch_messages=batch_messages,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            for i, (item, md, output_text) in enumerate(zip(batch_items, batch_md, batch_outputs), start=1):
                idx = batch_start + i
                record = {
                    "sample_id": md["sample_id"],
                    "region_id": md["region_id"],
                    "crop_method": "GRID",
                    "transcript": md["transcript"],
                    "p1": item["p1"],
                    "response": output_text,
                }

                bucket = ("GRID", record["sample_id"])
                records_by_bucket[bucket].append(record)

                print(f"[{idx}/{len(items)}] {record['sample_id']}__r{record['region_id']}")
                print(output_text)

                if jsonl_fp is not None:
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl_fp.flush()
                    try:
                        os.fsync(jsonl_fp.fileno())
                    except OSError:
                        pass

                if len(items) == 1 and args.output_file:
                    out_file = Path(args.output_file).expanduser().resolve()
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(output_text, encoding="utf-8")
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    _write_sample_grouped_json(output_root, records_by_bucket)


if __name__ == "__main__":
    main()

