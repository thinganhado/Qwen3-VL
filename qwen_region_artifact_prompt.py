#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"

DEFAULT_META_CSV = "/scratch3/che489/Ha/interspeech/datasets/region_phone_table_top3_all_with_ptype_feature.csv"
DEFAULT_P1_ROOT = "/scratch3/che489/Ha/interspeech/localization/Ms_region_outputs"
DEFAULT_P2_ROOT = "/scratch3/che489/Ha/interspeech/localization/region_crops_top3"
DEFAULT_P3_ROOT = "/scratch3/che489/Ha/interspeech/localization/region_crops_real"
DEFAULT_OUTPUT_DIR = "/scratch3/che489/Ha/interspeech/localization/qwen3_vlm"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-30B-A3B-Thinking"

DEFAULT_USER_TEMPLATE = (
    "This region corresponds to {time} section, {frequency} frequency band, {phoneme} token, and {feature} feature.\n"
    "P2 is produced by {crop_method}. Method definition: {method_definition}\n"
    "Compare P2 vs P3 and generate E in the required format."
)

METHOD_DEFINITION_MAP = {
    "GRID": "The crop is taken from a fixed square cell in an NxN grid over the spectrogram.",
    "SUPERPIXEL": "The crop is taken from an irregular region formed by grouping nearby pixels with similar appearance.",
    "SAM": "The crop is taken from a region that follows the visible edges of the pattern as closely as possible.",
}

TIME_MAP = {
    "S": "speech",
    "NS": "non-speech",
    "speech": "speech",
    "non_speech": "non-speech",
    "non-speech": "non-speech",
    "nonspeech": "non-speech",
}

FREQUENCY_MAP = {
    "L": "low",
    "M": "mid",
    "H": "high",
    "low": "low",
    "mid": "mid",
    "high": "high",
}

PHONEME_MAP = {
    "C": "consonant",
    "V": "vowel",
    "none": "unvoiced",
    "silent": "unvoiced",
    "unvoiced": "unvoiced",
    "consonant": "consonant",
    "vowel": "vowel",
}

FEATURE_MAP = {
    "Boundary/Coarticulation": "coarticulation",
    "boundary/coarticulation": "coarticulation",
    "coarticulation": "coarticulation",
    "Frication": "frication",
    "frication": "frication",
    "Formants": "formant bands",
    "formants": "formant bands",
    "formant bands": "formant bands",
    "Harmonic structure": "harmonic structure",
    "harmonic structure": "harmonic structure",
    "none": "no dominant speech feature",
    "None": "no dominant speech feature",
    "no dominant speech feature": "no dominant speech feature",
}

CROP_METHOD_MAP = {
    "GRID": "GRID",
    "SUPERPIXEL": "SUPERPIXEL",
    "SAM": "SAM",
    "grid": "GRID",
    "superpixel": "SUPERPIXEL",
    "sam": "SAM",
}


def _normalize_image_ref(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value

    # For local files, return plain absolute paths (not file:// URIs),
    # since some transformers builds do not accept file:// sources.
    if value.startswith("file://"):
        p = Path(value[len("file://"):]).expanduser().resolve()
    else:
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
        if DEFAULT_USER_TEMPLATE_FILE.exists():
            return _load_text_file(DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")
        return DEFAULT_USER_TEMPLATE
    return _load_text_file(Path(args.user_template_file), "--user-template-file")


def _normalize_choice(field_name: str, value: str, mapping: dict, allowed_hint: str) -> str:
    if value in mapping:
        return mapping[value]
    lowered = value.lower()
    if lowered in mapping:
        return mapping[lowered]
    raise ValueError(f"Unsupported --{field_name} value: {value}. Allowed: {allowed_hint}")


def _parse_sample_region_from_filename(path_value: str):
    p = Path(path_value)
    stem = p.stem
    m = re.match(r"^(?P<sample_id>.+)__r(?P<region_id>\d+)$", stem)
    if not m:
        raise ValueError(
            "Unable to parse sample_id/region_id from crop filename. "
            "Expected pattern: <sample_id>__r<region_id>.png"
        )
    return m.group("sample_id"), int(m.group("region_id"))


def _lookup_region_metadata(csv_path: Path, sample_id: str, method: str, region_id: int):
    resolved = csv_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"--meta-csv file does not exist: {resolved}")

    with resolved.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = str(row.get("sample_id", "")).strip()
            mth = str(row.get("method", "")).strip().lower()
            rid_str = str(row.get("region_id", "")).strip()
            if not sid or not mth or not rid_str:
                continue
            try:
                rid = int(rid_str)
            except ValueError:
                continue

            if sid == sample_id and mth == method.lower() and rid == region_id:
                t_val = str(row.get("T", "")).strip()
                f_val = str(row.get("F", "")).strip()
                # Accept both legacy P_type and current P column.
                p_val = str(row.get("P", row.get("P_type", ""))).strip()
                feat_val = str(row.get("feature", "")).strip() or "none"
                if not t_val or not f_val or not p_val:
                    raise ValueError(
                        f"Matched row missing T/F/P values for sample_id={sample_id}, method={method}, region_id={region_id}"
                    )
                return t_val, f_val, p_val, feat_val

    raise ValueError(
        f"No matching row in CSV for sample_id={sample_id}, method={method}, region_id={region_id}"
    )


def _resolve_metadata(args: argparse.Namespace, sample_id: str, method: str, region_id: int):
    crop_method_value = _normalize_choice(
        "crop-method", str(method), CROP_METHOD_MAP, "GRID, SUPERPIXEL, SAM"
    )

    if args.time and args.frequency and args.phoneme:
        time_raw = args.time
        freq_raw = args.frequency
        phoneme_raw = args.phoneme
        feature_raw = args.feature if args.feature else "none"
    else:
        time_raw, freq_raw, phoneme_raw, feature_raw = _lookup_region_metadata(
            csv_path=Path(args.meta_csv),
            sample_id=sample_id,
            method=crop_method_value,
            region_id=region_id,
        )

    time_value = _normalize_choice("time", time_raw, TIME_MAP, "speech, non-speech")
    frequency_value = _normalize_choice(
        "frequency", freq_raw, FREQUENCY_MAP, "low, mid, high"
    )
    phoneme_value = _normalize_choice("phoneme", phoneme_raw, PHONEME_MAP, "consonant, vowel, unvoiced")
    feature_value = _normalize_choice(
        "feature",
        feature_raw,
        FEATURE_MAP,
        "coarticulation, frication, formant bands, harmonic structure, no dominant speech feature",
    )

    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "crop_method": crop_method_value,
        "time": time_value,
        "frequency": frequency_value,
        "phoneme": phoneme_value,
        "feature": feature_value,
    }


def _discover_triplets(args: argparse.Namespace):
    if args.p2 is not None:
        if args.p1 is None or args.p3 is None:
            sample_id, region_id = _parse_sample_region_from_filename(args.p2)
            method = args.crop_method
            if method is None:
                method = Path(args.p2).parent.name
            method = _normalize_choice("crop-method", method, CROP_METHOD_MAP, "GRID, SUPERPIXEL, SAM")

            p2_path = Path(args.p2)
            p1_path = Path(args.p1) if args.p1 else Path(args.p1_root) / method.lower() / f"{sample_id}_{method.lower()}_img_edge_number.png"
            p3_path = Path(args.p3) if args.p3 else Path(args.p3_root) / method.lower() / p2_path.name
        else:
            p1_path = Path(args.p1)
            p2_path = Path(args.p2)
            p3_path = Path(args.p3)
            sample_id, region_id = _parse_sample_region_from_filename(str(p2_path))
            method = args.crop_method or p2_path.parent.name
            method = _normalize_choice("crop-method", method, CROP_METHOD_MAP, "GRID, SUPERPIXEL, SAM")

        if not p1_path.exists() or not p2_path.exists() or not p3_path.exists():
            raise FileNotFoundError(
                f"Missing one of p1/p2/p3: p1={p1_path} exists={p1_path.exists()}, "
                f"p2={p2_path} exists={p2_path.exists()}, p3={p3_path} exists={p3_path.exists()}"
            )

        return [{
            "p1": str(p1_path),
            "p2": str(p2_path),
            "p3": str(p3_path),
            "sample_id": sample_id,
            "region_id": region_id,
            "crop_method": method,
        }]

    p2_root = Path(args.p2_root).expanduser().resolve()
    p1_root = Path(args.p1_root).expanduser().resolve()
    p3_root = Path(args.p3_root).expanduser().resolve()

    if not p2_root.exists():
        raise FileNotFoundError(f"--p2-root does not exist: {p2_root}")

    triplets = []
    for p2_path in sorted(p2_root.rglob("*__r*.png")):
        try:
            sample_id, region_id = _parse_sample_region_from_filename(str(p2_path))
            method = _normalize_choice(
                "crop-method", p2_path.parent.name, CROP_METHOD_MAP, "GRID, SUPERPIXEL, SAM"
            )
        except Exception:
            continue

        p3_path = p3_root / method.lower() / p2_path.name
        p1_path = p1_root / method.lower() / f"{sample_id}_{method.lower()}_img_edge_number.png"

        if p1_path.exists() and p3_path.exists():
            triplets.append({
                "p1": str(p1_path),
                "p2": str(p2_path),
                "p3": str(p3_path),
                "sample_id": sample_id,
                "region_id": region_id,
                "crop_method": method,
            })

    if args.max_items is not None:
        triplets = triplets[: args.max_items]

    if not triplets:
        raise ValueError("No valid p1/p2/p3 triplets discovered.")

    return triplets


def build_messages(args: argparse.Namespace, item: dict):
    p1 = _normalize_image_ref(item["p1"])
    p2 = _normalize_image_ref(item["p2"])
    p3 = _normalize_image_ref(item["p3"])

    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)

    md = _resolve_metadata(
        args=args,
        sample_id=item["sample_id"],
        method=item["crop_method"],
        region_id=item["region_id"],
    )
    method_definition = args.method_definition or METHOD_DEFINITION_MAP[md["crop_method"]]

    user_prompt = user_template.format(
        time=md["time"],
        frequency=md["frequency"],
        phoneme=md["phoneme"],
        feature=md["feature"],
        crop_method=md["crop_method"],
        method_definition=method_definition,
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "P1 (full fake spectrogram):"},
                {"type": "image", "image": p1},
                {"type": "text", "text": "P2 (fake cropped region):"},
                {"type": "image", "image": p2},
                {"type": "text", "text": "P3 (real aligned cropped region):"},
                {"type": "image", "image": p3},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages, md


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local HF Qwen-VL prompt for spectrogram artifact analysis."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path. Default: local Qwen3-VL-30B-A3B-Thinking.")

    parser.add_argument("--p1", default=None, help="Optional single-item P1 path/URL.")
    parser.add_argument("--p2", default=None, help="Optional single-item P2 path/URL.")
    parser.add_argument("--p3", default=None, help="Optional single-item P3 path/URL.")

    parser.add_argument("--p1-root", default=DEFAULT_P1_ROOT, help="Root for P1 files.")
    parser.add_argument("--p2-root", default=DEFAULT_P2_ROOT, help="Root for P2 files to discover.")
    parser.add_argument("--p3-root", default=DEFAULT_P3_ROOT, help="Root for P3 files.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument(
        "--meta-csv",
        default=DEFAULT_META_CSV,
        help="CSV path with sample_id/method/region_id/T/F/P/feature.",
    )

    parser.add_argument(
        "--system-file",
        default=None,
        help=("Path to system prompt txt. Default: " f"{DEFAULT_SYSTEM_FILE.as_posix()}"),
    )
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=(
            "Path to user prompt template txt. Supports placeholders: "
            "{time}, {frequency}, {phoneme}, {feature}, {crop_method}, {method_definition}. "
            f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}"
        ),
    )

    parser.add_argument("--crop-method", default=None, help="Optional crop method override.")
    parser.add_argument("--time", default=None, help="Optional time override.")
    parser.add_argument("--frequency", default=None, help="Optional frequency override.")
    parser.add_argument("--phoneme", default=None, help="Optional phoneme override.")
    parser.add_argument("--feature", default=None, help="Optional feature override.")
    parser.add_argument("--method-definition", default=None, help="Optional crop method definition override.")

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype, e.g., auto, float16, bfloat16.")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of items to generate per forward pass.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling. If false, decoding is deterministic.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Per-sample output root.")
    parser.add_argument("--output-file", default=None, help="Single-item output file.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat batch output jsonl file.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Regenerate outputs even if per-sample region records already exist.",
    )
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

    do_sample = temperature > 0.0
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def _generate_batch(model, processor, batch_messages, max_new_tokens, do_sample, temperature, top_p):
    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    do_sample = temperature > 0.0
    generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _write_sample_grouped_json(output_dir: Path, records_by_sample: dict):
    for sample_id, records in records_by_sample.items():
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Keep the latest record per region_id (newly generated records override older ones).
        by_region = {}
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            by_region[int(rid)] = rec
        records_sorted = [by_region[rid] for rid in sorted(by_region.keys())]
        payload = {
            "sample_id": sample_id,
            "num_regions": len(records_sorted),
            "regions": records_sorted,
        }

        out_file = sample_dir / "json"
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_records_by_sample(output_dir: Path) -> dict:
    records_by_sample = defaultdict(list)
    if not output_dir.exists():
        return records_by_sample

    for p in output_dir.glob("*/json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sample_id = obj.get("sample_id")
        regions = obj.get("regions")
        if not sample_id or not isinstance(regions, list):
            continue
        for rec in regions:
            if isinstance(rec, dict) and "region_id" in rec:
                records_by_sample[str(sample_id)].append(rec)
    return records_by_sample


def _existing_done_keys(records_by_sample: dict) -> set:
    done = set()
    for sample_id, records in records_by_sample.items():
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            done.add((str(sample_id), int(rid)))
    return done


def main():
    args = parse_args()
    items = _discover_triplets(args)
    output_root = Path(args.output_dir).expanduser().resolve()

    # Resume behavior: load existing outputs and skip completed regions unless --overwrite.
    existing_records = defaultdict(list)
    if not args.overwrite:
        existing_records = _load_existing_records_by_sample(output_root)
        done_keys = _existing_done_keys(existing_records)
        before = len(items)
        items = [
            it for it in items
            if (str(it["sample_id"]), int(it["region_id"])) not in done_keys
        ]
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
        jsonl_fp = out_path.open(mode, encoding="utf-8")

    records_by_sample = defaultdict(list)
    if not args.overwrite:
        for sid, recs in existing_records.items():
            records_by_sample[sid].extend(recs)

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
                    "crop_method": md["crop_method"],
                    "time": md["time"],
                    "frequency": md["frequency"],
                    "phoneme": md["phoneme"],
                    "feature": md["feature"],
                    "p1": item["p1"],
                    "p2": item["p2"],
                    "p3": item["p3"],
                    "response": output_text,
                }

                records_by_sample[record["sample_id"]].append(record)

                print(f"[{idx}/{len(items)}] {record['sample_id']}__r{record['region_id']}")
                print(output_text)

                if jsonl_fp is not None:
                    jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

                if len(items) == 1 and args.output_file:
                    out_file = Path(args.output_file).expanduser().resolve()
                    out_file.parent.mkdir(parents=True, exist_ok=True)
                    out_file.write_text(output_text, encoding="utf-8")
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()

    _write_sample_grouped_json(output_root, records_by_sample)


if __name__ == "__main__":
    main()
