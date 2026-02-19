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
DEFAULT_SYSTEM_FILE = THIS_DIR / "query2_prompts" / "query2_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "query2_prompts" / "query2_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_gt_with_transcript.csv"
DEFAULT_IMAGE_FOLDER = ""
DEFAULT_OUTPUT_DIR = "/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/query2_outputs"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Thinking/"
DEFAULT_MODEL_PATHS = {
    "qwen3_30b_a3b_instruct": "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-30B-A3B-Instruct/",
    "qwen3_8b_stage1_merged": "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_merged_Qwen3-VL-8B-Instruct/",
    "qwen25_7b_stage1_merged": "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_merged_Qwen2.5-VL-7B-Instruct/",
}
DEFAULT_MODEL_KEY = "qwen3_30b_a3b_instruct"


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
    return _load_text_file(Path(args.system_file) if args.system_file else DEFAULT_SYSTEM_FILE, "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.user_template_file) if args.user_template_file else DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")


def _resolve_model_id(args: argparse.Namespace) -> str:
    if args.model_id:
        return args.model_id
    return DEFAULT_MODEL_PATHS[args.model_key]


def _resolve_image_path(img_path_raw: str, image_folder: str, csv_dir: Path):
    p = Path(str(img_path_raw)).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if image_folder:
            candidates.append(Path(image_folder).expanduser() / p)
        candidates.append(csv_dir / p)
        candidates.append(Path.cwd() / p)

    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        if resolved.exists():
            return resolved
    return None


def _discover_items(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    csv_dir = meta_csv.parent

    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")

    items = []
    used_sample_ids = set()
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            img_path_raw = str(row.get("img_path", "")).strip()
            prompt1_output = str(row.get("regions", "")).strip()
            transcript = str(row.get("transcript", "")).strip()
            prompt2_target = str(row.get("prompt2_target", "")).strip()
            if not img_path_raw:
                continue

            img_path = _resolve_image_path(img_path_raw, args.image_folder, csv_dir)
            if img_path is None:
                continue
            if args.img_stem_contains and args.img_stem_contains not in img_path.stem:
                continue

            sample_id = str(row.get("id", "")).strip() or img_path.stem or f"row_{row_idx}"
            unique_sample_id = sample_id
            if unique_sample_id in used_sample_ids:
                unique_sample_id = f"{sample_id}__row{row_idx}"
            used_sample_ids.add(unique_sample_id)

            items.append(
                {
                    "sample_id": unique_sample_id,
                    "crop_method": "GRID",
                    "p1": str(img_path),
                    "img_path": str(img_path),
                    "prompt1_output": prompt1_output,
                    "transcript": transcript,
                    "prompt2_target": prompt2_target,
                }
            )

    items = sorted(items, key=lambda x: x["sample_id"])
    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from CSV + spec root.")

    return items


def build_messages(args: argparse.Namespace, item: dict):
    p1 = _normalize_image_ref(item["p1"])
    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)

    transcript_text = str(item.get("transcript", "")).strip()
    prompt1_output = item.get("prompt1_output", "")
    user_prompt = user_template.format_map(
        defaultdict(
            str,
            {
                "prompt1_output": prompt1_output,
                "transcript": transcript_text,
                "sample_id": item["sample_id"],
                "sample_id_raw": item.get("sample_id_raw", item["sample_id"]),
            },
        )
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Spectrogram ({item['crop_method']}):"},
                {"type": "image", "image": p1},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    md = {
        "sample_id": item["sample_id"],
        "crop_method": item["crop_method"],
        "prompt1_output": prompt1_output,
        "transcript": transcript_text,
        "prompt2_target": item.get("prompt2_target", ""),
    }
    return messages, md


def parse_args():
    parser = argparse.ArgumentParser(description="Run local HF Qwen-VL query2 prompt for spectrogram artifact analysis.")
    parser.add_argument(
        "--model-key",
        choices=sorted(DEFAULT_MODEL_PATHS.keys()),
        default=DEFAULT_MODEL_KEY,
        help="Named model option. Ignored when --model-id is explicitly set.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help=f"HF model id or local model path (overrides --model-key). Legacy default was: {DEFAULT_MODEL_ID}",
    )

    parser.add_argument(
        "--meta-csv",
        default=DEFAULT_META_CSV,
        help="CSV with columns: img_path, regions, transcript, prompt2_target.",
    )
    parser.add_argument(
        "--image-folder",
        default=DEFAULT_IMAGE_FOLDER,
        help="Optional base folder for resolving relative img_path entries.",
    )
    parser.add_argument(
        "--img-stem-contains",
        default="_LA_D_",
        help="Only run rows where resolved img_path stem contains this substring. Set empty string to disable.",
    )

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=(
            "Path to user prompt template txt. Supports placeholders: "
            "{prompt1_output}, {transcript}, {sample_id}. "
            f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}"
        ),
    )

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype, e.g., auto, float16, bfloat16.")
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend. Use eager to avoid unstable fused kernels.",
    )
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


def _write_sample_json(output_dir: Path, records_by_bucket: dict):
    for (method, sample_id), records in records_by_bucket.items():
        if not records:
            continue
        method_dir = output_dir / str(method).lower()
        sample_dir = method_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        payload = records[-1]
        (sample_dir / "json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_existing_records_by_sample(output_dir: Path) -> dict:
    records_by_bucket = defaultdict(list)
    if not output_dir.exists():
        return records_by_bucket

    for p in output_dir.glob("*/*/json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        sample_id = str(rec.get("sample_id", "")).strip()
        method = str(rec.get("crop_method", p.parent.parent.name)).upper()
        if not sample_id:
            continue
        rec["sample_id"] = sample_id
        rec["crop_method"] = method
        records_by_bucket[(method, sample_id)].append(rec)
    return records_by_bucket


def _existing_done_keys(records_by_bucket: dict) -> set:
    done = set()
    for (method, sample_id), records in records_by_bucket.items():
        if records:
            done.add((str(sample_id), str(method).upper()))
    return done


def _load_existing_records_from_jsonl(jsonl_path: Path) -> dict:
    records_by_bucket = defaultdict(list)
    if not jsonl_path.exists():
        return records_by_bucket

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            sample_id = str(rec.get("sample_id", "")).strip()
            if not sample_id:
                continue
            method = str(rec.get("crop_method", "GRID")).upper()
            rec["sample_id"] = sample_id
            rec["crop_method"] = method
            records_by_bucket[(method, sample_id)].append(rec)
    return records_by_bucket


def _merge_records_by_bucket(dst: dict, src: dict) -> None:
    seen = _existing_done_keys(dst)
    for (method, sample_id), records in src.items():
        bucket = (str(method).upper(), str(sample_id))
        if not records:
            continue
        key = (bucket[1], bucket[0])
        if key in seen:
            continue
        seen.add(key)
        rec = records[-1]
        rec["sample_id"] = bucket[1]
        rec["crop_method"] = bucket[0]
        dst[bucket].append(rec)


def main():
    args = parse_args()
    args.model_id = _resolve_model_id(args)
    items = _discover_items(args)
    output_root = Path(args.output_dir).expanduser().resolve()

    existing_records = defaultdict(list)
    existing_jsonl_records = defaultdict(list)
    if not args.overwrite:
        existing_records = _load_existing_records_by_sample(output_root)
        done_keys = _existing_done_keys(existing_records)
        if args.output_jsonl:
            jsonl_path = Path(args.output_jsonl).expanduser().resolve()
            existing_jsonl_records = _load_existing_records_from_jsonl(jsonl_path)
            done_keys.update(_existing_done_keys(existing_jsonl_records))
        before = len(items)
        items = [it for it in items if (str(it["sample_id"]), str(it.get("crop_method", "GRID")).upper()) not in done_keys]
        skipped = before - len(items)
        if skipped > 0:
            print(f"[resume] skipped_existing_samples={skipped}")
        if len(items) == 0:
            print("[resume] no pending samples; nothing to generate.")
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
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
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
        _merge_records_by_bucket(records_by_bucket, existing_records)
        _merge_records_by_bucket(records_by_bucket, existing_jsonl_records)

    try:
        for batch_start in range(0, len(items), args.batch_size):
            batch_items = items[batch_start : batch_start + args.batch_size]
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
                    "crop_method": md["crop_method"],
                    "prompt1_output": md["prompt1_output"],
                    "transcript": md["transcript"],
                    "prompt2_target": md["prompt2_target"],
                    "p1": item["p1"],
                    "response": output_text,
                }

                bucket = (record["crop_method"], record["sample_id"])
                records_by_bucket[bucket].append(record)

                print(f"[{idx}/{len(items)}] {record['sample_id']}")
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

    _write_sample_json(output_root, records_by_bucket)


if __name__ == "__main__":
    main()
