#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "query2_prompts" / "query2_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "query2_prompts" / "query2_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_gt_with_transcript.csv"
DEFAULT_IMAGE_FOLDER = ""
DEFAULT_P1_JSON_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/Qwen3-VL-30B_test"
DEFAULT_OUTPUT_DIR = "/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/query2_outputs_from_p1json"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-30B-A3B-Instruct/"
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
    return resolved.read_text(encoding="utf-8-sig").strip()


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


def _extract_numbers_list_text(raw_response: str) -> str:
    text = str(raw_response or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    nums = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            for x in parsed:
                s = str(x).strip()
                if re.fullmatch(r"-?\d+", s):
                    nums.append(int(s))
    except Exception:
        pass

    if not nums:
        nums = [int(m.group(0)) for m in re.finditer(r"-?\d+", text)]

    return "[" + ", ".join(str(n) for n in nums) + "]"


def _sample_id_candidates(row: dict, img_path: Path, row_idx: int):
    candidates = []

    # Prefer explicit sample id columns when available.
    for key in ("sample_id", "sample", "sampleid"):
        v = str(row.get(key, "")).strip()
        if v:
            candidates.append(v)

    row_id = str(row.get("id", "")).strip()
    if row_id:
        candidates.append(row_id)

    stem = img_path.stem.strip()
    if stem:
        candidates.append(stem)
        # Some CSV image stems include split tags that are not used in prompt1 output folder names.
        candidates.append(stem.replace("_LA_D_", "_"))
        candidates.append(stem.replace("_LA_D_", ""))

    if not candidates:
        candidates.append(f"row_{row_idx}")

    seen = set()
    deduped = []
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def _load_prompt1_output(sample_id_candidates, root: Path):
    last_missing = None
    for sample_id in sample_id_candidates:
        p = root / sample_id / "json"
        if not p.exists():
            last_missing = p
            continue
        payload = json.loads(p.read_text(encoding="utf-8-sig"))
        response = payload.get("response", "")
        return sample_id, _extract_numbers_list_text(response)
    raise FileNotFoundError(f"Prompt1 JSON not found for candidates={sample_id_candidates}. Last tried: {last_missing}")


def _discover_items(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    csv_dir = meta_csv.parent
    p1_root = Path(args.prompt1_json_root).expanduser().resolve()

    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")
    if not p1_root.exists():
        raise FileNotFoundError(f"--prompt1-json-root does not exist: {p1_root}")

    items = []
    used_sample_ids = set()
    skipped_missing_p1 = 0
    missing_examples = []

    with meta_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=1):
            img_path_raw = str(row.get("img_path", "")).strip()
            transcript = str(row.get("transcript", "")).strip()
            prompt2_target = str(row.get("prompt2_target", "")).strip()
            if not img_path_raw:
                continue

            img_path = _resolve_image_path(img_path_raw, args.image_folder, csv_dir)
            if img_path is None:
                continue
            if args.img_stem_contains and args.img_stem_contains not in img_path.stem:
                continue

            sample_id_candidates = _sample_id_candidates(row, img_path, row_idx)
            sample_id = sample_id_candidates[0]
            unique_sample_id = sample_id
            if unique_sample_id in used_sample_ids:
                unique_sample_id = f"{sample_id}__row{row_idx}"
            used_sample_ids.add(unique_sample_id)

            try:
                matched_sample_id, prompt1_output = _load_prompt1_output(sample_id_candidates, p1_root)
            except FileNotFoundError:
                skipped_missing_p1 += 1
                if len(missing_examples) < 5:
                    missing_examples.append(
                        {
                            "row": row_idx,
                            "img_stem": img_path.stem,
                            "candidates": sample_id_candidates,
                        }
                    )
                if args.strict_prompt1_json:
                    raise
                continue

            items.append(
                {
                    "sample_id": unique_sample_id,
                    "sample_id_raw": matched_sample_id,
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

    if skipped_missing_p1 > 0:
        print(f"[warn] skipped_missing_prompt1_json={skipped_missing_p1}")
        if missing_examples:
            print(f"[warn] example_missing_prompt1_json={json.dumps(missing_examples, ensure_ascii=True)}")
    if not items:
        raise ValueError("No valid items discovered from CSV + prompt1 JSON root.")
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
        "sample_id_raw": item.get("sample_id_raw", item["sample_id"]),
        "crop_method": item["crop_method"],
        "prompt1_output": prompt1_output,
        "transcript": transcript_text,
        "prompt2_target": item.get("prompt2_target", ""),
    }
    return messages, md


def parse_args():
    parser = argparse.ArgumentParser(description="Run local HF Qwen-VL query2 prompt with prompt1_output loaded from baseline JSON.")
    parser.add_argument(
        "--model-key",
        choices=sorted(DEFAULT_MODEL_PATHS.keys()),
        default=DEFAULT_MODEL_KEY,
        help="Named model option. Ignored when --model-id is explicitly set.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path (overrides --model-key).")
    parser.add_argument("--meta-csv", default=DEFAULT_META_CSV, help="CSV with at least img_path, transcript, prompt2_target.")
    parser.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER, help="Optional base folder for relative img_path entries.")
    parser.add_argument("--prompt1-json-root", default=DEFAULT_P1_JSON_ROOT, help="Root containing <sample_id>/json from prompt1 runs.")
    parser.add_argument("--strict-prompt1-json", action="store_true", help="Fail immediately if any sample prompt1 JSON is missing.")
    parser.add_argument("--img-stem-contains", default="_LA_D_", help="Only run rows where resolved img_path stem contains this substring.")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument("--user-template-file", default=None, help=f"Path to user template txt. Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--print-messages", action="store_true")
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
    def _safe_path_component(value: str, max_len: int = 120) -> str:
        text = str(value or "").strip()
        text = re.sub(r"[\\/:\*\?\"<>\|\s]+", "_", text)
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
        if not text:
            text = "sample"
        if len(text) <= max_len:
            return text
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
        keep = max_len - 11
        return f"{text[:keep]}_{digest}"

    for (method, sample_id), records in records_by_bucket.items():
        if not records:
            continue
        method_dir = output_dir / str(method).lower()
        sample_dir = method_dir / _safe_path_component(sample_id)
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
                    "sample_id_raw": md["sample_id_raw"],
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
