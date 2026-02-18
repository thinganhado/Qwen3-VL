#!/usr/bin/env python3
import argparse
import csv
import json
import os
from fnmatch import fnmatch
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "baseline_prompts" / "baseline_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "baseline_prompts" / "baseline_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_gt.csv"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-235B-A22B-Instruct/"
DEFAULT_OUTPUT_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_strongVLM/"


def _load_text_file(path: Path, field_name: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{field_name} file does not exist: {resolved}")
    return resolved.read_text(encoding="utf-8").strip()


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.system_file) if args.system_file else DEFAULT_SYSTEM_FILE, "--system-file")


def _resolve_user_template(args: argparse.Namespace) -> str:
    return _load_text_file(Path(args.user_template_file) if args.user_template_file else DEFAULT_USER_TEMPLATE_FILE, "--user-template-file")


def _discover_items(args: argparse.Namespace):
    meta_csv = Path(args.meta_csv).expanduser().resolve()
    if not meta_csv.exists():
        raise FileNotFoundError(f"--meta-csv does not exist: {meta_csv}")

    items = []
    with meta_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path_raw = str(row.get("img_path", "")).strip()
            if not img_path_raw:
                continue
            img_path = Path(img_path_raw).expanduser().resolve()
            if not img_path.exists():
                continue

            sample_id = img_path.stem
            if args.sample_id_glob and not fnmatch(sample_id, args.sample_id_glob):
                continue
            gt_regions = str(row.get("regions", "")).strip()
            items.append(
                {
                    "sample_id": sample_id,
                    "img_path": str(img_path),
                    "gt_regions": gt_regions,
                }
            )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from stage1_gt.csv.")

    return sorted(items, key=lambda x: x["sample_id"])


def _build_messages(args: argparse.Namespace, item: dict):
    system_prompt = _resolve_system_prompt(args)
    user_prompt = _resolve_user_template(args)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Spectrogram (GRID with axes):"},
                {"type": "image", "image": item["img_path"]},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages


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


def _load_existing_sample_ids(output_jsonl: Path) -> set:
    done = set()
    if not output_jsonl.exists():
        return done

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            sample_id = str(rec.get("sample_id", "")).strip()
            if sample_id:
                done.add(sample_id)
    return done


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline Qwen-VL prompt on stage1_gt.csv image rows.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")
    parser.add_argument("--meta-csv", default=DEFAULT_META_CSV, help="CSV path containing img_path and regions columns.")
    parser.add_argument(
        "--sample-id-glob",
        default="*_LA_D_*",
        help="Only include rows whose img_path stem matches this glob. Use empty string to disable.",
    )

    parser.add_argument("--system-file", default=None, help=f"Path to system prompt txt. Default: {DEFAULT_SYSTEM_FILE.as_posix()}")
    parser.add_argument("--user-template-file", default=None, help=f"Path to user prompt txt. Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}")

    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for discovered items.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split discovered items across N shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output root directory.")
    parser.add_argument("--output-jsonl", default=None, help="Optional flat output jsonl path.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Regenerate outputs even if already present.")
    parser.add_argument("--print-messages", action="store_true", help="Print built messages before generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    items = _discover_items(args)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
        print(f"[shard] shard_id={args.shard_id}/{args.num_shards} items={len(items)}")
        if not items:
            raise ValueError("No items assigned to this shard.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else output_dir / "qwen_baseline_outputs.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        done = _load_existing_sample_ids(output_jsonl)
        before = len(items)
        items = [it for it in items if it["sample_id"] not in done]
        skipped = before - len(items)
        if skipped > 0:
            print(f"[resume] skipped_existing_samples={skipped}")
        if not items:
            print("[resume] no pending samples; nothing to generate.")
            return

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    mode = "w" if args.overwrite else "a"
    with output_jsonl.open(mode, encoding="utf-8", buffering=1) as jsonl_fp:
        for idx, item in enumerate(items, start=1):
            messages = _build_messages(args, item)
            if args.print_messages:
                print(messages)

            output_text = _generate_one(
                model=model,
                processor=processor,
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "sample_id": item["sample_id"],
                "img_path": item["img_path"],
                "gt_regions": item["gt_regions"],
                "response": output_text,
            }

            sample_dir = output_dir / item["sample_id"]
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

            jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fp.flush()
            try:
                os.fsync(jsonl_fp.fileno())
            except OSError:
                pass

            print(f"[{idx}/{len(items)}] {item['sample_id']}")
            print(output_text)


if __name__ == "__main__":
    main()
