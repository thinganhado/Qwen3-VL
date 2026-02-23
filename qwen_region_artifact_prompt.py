#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"

DEFAULT_META_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_phone_table_topk3.csv"
DEFAULT_MFA_JSON_ROOT = "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned/"
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid/"
DEFAULT_OUTPUT_DIR = "/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/outputs"
DEFAULT_MODEL_ID = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Thinking/"


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


def _resolve_spec_image_path(spec_root: Path, sample_id: str, method: str) -> Optional[Path]:
    method_l = str(method or "").strip().lower()

    # Prefer exact, deterministic names first.
    exact_candidates = [
        spec_root / f"{sample_id}_{method_l}_img_edge_number_axes.png",
        spec_root / f"{sample_id}_{method_l}_img_edge_number.png",
    ]
    if method_l == "grid":
        exact_candidates = [
            spec_root / f"{sample_id}_grid_img_edge_number_axes.png",
            spec_root / f"{sample_id}_grid_img_edge_number.png",
        ] + exact_candidates

    for c in exact_candidates:
        if c.exists():
            return c

    # Then handle parameterized names like superpixel_n40_c20.0 / grid_n4 / sam_*
    patterns = [
        f"{sample_id}_{method_l}*_img_edge_number_axes.png",
        f"{sample_id}_{method_l}*_img_edge_number.png",
    ]

    for pat in patterns:
        matches = sorted(spec_root.glob(pat))
        if matches:
            return matches[0]

    # Final fallback: any image for this sample id.
    for pat in (f"{sample_id}_*_img_edge_number_axes.png", f"{sample_id}_*_img_edge_number.png"):
        matches = sorted(spec_root.glob(pat))
        if matches:
            return matches[0]

    return None


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

            method = str(row.get("method", "grid")).strip() or "grid"
            method_u = method.upper()
            p1_path = _resolve_spec_image_path(spec_root, sample_id, method)
            mfa_json_path = mfa_root / f"{sample_id}.json"
            if p1_path is None or not p1_path.exists():
                continue

            items.append(
                {
                    "sample_id": sample_id,
                    "sample_id_raw": sample_id_raw,
                    "region_id": region_id,
                    "time": str(row.get("T", "")).strip(),
                    "frequency": str(row.get("F", "")).strip(),
                    "phonetic": str(row.get("P_type", "")).strip(),
                    "crop_method": method_u,
                    "p1": str(p1_path),
                    "mfa_json": str(mfa_json_path),
                }
            )

    if args.max_items is not None:
        items = items[: args.max_items]

    if not items:
        raise ValueError("No valid items discovered from CSV + spec root.")

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
                    "time": item.get("time", ""),
                    "frequency": item.get("frequency", ""),
                    "phonetic": item.get("phonetic", ""),
                    "transcript": transcript_text,
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
                {"type": "text", "text": f"Transcript (word tier): {transcript_text}" if transcript_text else "Transcript (word tier): [missing]"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    md = {
        "sample_id": item["sample_id"],
        "region_id": item["region_id"],
        "crop_method": item["crop_method"],
        "transcript": transcript_text,
    }
    return messages, md


def parse_args():
    parser = argparse.ArgumentParser(description="Run local HF Qwen-VL prompt for spectrogram artifact analysis.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id or local model path.")
    parser.add_argument(
        "--backend",
        default="transformers",
        choices=["transformers", "vllm"],
        help="Inference backend. Use vllm for FP8 checkpoints.",
    )

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
            "{ID}, {sample_id}, {time}, {frequency}, {phonetic}, {transcript}. "
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
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="vLLM tensor parallel size. Default: GPU count.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization fraction.")
    parser.add_argument("--vllm-enforce-eager", action="store_true", help="Enable vLLM eager mode.")
    parser.add_argument("--vllm-max-model-len", type=int, default=None, help="Optional vLLM max model length.")
    parser.add_argument("--vllm-max-images-per-prompt", type=int, default=1, help="vLLM multimodal image limit per prompt.")

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


def _import_vllm_deps():
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as e:
        raise ImportError("qwen_vl_utils is required for --backend vllm. Install qwen_vl_utils>=0.0.14.") from e
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise ImportError("vllm is required for --backend vllm.") from e
    return LLM, SamplingParams, process_vision_info


def _prepare_inputs_for_vllm(messages, processor, process_vision_info):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {
        "return_video_kwargs": True,
        "return_video_metadata": True,
    }
    image_processor = getattr(processor, "image_processor", None)
    patch_size = getattr(image_processor, "patch_size", None)
    if patch_size is not None:
        kwargs["image_patch_size"] = patch_size

    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, **kwargs)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs or {},
    }


def _build_vllm_sampling_params(args, SamplingParams):
    sample_flag = bool(args.do_sample and args.temperature > 0.0)
    temperature = args.temperature if sample_flag else 0.0
    top_p = args.top_p if sample_flag else 1.0
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )


def _generate_batch_vllm(llm, processor, process_vision_info, batch_messages, sampling_params):
    batch_inputs = [
        _prepare_inputs_for_vllm(messages=m, processor=processor, process_vision_info=process_vision_info)
        for m in batch_messages
    ]
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    texts = []
    for out in outputs:
        if getattr(out, "outputs", None):
            texts.append(out.outputs[0].text)
        else:
            texts.append("")
    return texts


def _infer_tp_size_from_env() -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        parts = [p.strip() for p in cvd.split(",") if p.strip()]
        if parts:
            return len(parts)
    return 1


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
            rid = rec.get("region_id")
            if not sample_id or rid is None:
                continue
            try:
                rid = int(rid)
            except Exception:
                continue
            method = str(rec.get("crop_method", "GRID")).upper()
            rec["sample_id"] = sample_id
            rec["region_id"] = rid
            rec["crop_method"] = method
            records_by_bucket[(method, sample_id)].append(rec)
    return records_by_bucket


def _merge_records_by_bucket(dst: dict, src: dict) -> None:
    seen = _existing_done_keys(dst)
    for (method, sample_id), records in src.items():
        bucket = (str(method).upper(), str(sample_id))
        for rec in records:
            rid = rec.get("region_id")
            if rid is None:
                continue
            try:
                rid = int(rid)
            except Exception:
                continue
            key = (bucket[1], rid, bucket[0])
            if key in seen:
                continue
            seen.add(key)
            rec["sample_id"] = bucket[1]
            rec["region_id"] = rid
            rec["crop_method"] = bucket[0]
            dst[bucket].append(rec)


def main():
    args = parse_args()
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
        items = [it for it in items if (str(it["sample_id"]), int(it["region_id"]), str(it.get("crop_method", "GRID")).upper()) not in done_keys]
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

    model = None
    llm = None
    sampling_params = None
    process_vision_info = None

    if args.backend == "transformers":
        torch_dtype = _resolve_torch_dtype(args.dtype)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            torch_dtype=torch_dtype,
            attn_implementation=args.attn_implementation,
            device_map=args.device_map,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    elif args.backend == "vllm":
        # Prevent CUDA re-init errors in vLLM worker subprocesses.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        LLM, SamplingParams, process_vision_info = _import_vllm_deps()
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

        tp_size = args.tensor_parallel_size
        if tp_size is None:
            tp_size = _infer_tp_size_from_env()

        llm_kwargs = {
            "model": args.model_id,
            "trust_remote_code": True,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "enforce_eager": args.vllm_enforce_eager,
            "limit_mm_per_prompt": {"image": args.vllm_max_images_per_prompt},
            "seed": 0,
        }
        if args.vllm_max_model_len is not None:
            llm_kwargs["max_model_len"] = args.vllm_max_model_len

        llm = LLM(**llm_kwargs)
        sampling_params = _build_vllm_sampling_params(args, SamplingParams)
    else:
        raise ValueError(f"Unsupported --backend: {args.backend}")

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
            batch_items = items[batch_start: batch_start + args.batch_size]
            batch_built = [build_messages(args, item) for item in batch_items]
            batch_messages = [x[0] for x in batch_built]
            batch_md = [x[1] for x in batch_built]

            if args.print_messages:
                for m in batch_messages:
                    print(m)

            if args.backend == "vllm":
                batch_outputs = _generate_batch_vllm(
                    llm=llm,
                    processor=processor,
                    process_vision_info=process_vision_info,
                    batch_messages=batch_messages,
                    sampling_params=sampling_params,
                )
            elif len(batch_messages) == 1:
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
                    "crop_method": item["crop_method"],
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

