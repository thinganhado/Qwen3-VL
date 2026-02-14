#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_FILE = THIS_DIR / "prompts" / "region_forensics_system.txt"
DEFAULT_USER_TEMPLATE_FILE = THIS_DIR / "prompts" / "region_forensics_user.txt"

DEFAULT_USER_TEMPLATE = (
    "This region corresponds to {time} section, {frequency} frequency band, and {phoneme}.\n"
    "P2 is produced by {crop_method}. Method definition: {method_definition}\n"
    "Compare P2 vs P3 and generate E in the required format."
)

METHOD_DEFINITION_MAP = {
    "GRID": "The crop is taken from a fixed square cell in an NxN grid over the spectrogram.",
    "SUPERPIXEL": "The crop is taken from an irregular region formed by grouping nearby pixels with similar appearance.",
    "SAM": "The crop is taken from a region that follows the visible edges of the pattern as closely as possible.",
}


def _normalize_image_ref(value: str) -> str:
    if value.startswith(("http://", "https://", "file://", "data:")):
        return value
    p = Path(value).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image path does not exist: {p}")
    return p.as_uri()


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


def build_messages(args: argparse.Namespace):
    p1 = _normalize_image_ref(args.p1)
    p2 = _normalize_image_ref(args.p2)
    p3 = _normalize_image_ref(args.p3)

    system_prompt = _resolve_system_prompt(args)
    user_template = _resolve_user_template(args)
    method_definition = args.method_definition or METHOD_DEFINITION_MAP[args.crop_method]

    user_prompt = user_template.format(
        time=args.time,
        frequency=args.frequency,
        phoneme=args.phoneme,
        crop_method=args.crop_method,
        method_definition=method_definition,
    )

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local HF Qwen-VL prompt for spectrogram artifact analysis."
    )
    parser.add_argument("--model-id", required=True, help="HF model id or local model path.")
    parser.add_argument("--p1", required=True, help="Path/URL for P1 full fake spectrogram.")
    parser.add_argument("--p2", required=True, help="Path/URL for P2 fake crop.")
    parser.add_argument("--p3", required=True, help="Path/URL for P3 real aligned crop.")
    parser.add_argument(
        "--system-file",
        default=None,
        help=(
            "Path to system prompt txt. Default: "
            f"{DEFAULT_SYSTEM_FILE.as_posix()}"
        ),
    )
    parser.add_argument(
        "--user-template-file",
        default=None,
        help=(
            "Path to user prompt template txt. Supports placeholders: "
            "{time}, {frequency}, {phoneme}, {crop_method}, {method_definition}. "
            f"Default: {DEFAULT_USER_TEMPLATE_FILE.as_posix()}"
        ),
    )

    parser.add_argument("--time", required=True, choices=["S", "NS"], help="Region time label.")
    parser.add_argument(
        "--frequency", required=True, choices=["L", "M", "H"], help="Region frequency label."
    )
    parser.add_argument("--phoneme", required=True, choices=["C", "V"], help="Region phoneme label.")
    parser.add_argument(
        "--crop-method", required=True, choices=["GRID", "SUPERPIXEL", "SAM"], help="Crop method."
    )
    parser.add_argument(
        "--method-definition",
        default=None,
        help="Short description of how the crop method defines P2.",
    )

    parser.add_argument("--device-map", default="auto", help="Transformers device_map.")
    parser.add_argument("--dtype", default="auto", help="Model dtype, e.g., auto, float16, bfloat16.")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--output-file", default=None, help="Optional file to save model output.")
    parser.add_argument(
        "--print-messages",
        action="store_true",
        help="Print the built messages before generation.",
    )
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


def main():
    args = parse_args()
    messages = build_messages(args)

    if args.print_messages:
        print(messages)

    torch_dtype = _resolve_torch_dtype(args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    do_sample = args.temperature > 0.0
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    generated_ids = model.generate(**inputs, **generate_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(output_text)

    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
