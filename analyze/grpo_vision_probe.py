#!/usr/bin/env python3
import argparse
import json
import random
import traceback
import types
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def completion_to_text(completion: Any, extract_text_content) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return extract_text_content(completion[0].get("content", ""))
    return str(completion)


def _find_image_keys(obj: Any, found: Optional[set] = None) -> set:
    if found is None:
        found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"image", "images", "pixel_values", "image_grid_thw"}:
                found.add(k)
            _find_image_keys(v, found)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _find_image_keys(item, found)
    return found


@dataclass
class RunProbeResult:
    mode: str
    success: bool
    error: Optional[str]
    dataset_keys: List[str]
    dataset_has_image: bool
    generation_call_count: int
    generation_first_input_type: Optional[str]
    generation_first_image_keys: List[str]
    processor_call_count: int
    processor_calls_with_images: int
    processor_chat_template_calls: int
    processor_chat_template_calls_with_images: int
    reward_kwargs_has_image: bool
    reward_kwargs_keys: List[str]
    first_completions: List[str]
    first_reward_means: List[Optional[float]]


def mutate_image_column(dataset, mode: str):
    if mode == "normal":
        return dataset

    if mode == "broken":
        def _break(ex, idx):
            if idx == 0:
                ex["image"] = "__BROKEN_IMAGE_OBJECT__"
            return ex

        return dataset.map(_break, with_indices=True)

    if mode == "shuffled":
        n = len(dataset)
        if n <= 1:
            return dataset
        images = list(dataset["image"])
        rotated = images[1:] + images[:1]
        ds_no_image = dataset.remove_columns(["image"])
        return ds_no_image.add_column("image", rotated)

    raise ValueError(f"Unsupported mode: {mode}")


def run_probe(
    *,
    model: str,
    subset_size: int,
    seed: int,
    mode: str,
    output_dir: str,
    use_vllm: bool,
    vllm_server_host: Optional[str],
) -> RunProbeResult:
    from datasets import load_dataset
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer
    from train_grpo import (
        _extract_text_content,
        build_sparc_reward_functions,
        to_grpo_prompt_format,
    )

    set_seed(seed)

    raw = load_dataset("lkaesberg/SPaRC", "all", split=f"train[:{subset_size}]")
    processor = AutoProcessor.from_pretrained(model)
    if not hasattr(processor, "tokenizer"):
        raise ValueError("Model processor must expose `tokenizer` for prompt truncation.")

    dataset = to_grpo_prompt_format(
        raw,
        processor.tokenizer,
        use_vision_variant=True,
        vision_plot_type="original",
    )
    dataset = mutate_image_column(dataset, mode)

    probe: Dict[str, Any] = {
        "generation_call_count": 0,
        "generation_first_input_type": None,
        "generation_first_image_keys": [],
        "processor_call_count": 0,
        "processor_calls_with_images": 0,
        "processor_chat_template_calls": 0,
        "processor_chat_template_calls_with_images": 0,
        "reward_kwargs_has_image": False,
        "reward_kwargs_keys": [],
        "first_completions": [],
        "first_reward_values": {},
    }

    original_processor_call = processor.__call__

    def wrapped_processor_call(self, *args, **kwargs):
        probe["processor_call_count"] += 1
        if kwargs.get("images") is not None:
            probe["processor_calls_with_images"] += 1
        return original_processor_call(*args, **kwargs)

    processor.__call__ = types.MethodType(wrapped_processor_call, processor)

    if hasattr(processor, "apply_chat_template"):
        original_apply_chat_template = processor.apply_chat_template

        def wrapped_apply_chat_template(*args, **kwargs):
            probe["processor_chat_template_calls"] += 1
            if kwargs.get("images") is not None:
                probe["processor_chat_template_calls_with_images"] += 1
            return original_apply_chat_template(*args, **kwargs)

        processor.apply_chat_template = wrapped_apply_chat_template

    base_rewards = build_sparc_reward_functions(
        list(raw),
        use_vision_variant=True,
        vision_plot_type="original",
    )

    wrapped_rewards = []
    for idx, reward_fn in enumerate(base_rewards):
        def _wrap(i, fn):
            def _wrapped(completions, prompts, **kwargs):
                values = fn(completions, prompts, **kwargs)
                if i not in probe["first_reward_values"]:
                    probe["first_reward_values"][i] = values
                if not probe["reward_kwargs_keys"]:
                    keys = sorted(list(kwargs.keys()))
                    probe["reward_kwargs_keys"] = keys
                    probe["reward_kwargs_has_image"] = "image" in kwargs
                return values

            return _wrapped

        wrapped_rewards.append(_wrap(idx, reward_fn))

    def probe_reward(completions, prompts, **kwargs):
        if not probe["first_completions"]:
            probe["first_completions"] = [completion_to_text(c, _extract_text_content) for c in completions]
            if not probe["reward_kwargs_keys"]:
                keys = sorted(list(kwargs.keys()))
                probe["reward_kwargs_keys"] = keys
                probe["reward_kwargs_has_image"] = "image" in kwargs
        return [0.0] * len(completions)

    wrapped_rewards.append(probe_reward)

    config_kwargs = dict(
        output_dir=output_dir,
        report_to="none",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=False,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        do_eval=False,
        max_completion_length=256,
        max_steps=1,
        num_generations=1,
        num_train_epochs=1,
        reward_weights=[1.0, 0.1, 0.1, 0.1, 0.1, 0.01, 0.0],
        scale_rewards=False,
        loss_type="dr_grpo",
        gradient_checkpointing=False,
    )
    if use_vllm:
        if not vllm_server_host:
            raise ValueError("--vllm_server_host is required when --use_vllm is set")
        config_kwargs.update(
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=vllm_server_host,
            vllm_server_timeout=600,
        )
    else:
        config_kwargs.update(use_vllm=False)

    config = GRPOConfig(**config_kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=config,
        reward_funcs=wrapped_rewards,
        train_dataset=dataset,
        processing_class=processor,
    )

    original_generate = trainer._generate_and_score_completions

    def wrapped_generate(self, *args, **kwargs):
        probe["generation_call_count"] += 1
        if probe["generation_first_input_type"] is None:
            first = args[0] if args else kwargs
            probe["generation_first_input_type"] = type(first).__name__
            keys = sorted(list(_find_image_keys(first)))
            probe["generation_first_image_keys"] = keys
        return original_generate(*args, **kwargs)

    trainer._generate_and_score_completions = types.MethodType(wrapped_generate, trainer)

    try:
        trainer.train()
        success = True
        error = None
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"

    reward_means = []
    for i in range(len(base_rewards)):
        vals = probe["first_reward_values"].get(i)
        reward_means.append(float(sum(vals) / len(vals)) if vals is not None and len(vals) > 0 else None)

    return RunProbeResult(
        mode=mode,
        success=success,
        error=error,
        dataset_keys=sorted(list(dataset.column_names)),
        dataset_has_image=("image" in dataset.column_names),
        generation_call_count=probe["generation_call_count"],
        generation_first_input_type=probe["generation_first_input_type"],
        generation_first_image_keys=probe["generation_first_image_keys"],
        processor_call_count=probe["processor_call_count"],
        processor_calls_with_images=probe["processor_calls_with_images"],
        processor_chat_template_calls=probe["processor_chat_template_calls"],
        processor_chat_template_calls_with_images=probe["processor_chat_template_calls_with_images"],
        reward_kwargs_has_image=probe["reward_kwargs_has_image"],
        reward_kwargs_keys=probe["reward_kwargs_keys"],
        first_completions=probe["first_completions"],
        first_reward_means=reward_means,
    )


def compare_results(normal: RunProbeResult, shuffled: RunProbeResult) -> Dict[str, Any]:
    completions_differ = normal.first_completions != shuffled.first_completions
    rewards_differ = normal.first_reward_means != shuffled.first_reward_means
    return {
        "completions_differ": completions_differ,
        "reward_means_differ": rewards_differ,
        "ab_different": completions_differ or rewards_differ,
    }


def verdict(normal: RunProbeResult, broken: RunProbeResult, ab_cmp: Dict[str, Any]) -> Dict[str, Any]:
    image_seen_generation = bool(normal.generation_first_image_keys)
    processor_used_images = normal.processor_calls_with_images > 0 or normal.processor_chat_template_calls_with_images > 0
    broken_failed = not broken.success
    ab_different = bool(ab_cmp.get("ab_different", False))

    high_conf_used = image_seen_generation and processor_used_images and broken_failed and ab_different
    high_conf_ignored = (not image_seen_generation and not processor_used_images and not broken_failed and not ab_different)

    return {
        "image_seen_generation": image_seen_generation,
        "processor_used_images": processor_used_images,
        "broken_failed": broken_failed,
        "ab_different": ab_different,
        "high_confidence_image_used": high_conf_used,
        "high_confidence_image_ignored": high_conf_ignored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny probe to check whether GRPO vision training actually uses image inputs")
    parser.add_argument("--model", type=str, required=True, help="Vision-capable model id")
    parser.add_argument("--subset_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_vision_probe")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM server mode")
    parser.add_argument("--vllm_server_host", type=str, default=None, help="Required with --use_vllm")
    args = parser.parse_args()

    normal = run_probe(
        model=args.model,
        subset_size=args.subset_size,
        seed=args.seed,
        mode="normal",
        output_dir=f"{args.output_dir}_normal",
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
    )
    broken = run_probe(
        model=args.model,
        subset_size=args.subset_size,
        seed=args.seed,
        mode="broken",
        output_dir=f"{args.output_dir}_broken",
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
    )
    shuffled = run_probe(
        model=args.model,
        subset_size=args.subset_size,
        seed=args.seed,
        mode="shuffled",
        output_dir=f"{args.output_dir}_shuffled",
        use_vllm=args.use_vllm,
        vllm_server_host=args.vllm_server_host,
    )

    ab_cmp = compare_results(normal, shuffled)
    conf = verdict(normal, broken, ab_cmp)

    output = {
        "normal": asdict(normal),
        "broken": asdict(broken),
        "shuffled": asdict(shuffled),
        "ab_comparison": ab_cmp,
        "decision": conf,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
