import argparse
import os
from typing import List, Any, Dict
import torch.distributed as dist

import wandb
from accelerate import PartialState
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

from sparc.prompt import generate_prompt
from sparc.validation import extract_solution_path, validate_solution, analyze_path


def build_sparc_reward_functions(original_examples: List[Dict[str, Any]]):
    """Builds a list of reward functions, one per reward component.

    Returns a list[callable], each taking (completions, prompts, **kwargs) -> list[float].
    """
    prompt_to_puzzle: Dict[str, Dict[str, Any]] = {}
    for example in original_examples:
        prompt_text = generate_prompt(example).strip()
        prompt_to_puzzle[prompt_text] = example

    is_main = PartialState().is_main_process

    def _normalize_texts(completion, prompt):
        if isinstance(prompt, list) and len(prompt) > 0 and isinstance(prompt[0], dict):
            # Prefer the first user message content if present; fallback to first message content
            try:
                prompt_text_local = next(
                    (m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"),
                    prompt[0].get("content", str(prompt)),
                )
            except Exception:
                prompt_text_local = prompt[0].get("content", str(prompt))
        else:
            prompt_text_local = str(prompt)

        if isinstance(completion, list) and len(completion) > 0 and isinstance(completion[0], dict):
            completion_text_local = completion[0].get("content", str(completion))
        else:
            completion_text_local = str(completion)

        return prompt_text_local, completion_text_local

    # 1) Perfect solution reward (1.0 if fully correct, else 0.0)
    def reward_perfect_solution(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            try:
                extracted_path = extract_solution_path(completion_text, puzzle)
                if extracted_path is not None and validate_solution(extracted_path, puzzle):
                    rewards_local.append(1.0)
                else:
                    rewards_local.append(0.0)
            except Exception:
                rewards_local.append(0.0)
        if is_main and len(rewards_local) > 0 and wandb.run is not None:
            wandb.log({"rewards/perfect_mean": sum(rewards_local) / len(rewards_local)})
        return rewards_local

    # Helper to compute analysis dict safely
    def _extract_path(completion_text: str, puzzle: Dict[str, Any]):
        try:
            return extract_solution_path(completion_text, puzzle)
        except Exception:
            return None

    def _is_perfect(completion_text: str, puzzle: Dict[str, Any]) -> bool:
        try:
            extracted_path = extract_solution_path(completion_text, puzzle)
            return bool(extracted_path is not None and validate_solution(extracted_path, puzzle))
        except Exception:
            return False

    def _safe_analysis(completion_text: str, puzzle: Dict[str, Any]):
        try:
            extracted_path = extract_solution_path(completion_text, puzzle)
            if extracted_path is None:
                return None
            return analyze_path(extracted_path, puzzle)
        except Exception:
            return None

    # 2) Starts at start and ends at exit (0.25)
    def reward_starts_and_ends(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("starts_at_start_ends_at_exit", False) else 0.0)
        return rewards_local

    # 3) Connected line (0.25)
    def reward_connected_line(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("connected_line", False) else 0.0)
        return rewards_local

    # 4) Non-intersecting line (0.25)
    def reward_non_intersecting(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("non_intersecting_line", False) else 0.0)
        return rewards_local

    # 5) No rule crossing (0.25)
    def reward_no_rule_crossing(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is None:
                rewards_local.append(0.0)
                continue
            if _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
            else:
                analysis = _safe_analysis(completion_text, puzzle)
                rewards_local.append(1.0 if analysis and analysis.get("no_rule_crossing", False) else 0.0)
        return rewards_local

    # 6) Format hint reward (small reward for emitting expected format when path not valid)
    def reward_format_hint(completions, prompts, **kwargs):
        rewards_local: List[float] = []
        for completion, prompt in zip(completions, prompts):
            prompt_text, completion_text = _normalize_texts(completion, prompt)
            puzzle = prompt_to_puzzle.get(prompt_text)
            if puzzle is not None and _is_perfect(completion_text, puzzle):
                rewards_local.append(0.0)
                continue
            has_format = ("####" in completion_text and "(" in completion_text and ")" in completion_text)
            rewards_local.append(1.0 if has_format else 0.0)
        return rewards_local

    # Return all reward functions as a list
    return [
        reward_perfect_solution,
        reward_starts_and_ends,
        reward_connected_line,
        reward_non_intersecting,
        reward_no_rule_crossing,
        reward_format_hint,
    ]


def to_grpo_prompt_format(dataset: Dataset, tokenizer: AutoTokenizer, max_prompt_length: int = 5000) -> Dataset:
    def _map_fn(ex):
        prompt = generate_prompt(ex).strip()
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        system_msg = (
            "Use <think>...</think> to reason privately. Keep thinking concise. "
            "After thinking, output ONLY the final SPaRC path in the exact format '####(x0,y0)->(x1,y1)->...'. "
            "Do not include any extra text outside the final path line."
        )
        return {
            "prompt": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            "puzzle_data": ex,
        }

    return dataset.map(_map_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--vllm_server_host", type=str, required=True, help="Hostname or IP of the vLLM server (port 8000)")
    parser.add_argument("--wandb_project", type=str, default="sparc-grpo")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Optional W&B run id to resume")
    parser.add_argument("--run_name_addition", type=str, default=os.environ.get("RUN_NAME_ADDITION", ""), help="Optional suffix to append to run/model name")
    args = parser.parse_args()

    state = PartialState()
    is_main = state.is_main_process

    # Compute unified run name: <base>-SPaRC-GRPO[-addition]
    _base = args.model.split('/')[-1]
    _addition = (args.run_name_addition or "").strip()
    _suffix = f"-{_addition}" if _addition else ""
    run_name = f"{_base}-SPaRC-GRPO{_suffix}"

    if is_main:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=args.wandb_run_id, name=run_name, resume="allow" if args.wandb_run_id else None, config={
            "model": args.model,
            "dataset": "lkaesberg/SPaRC",
            "trainer": "GRPO",
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": args.vllm_server_host},
            settings=wandb.Settings(init_timeout=3600)
        )

    # Load datasets
    train_raw = load_dataset("lkaesberg/SPaRC", "all", split="train")
    eval_raw = load_dataset("lkaesberg/SPaRC", "all", split="test[:100]")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Transform to prompt format
    train_ds = to_grpo_prompt_format(train_raw, tokenizer)
    eval_ds = to_grpo_prompt_format(eval_raw, tokenizer)

    # Build reward functions with access to original examples
    combined_examples: List[Dict[str, Any]] = list(train_raw) + list(eval_raw)
    rewards = build_sparc_reward_functions(combined_examples)

    # Minimal GRPO config per docs; vLLM in server mode
    config = GRPOConfig(
        output_dir=f"./checkpoints/grpo_outputs_{args.model.replace('/', '_')}",
        report_to="wandb",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        do_eval=False,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=args.vllm_server_host,
        vllm_server_timeout=3600,
        gradient_checkpointing=True,
        max_completion_length=10000, 
        num_generations=4,
        num_train_epochs=4,
        # Built-in weighting of multiple reward functions
        reward_weights=[1.0, 0.1, 0.1, 0.1, 0.1, 0.01],
        scale_rewards=False,
        loss_type="dr_grpo"
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=rewards,
        train_dataset=train_ds,
        #eval_dataset=eval_ds,
    )

    # Resume only if a run id was provided
    trainer.train(resume_from_checkpoint=bool(args.wandb_run_id))

    trainer.save_model(f"./models/{run_name}")

    if is_main:
        wandb.finish()


if __name__ == "__main__":
    main()

