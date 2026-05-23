#!/usr/bin/env python3
"""Standalone eval script: load any GRPO/SFT checkpoint (HF or FSDP format) for Qwen3-1.7B
and run C003-compliant 16-episode ALFWorld validation via vLLM.
Input: checkpoint path. Output: JSON results file with per-episode metrics + summary.

Format alignment: messages match RAGEN ctx_manager.py + AlfWorldMemory format exactly.
"""

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAGEN"))
from ragen.env.alfworld.env import AlfredTXTEnv
from ragen.env.alfworld.config import AlfredEnvConfig

from vllm import LLM, SamplingParams

# --- RAGEN-aligned env_instruction (from config/envs.yaml AlfworldSG) ---
ENV_INSTRUCTION = (
    "You are an expert agent in the ALFRED Embodied Environment.\n"
    "Complete household tasks by navigating and interacting with objects.\n\n"
    "Before each action, you MUST first produce a self-guidance assessment in your reasoning:\n"
    "[Assessment: positive/neutral/negative] - evaluate your current progress toward the task goal.\n"
    "[Reasoning: one sentence analyzing what has been accomplished and what remains.]\n"
    "[Suggestion: the best next action from admissible actions.]\n\n"
    "Your reasoning MUST be enclosed within <think> </think> tags.\n"
    "After the closing </think> tag, output EXACTLY ONE admissible action command directly.\n"
    "The action must be a precise ALFWorld command from the admissible actions list "
    "(e.g. go to desk 1, pick up pen 2, open drawer 1).\n"
    "Do NOT wrap the action in any tags. Do NOT output natural language descriptions "
    "or anything after the action.\n\n"
    "Example: <think>[Assessment: neutral] I need to find a pen. The desk is a likely location. "
    "[Suggestion: go to desk 1]</think>go to desk 1\n"
)

# ctx_manager._build_system_content: "You're a helpful assistant. " + env_instruction
SYSTEM_PROMPT = "You're a helpful assistant. " + ENV_INSTRUCTION

LENGTH_PROMPT = "Max response length: 512 words (tokens)."
HISTORY_LENGTH = 50

TASK_TYPE_MAP = {
    "pick_and_place_simple": "put",
    "pick_two_obj_and_place": "put",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_clean_then_place_in_recep": "clean",
    "look_at_obj_in_light": "examine",
}

MAX_PROMPT_TOKENS = 30000

# --- Observation cleaning (mirrors AlfWorldMemory) ---
CLEAN_PATTERNS = [
    r"-= Welcome to TextWorld, ALFRED! =-",
    r"You have \d+ actions left\.?",
    r"Always output:.*?(?=\n|$)",
    r"Strictly follow this format\..*?(?=\n|$)",
    r"Max response length:.*?(?=\n|$)",
]


def clean_observation(obs):
    cleaned = obs
    for pattern in CLEAN_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.MULTILINE | re.DOTALL)
    cleaned = re.sub(r"Admissible actions:.*?(?=\n|$)", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned).strip()
    return cleaned


def extract_task(state):
    match = re.search(r"Your task is to: (.+?)(?:\.\n|\n|$)", state, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_admissible_actions(state):
    match = re.search(r"Admissible actions: \[(.+?)\]", state, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_action_from_response(llm_response):
    match = re.search(r'<answer>(.*?)</answer>', llm_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'</think>\s*(.*)', llm_response, re.DOTALL)
    if match:
        action = match.group(1).strip().split("\n")[0].strip()
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            action = action.replace(tok, "").strip()
        return action
    return llm_response.strip()


def build_user_content(obs, history, step_idx, max_steps, task_desc=None):
    """Build user content matching AlfWorldMemory.build_user_content format."""
    content_parts = []

    if task_desc:
        content_parts.append(f"Your task is to: {task_desc}.")

    if step_idx > 0:
        content_parts.append(
            f"Prior to this step, you have already taken {step_idx} step(s)."
        )

    history_start = max(0, len(history) - HISTORY_LENGTH)
    visible_history = history[history_start:]
    if visible_history:
        content_parts.append(
            f"Below are the most recent {len(visible_history)} observations and "
            "the corresponding actions you took:"
        )
        for h in visible_history:
            h_obs = clean_observation(h["obs"])
            if len(h_obs) > 200:
                h_obs = h_obs[:197] + "..."
            h_action = extract_action_from_response(h["response"])
            content_parts.append(
                f"[Observation {h['step']}: '{h_obs}', Action {h['step']}: '{h_action}']"
            )

    current_step = step_idx + 1
    clean_current = clean_observation(obs)
    content_parts.append(
        f"You are now at step {current_step} and your current observation is: {clean_current}"
    )

    admissible = extract_admissible_actions(obs)
    if admissible:
        content_parts.append(
            f"Your admissible actions of the current situation are: [{admissible}]"
        )

    actions_left = max_steps - step_idx
    content_parts.append(
        f"Now it's your turn to take an action. "
        f"You have {actions_left} actions left. "
        f"Output your reasoning in <think>...</think> tags, then output exactly one admissible action command directly after </think>. "
        f"Do not use <answer> tags. The action must be a precise ALFWorld command. {LENGTH_PROMPT}"
    )

    return "\n".join(content_parts)


def parse_action(response: str):
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))
    m = re.search(r"<think>.*?</think>\s*<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip(), has_think, True
    m = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if m:
        action = m.group(1).strip().split("\n")[0].strip()
        for tok in ["<|im_end|>", "<|endoftext|>"]:
            action = action.replace(tok, "").strip()
        if action:
            return action, has_think, False
    return "", has_think, False


def get_task_type(game_file):
    if not game_file:
        return "unknown"
    for prefix, ttype in TASK_TYPE_MAP.items():
        if prefix in game_file:
            return ttype
    return "unknown"


def truncate_user_content(messages, tokenizer, max_tokens):
    """Truncate by trimming history entries from the user content if too long."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_count = len(tokenizer.encode(text))
    if token_count <= max_tokens:
        return messages
    # Already single-turn, just return as-is (context window bounded by HISTORY_LENGTH)
    return messages


def run_episode(llm, sampling_params, tokenizer, env, episode_idx, max_steps=50):
    random.seed(episode_idx)
    np.random.seed(episode_idx)

    obs = env.reset(seed=episode_idx, mode="val")
    game_file = getattr(env, "current_game_file", "")
    task_type = get_task_type(game_file)
    task_desc = extract_task(obs)

    history = []
    steps = []
    for step_i in range(max_steps):
        user_content = build_user_content(obs, history, step_i, max_steps, task_desc)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        msgs = truncate_user_content(messages, tokenizer, MAX_PROMPT_TOKENS)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        clean_response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        action, has_think, has_answer = parse_action(clean_response)
        format_ok = has_think and (has_answer or bool(action))
        action_is_valid = action in env.available_actions if action else False

        step_info = {
            "step": step_i,
            "action": action,
            "has_think": has_think,
            "has_answer": has_answer,
            "format_ok": format_ok,
            "action_is_valid": action_is_valid,
            "action_is_effective": False,
        }

        if not action:
            step_info["success"] = False
            step_info["done"] = True
            step_info["reward"] = 0.0
            steps.append(step_info)
            break

        obs_before = obs
        obs, reward, done, info = env.step(action)
        step_info["reward"] = reward
        step_info["success"] = info.get("success", False)
        step_info["done"] = done
        step_info["action_is_effective"] = info.get("action_is_effective", False)
        steps.append(step_info)

        if done:
            break

        history.append({
            "step": step_i + 1,
            "obs": obs_before,
            "response": clean_response,
        })

    return steps, task_type


def detect_checkpoint_format(path):
    if os.path.isfile(os.path.join(path, "config.json")):
        if (os.path.isfile(os.path.join(path, "model.safetensors"))
                or any(f.startswith("model-") and f.endswith(".safetensors") for f in os.listdir(path))
                or os.path.isfile(os.path.join(path, "pytorch_model.bin"))):
            return "hf"
    actor_dir = os.path.join(path, "actor")
    if os.path.isdir(actor_dir) and os.path.isfile(os.path.join(actor_dir, "fsdp_config.json")):
        return "fsdp"
    if os.path.isfile(os.path.join(path, "fsdp_config.json")):
        return "fsdp"
    raise ValueError(
        f"Cannot detect checkpoint format at {path}. "
        f"Expected HF (config.json + model.safetensors) or FSDP (actor/fsdp_config.json)."
    )


def convert_fsdp_to_hf(fsdp_dir, output_dir, dtype_str="bf16"):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    target_dtype = dtype_map[dtype_str]

    actor_dir = fsdp_dir
    if os.path.isdir(os.path.join(fsdp_dir, "actor")):
        actor_dir = os.path.join(fsdp_dir, "actor")

    fsdp_config_path = os.path.join(actor_dir, "fsdp_config.json")
    with open(fsdp_config_path) as f:
        fsdp_config = json.load(f)
    world_size = fsdp_config["world_size"]

    hf_dir = os.path.join(actor_dir, "huggingface")
    if not os.path.isdir(hf_dir):
        raise FileNotFoundError(f"huggingface/ subdirectory not found in {actor_dir}")

    print(f"  Converting FSDP checkpoint (world_size={world_size}) -> HF ...", flush=True)
    shards = []
    for rank in range(world_size):
        p = os.path.join(actor_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        shards.append(torch.load(p, map_location="cpu", weights_only=False))

    full_state_dict = {}
    for key in list(shards[0].keys()):
        local_tensors = []
        shard_dim = 0
        for rank in range(world_size):
            dt = shards[rank][key]
            if hasattr(dt, "placements") and len(dt.placements) > 0:
                shard_dim = dt.placements[0].dim
            lt = dt._local_tensor if hasattr(dt, "_local_tensor") else dt.to_local()
            local_tensors.append(lt)
        full_state_dict[key] = torch.cat(local_tensors, dim=shard_dim).to(target_dtype)
    del shards

    config = AutoConfig.from_pretrained(hf_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=target_dtype)
    model.to_empty(device="cpu")

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, state_dict=full_state_dict, max_shard_size="5GB")
    del full_state_dict, model

    for fname in os.listdir(hf_dir):
        src = os.path.join(hf_dir, fname)
        dst = os.path.join(output_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    print(f"  Converted to {output_dir}", flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Standalone ALFWorld eval: load checkpoint and run C003-compliant validation."
    )
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to checkpoint (HF dir or FSDP global_step dir)")
    parser.add_argument("--num_episodes", type=int, default=16,
                        help="Number of eval episodes (C003: must be >=16, default 16)")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Max steps per episode (default 50)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature (default 0.4)")
    parser.add_argument("--output_dir", default="/data/eval_results/",
                        help="Directory for JSON result files")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default 0)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35,
                        help="vLLM GPU memory utilization (default 0.35, safe for coexisting with training)")
    parser.add_argument("--fsdp_convert_dir", default=None,
                        help="Where to save converted HF model if checkpoint is FSDP "
                             "(default: <checkpoint_path>_hf)")
    parser.add_argument("--eval_dataset", default="eval_in_distribution",
                        choices=["eval_in_distribution", "eval_out_of_distribution"],
                        help="Dataset split: eval_in_distribution (16 games) or eval_out_of_distribution (134 OOD tasks)")
    args = parser.parse_args()

    if args.num_episodes < 16:
        print(f"ERROR: C003 requires >=16 episodes, got {args.num_episodes}. Aborting.", flush=True)
        sys.exit(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    start_time = time.time()
    ckpt_path = os.path.abspath(args.checkpoint_path)

    fmt = detect_checkpoint_format(ckpt_path)
    print(f"Checkpoint: {ckpt_path} (format: {fmt})", flush=True)

    if fmt == "fsdp":
        convert_dir = args.fsdp_convert_dir or (ckpt_path.rstrip("/") + "_hf")
        if os.path.isfile(os.path.join(convert_dir, "config.json")):
            print(f"  Using existing HF conversion at {convert_dir}", flush=True)
            model_path = convert_dir
        else:
            model_path = convert_fsdp_to_hf(ckpt_path, convert_dir)
    else:
        model_path = ckpt_path

    print(f"Loading model from {model_path} ...", flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        enforce_eager=True,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    os.environ.setdefault("ALFWORLD_DATA", "/data/alfworld")
    ragen_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAGEN")
    alfworld_cfg = os.path.join(ragen_root, "ragen", "env", "alfworld", "alfworld_config.yaml")
    env_config = AlfredEnvConfig(config_file=alfworld_cfg, eval_dataset=args.eval_dataset)
    env = AlfredTXTEnv(config=env_config, mode="val")
    print(f"Environment initialized ({env.num_games} games available)", flush=True)

    ep_results = []
    eval_start = time.time()

    for ep_i in range(1, args.num_episodes + 1):
        ep_start = time.time()
        steps, task_type = run_episode(
            llm, sampling_params, tokenizer, env,
            episode_idx=ep_i - 1,
            max_steps=args.max_steps,
        )
        ep_time = time.time() - ep_start

        total_steps = len(steps)
        valid_actions = sum(1 for s in steps if s["action_is_valid"])
        effective_actions = sum(1 for s in steps if s["action_is_effective"])
        format_ok_count = sum(1 for s in steps if s["format_ok"])
        success = any(s.get("success", False) for s in steps)

        result = {
            "episode": ep_i,
            "task_type": task_type,
            "total_steps": total_steps,
            "success": success,
            "action_is_valid_rate": valid_actions / total_steps if total_steps > 0 else 0,
            "action_is_effective_rate": effective_actions / total_steps if total_steps > 0 else 0,
            "format_ok_rate": format_ok_count / total_steps if total_steps > 0 else 0,
            "time_seconds": round(ep_time, 1),
            "steps": steps,
        }
        ep_results.append(result)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  EP {ep_i:>2}/{args.num_episodes}: {status} | "
            f"steps={total_steps:2d} "
            f"valid={valid_actions}/{total_steps} "
            f"effective={effective_actions}/{total_steps} "
            f"fmt={format_ok_count}/{total_steps} "
            f"type={task_type} "
            f"({ep_time:.1f}s)",
            flush=True,
        )

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    n = len(ep_results)
    success_rate = sum(1 for r in ep_results if r["success"]) / n
    avg_valid = np.mean([r["action_is_valid_rate"] for r in ep_results])
    avg_effective = np.mean([r["action_is_effective_rate"] for r in ep_results])
    avg_format = np.mean([r["format_ok_rate"] for r in ep_results])
    avg_steps = np.mean([r["total_steps"] for r in ep_results])

    print("\n" + "=" * 60, flush=True)
    print(f"RESULTS ({n} episodes, C003-compliant)", flush=True)
    print("=" * 60, flush=True)
    print(f"  success_rate           : {success_rate:.1%} ({sum(1 for r in ep_results if r['success'])}/{n})", flush=True)
    print(f"  action_is_valid rate   : {avg_valid:.1%}", flush=True)
    print(f"  action_is_effective rate: {avg_effective:.1%}", flush=True)
    print(f"  format_ok rate         : {avg_format:.1%}", flush=True)
    print(f"  avg_steps/episode      : {avg_steps:.1f}", flush=True)
    print(f"  eval time              : {eval_time:.1f}s ({eval_time/n:.1f}s/episode)", flush=True)
    print(f"  total time             : {total_time:.1f}s", flush=True)
    print("=" * 60, flush=True)

    type_stats = defaultdict(lambda: {"count": 0, "success": 0})
    for r in ep_results:
        tt = r.get("task_type", "unknown")
        type_stats[tt]["count"] += 1
        if r["success"]:
            type_stats[tt]["success"] += 1
    print("\nPer-task-type breakdown:", flush=True)
    print(f"  {'Type':<12} {'Count':>5} {'Success':>7} {'Rate':>8}", flush=True)
    print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*8}", flush=True)
    for tt in sorted(type_stats.keys()):
        s = type_stats[tt]
        rate = s["success"] / s["count"] if s["count"] > 0 else 0
        print(f"  {tt:<12} {s['count']:>5} {s['success']:>7} {rate:>7.1%}", flush=True)
    print(flush=True)

    ckpt_name = os.path.basename(ckpt_path.rstrip("/"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"eval_{ckpt_name}_{args.num_episodes}ep_{timestamp}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)

    summary = {
        "checkpoint_path": ckpt_path,
        "model_path": model_path,
        "checkpoint_format": fmt,
        "num_episodes": n,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "gpu": args.gpu,
        "c003_compliant": n >= 16,
        "eval_dataset": args.eval_dataset,
        "success_rate": round(success_rate, 4),
        "action_is_valid_rate": round(float(avg_valid), 4),
        "action_is_effective_rate": round(float(avg_effective), 4),
        "format_ok_rate": round(float(avg_format), 4),
        "avg_steps": round(float(avg_steps), 1),
        "eval_time_seconds": round(eval_time, 1),
        "total_time_seconds": round(total_time, 1),
        "per_task_type": {tt: dict(s) for tt, s in type_stats.items()},
        "episodes": ep_results,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
