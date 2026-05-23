#!/usr/bin/env python3
"""
ScienceWorld eval script for SFT/RL models using vLLM.
Evaluates on dev/test splits with per-task-type breakdown.

Input:  --checkpoint (HF model path), --temperature, --eval_split
Output: JSON file with per-task-type success rates and episode details.

Dependencies: scienceworld (+ Java runtime), vllm, transformers
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import scienceworld
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Must match the SFT training system prompt exactly.
SYSTEM_PROMPT = (
    "You're a helpful assistant. You are a science student performing experiments "
    "in a virtual science lab.\nComplete the given task by navigating rooms and "
    "interacting with objects.\n\n"
    "Your reasoning MUST be enclosed within <think> </think> tags.\n"
    "After the closing </think> tag, output EXACTLY ONE action command directly.\n"
    "The action must be from the admissible actions list.\n"
    "Do NOT wrap the action in any tags. Do NOT output anything after the action.\n\n"
    "Example: <think>I need to boil water. I should go to the kitchen first.</think>"
    "open door to kitchen\n"
)

MAX_VALID_ACTIONS = 50
MAX_PROMPT_TOKENS = 30000


class EpisodeTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise EpisodeTimeout("Episode exceeded time limit")


def _kill_java_gateways():
    subprocess.run(["pkill", "-9", "-f", "py4j.GatewayServer"],
                   capture_output=True, timeout=5)
    time.sleep(1)


def parse_action(response):
    """Parse action from model response. Returns (action, has_think, has_answer)."""
    has_think = bool(re.search(r"<think>.*?</think>", response, re.DOTALL))

    m = re.search(r"<think>.*?</think>\s*<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip(), has_think, True

    m = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    if m:
        action = m.group(1).strip().split("\n")[0].strip()
        for tok in ["<|im_end|>", "<|endoftext|>", "<answer>", "</answer>"]:
            action = action.replace(tok, "").strip()
        if action:
            return action, has_think, False

    return "", has_think, False


def format_user_turn(obs, task_desc, step_idx, max_steps, valid_actions, cumulative_score):
    """Format user turn to match SFT training data format exactly."""
    shown = valid_actions[:MAX_VALID_ACTIONS]
    actions_str = ", ".join(shown)

    if step_idx == 0:
        content = (
            f"\nTurn {step_idx + 1}:\n"
            f"Task: {task_desc}\n\n"
            f"State:\n{obs.strip()}\n\n"
        )
    else:
        content = (
            f"Reward:\n{cumulative_score}\n"
            f"\nTurn {step_idx + 1}:\n"
            f"State:\n{obs.strip()}\n\n"
        )

    content += (
        f"Admissible actions: [{actions_str}]\n"
        f"You have {max_steps - step_idx} actions left. "
        "Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> "
        "with no extra text. Strictly follow this format. "
        "Max response length: 512 words (tokens).\n"
    )
    return content


def truncate_messages(messages, tokenizer, max_tokens):
    """Keep system + first user + last N turns under token budget."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_count = len(tokenizer.encode(text))
    if token_count <= max_tokens:
        return messages

    preserved = messages[:2]
    rest = messages[2:]
    while rest and token_count > max_tokens:
        rest = rest[min(2, len(rest)):]
        trial = preserved + rest
        text = tokenizer.apply_chat_template(trial, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer.encode(text))

    return preserved + rest


def get_variations(env, task_name, eval_split):
    """Get variation indices for the given split."""
    env.load(task_name, 0)
    if eval_split == "dev":
        return env.get_variations_dev()
    else:
        return env.get_variations_test()


def run_episode(llm, sampling_params, tokenizer, env, task_name, var_idx, max_steps):
    """Run one evaluation episode. Returns result dict."""
    env.load(task_name, var_idx)
    obs, info = env.reset()
    task_desc = info.get("taskDesc", task_name)

    valid_actions = env.get_valid_action_object_combinations()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_user_turn(obs, task_desc, 0, max_steps, valid_actions, 0)},
    ]

    cumulative_score = 0
    num_steps = 0
    valid_count = 0
    format_count = 0

    for step_i in range(max_steps):
        msgs = truncate_messages(messages, tokenizer, MAX_PROMPT_TOKENS)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        clean = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        action, has_think, has_answer = parse_action(clean)

        action_is_valid = action in valid_actions

        num_steps += 1
        if action_is_valid:
            valid_count += 1
        if has_think and has_answer:
            format_count += 1

        obs, reward, done, step_info = env.step(action)
        cumulative_score = step_info.get("score", cumulative_score)
        valid_actions = env.get_valid_action_object_combinations()

        if done:
            break

        messages.append({"role": "assistant", "content": clean})
        messages.append({"role": "user", "content": format_user_turn(
            obs, task_desc, step_i + 1, max_steps, valid_actions, cumulative_score
        )})

    return {
        "task": task_name,
        "variation": var_idx,
        "success": cumulative_score >= 100,
        "final_score": cumulative_score,
        "num_steps": num_steps,
        "valid_action_rate": valid_count / max(num_steps, 1),
        "format_rate": format_count / max(num_steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on ScienceWorld")
    parser.add_argument("--checkpoint", required=True, help="Path to HF model checkpoint")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--eval_split", default="dev", choices=["dev", "test"])
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--task_names", default=None, help="Comma-separated task names (default: all)")
    parser.add_argument("--max_variations", type=int, default=-1, help="Max variations per task (-1=all)")
    parser.add_argument("--output", default="eval_scienceworld_results.json")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL results")
    parser.add_argument("--episode_timeout", type=int, default=300, help="Per-episode timeout in seconds")
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    results_jsonl = args.output.replace(".json", ".jsonl")

    print(f"Loading model from {args.checkpoint}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    llm = LLM(
        model=args.checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print("Initializing ScienceWorld...", flush=True)
    env = scienceworld.ScienceWorldEnv("")

    if args.task_names:
        task_names = [t.strip() for t in args.task_names.split(",")]
    else:
        task_names = env.get_task_names()

    eval_plan = []
    for task_name in task_names:
        variations = get_variations(env, task_name, args.eval_split)
        if args.max_variations > 0:
            variations = variations[:args.max_variations]
        for var_idx in variations:
            eval_plan.append((task_name, var_idx))

    print(f"Eval plan: {len(eval_plan)} episodes across {len(task_names)} task types "
          f"(split={args.eval_split}, T={args.temperature})", flush=True)

    # Resume: load completed episodes from JSONL
    completed = {}
    if args.resume and os.path.exists(results_jsonl):
        with open(results_jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (r["task"], r["variation"])
                completed[key] = r
        print(f"Resuming: {len(completed)} episodes already completed, "
              f"skipping to episode {len(completed)+1}", flush=True)

    results = []
    start_time = time.time()
    timeout_count = 0
    error_count = 0

    signal.signal(signal.SIGALRM, _timeout_handler)

    for i, (task_name, var_idx) in enumerate(eval_plan):
        if (task_name, var_idx) in completed:
            results.append(completed[(task_name, var_idx)])
            continue

        t0 = time.time()
        signal.alarm(args.episode_timeout)
        try:
            result = run_episode(llm, sampling_params, tokenizer, env, task_name, var_idx, args.max_steps)
        except EpisodeTimeout:
            elapsed = time.time() - t0
            result = {
                "task": task_name,
                "variation": var_idx,
                "success": False,
                "final_score": 0,
                "num_steps": 0,
                "valid_action_rate": 0,
                "format_rate": 0,
                "status": "timeout",
            }
            timeout_count += 1
            print(f"[{i+1}/{len(eval_plan)}] TIMEOUT {task_name} v{var_idx} ({elapsed:.1f}s) "
                  f"-- rebuilding env", flush=True)
            try:
                env.close()
            except Exception:
                pass
            _kill_java_gateways()
            env = scienceworld.ScienceWorldEnv("")
        except Exception as e:
            elapsed = time.time() - t0
            result = {
                "task": task_name,
                "variation": var_idx,
                "success": False,
                "final_score": 0,
                "num_steps": 0,
                "valid_action_rate": 0,
                "format_rate": 0,
                "status": f"error: {str(e)[:200]}",
            }
            error_count += 1
            print(f"[{i+1}/{len(eval_plan)}] ERROR {task_name} v{var_idx} ({elapsed:.1f}s): "
                  f"{str(e)[:100]}", flush=True)
            try:
                env.close()
            except Exception:
                pass
            _kill_java_gateways()
            env = scienceworld.ScienceWorldEnv("")
        else:
            elapsed = time.time() - t0
            status = "OK" if result["success"] else f"FAIL({result['final_score']})"
            print(
                f"[{i+1}/{len(eval_plan)}] {status} {task_name} v{var_idx} "
                f"({result['num_steps']} steps, valid={result['valid_action_rate']:.0%}, "
                f"fmt={result['format_rate']:.0%}, {elapsed:.1f}s)",
                flush=True,
            )
        finally:
            signal.alarm(0)

        results.append(result)

        with open(results_jsonl, 'a') as f:
            f.write(json.dumps(result) + '\n')

    total_time = time.time() - start_time
    try:
        env.close()
    except Exception:
        pass

    # Per-task-type aggregation
    type_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_score": 0,
                                       "total_steps": 0, "total_valid": 0, "total_format": 0})
    for r in results:
        s = type_stats[r["task"]]
        s["count"] += 1
        s["success"] += int(r["success"])
        s["total_score"] += r["final_score"]
        s["total_steps"] += r["num_steps"]
        s["total_valid"] += r["valid_action_rate"]
        s["total_format"] += r["format_rate"]

    print("\n" + "=" * 90, flush=True)
    print(f"ScienceWorld Eval Results (split={args.eval_split}, T={args.temperature}, "
          f"n={len(results)}, timeouts={timeout_count}, errors={error_count})", flush=True)
    print("=" * 90, flush=True)
    print(f"{'Task':<50} {'N':>4} {'Succ':>5} {'Rate':>7} {'AvgScore':>8} {'AvgStep':>7}", flush=True)
    print("-" * 90, flush=True)

    total_n = 0
    total_succ = 0
    total_score = 0
    for task_name in task_names:
        if task_name not in type_stats:
            continue
        s = type_stats[task_name]
        rate = s["success"] / s["count"]
        avg_score = s["total_score"] / s["count"]
        avg_steps = s["total_steps"] / s["count"]
        print(f"{task_name:<50} {s['count']:>4} {s['success']:>5} {rate:>6.1%} "
              f"{avg_score:>8.1f} {avg_steps:>7.1f}", flush=True)
        total_n += s["count"]
        total_succ += s["success"]
        total_score += s["total_score"]

    overall_rate = total_succ / max(total_n, 1)
    overall_score = total_score / max(total_n, 1)
    print("-" * 90, flush=True)
    print(f"{'TOTAL':<50} {total_n:>4} {total_succ:>5} {overall_rate:>6.1%} "
          f"{overall_score:>8.1f}", flush=True)
    print(f"\nTotal time: {total_time:.0f}s ({total_time/max(total_n,1):.1f}s/episode)", flush=True)

    # Save JSON
    output = {
        "checkpoint": args.checkpoint,
        "eval_split": args.eval_split,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "total_episodes": total_n,
        "total_success": total_succ,
        "overall_success_rate": round(overall_rate, 4),
        "overall_avg_score": round(overall_score, 2),
        "total_time_seconds": round(total_time, 1),
        "timeouts": timeout_count,
        "errors": error_count,
        "per_task_type": {},
        "episodes": results,
    }
    for task_name in task_names:
        if task_name not in type_stats:
            continue
        s = type_stats[task_name]
        output["per_task_type"][task_name] = {
            "count": s["count"],
            "success": s["success"],
            "success_rate": round(s["success"] / s["count"], 4),
            "avg_score": round(s["total_score"] / s["count"], 2),
            "avg_steps": round(s["total_steps"] / s["count"], 1),
            "avg_valid_rate": round(s["total_valid"] / s["count"], 4),
            "avg_format_rate": round(s["total_format"] / s["count"], 4),
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
