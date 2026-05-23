#!/usr/bin/env python3
"""Eval exp021 checkpoint using eval_sft_fast logic."""
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import json, os, random, re, sys, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAGEN"))
from ragen.env.alfworld.env import AlfredTXTEnv
from ragen.env.alfworld.config import AlfredEnvConfig
from vllm import LLM, SamplingParams

SYSTEM_PROMPT = (
    "You're a helpful assistant. You are an expert agent in the ALFRED Embodied Environment.\n"
    "Complete household tasks by navigating and interacting with objects.\n\n"
    "Before each action, you MUST first produce a self-guidance assessment:\n"
    "[Assessment: positive/neutral/negative] - evaluate your current progress.\n"
    "[Reasoning: one sentence analyzing what has been accomplished and what remains.]\n"
    "[Suggestion: the best next action from admissible actions.]\n\n"
    "Your reasoning with self-guidance MUST be enclosed within <think> </think> tags.\n"
    "Then choose an admissible action and present it within <answer>...</answer> tags."
)

TASK_TYPE_MAP = {
    "pick_and_place_simple": "put",
    "pick_two_obj_and_place": "put",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_clean_then_place_in_recep": "clean",
    "look_at_obj_in_light": "examine",
}

MAX_PROMPT_TOKENS = 30000

def parse_action(response):
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

def truncate_messages(messages, tokenizer, max_tokens):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_count = len(tokenizer.encode(text))
    if token_count <= max_tokens:
        return messages
    preserved = messages[:2]
    rest = messages[2:]
    while rest and token_count > max_tokens:
        removed = min(2, len(rest))
        rest = rest[removed:]
        trial = preserved + rest
        text = tokenizer.apply_chat_template(trial, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer.encode(text))
    return preserved + rest

def run_episode(llm, sampling_params, tokenizer, env, episode_idx, max_steps=30):
    random.seed(episode_idx)
    np.random.seed(episode_idx)
    obs = env.reset(seed=episode_idx, mode="val")
    game_file = getattr(env, "current_game_file", "")
    task_type = get_task_type(game_file)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs},
    ]
    steps = []
    for step_i in range(max_steps):
        msgs = truncate_messages(messages, tokenizer, MAX_PROMPT_TOKENS)
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([text], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        clean_response = response.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        action, has_think, has_answer = parse_action(clean_response)
        format_ok = has_think and (has_answer or bool(action))
        action_is_valid = action in env.available_actions if action else False
        step_info = {"step": step_i, "action": action, "has_think": has_think,
                     "has_answer": has_answer, "format_ok": format_ok,
                     "action_is_valid": action_is_valid}
        if not action:
            step_info["success"] = False
            step_info["done"] = True
            steps.append(step_info)
            break
        obs, reward, done, info = env.step(action)
        step_info["reward"] = reward
        step_info["success"] = info.get("success", False)
        step_info["done"] = done
        steps.append(step_info)
        if done:
            break
        messages.append({"role": "assistant", "content": clean_response})
        messages.append({"role": "user", "content": obs})
    return steps, task_type

def main():
    model_path = "./checkpoints/exp021_ragen_sft_v4/final"
    num_episodes = 20
    max_steps = 30
    temperature = 0.4
    output_path = "./artifacts/eval_exp021_sft_fast.json"

    start_time = time.time()
    print(f"Loading model from {model_path} with vLLM...", flush=True)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              max_model_len=32768, trust_remote_code=True, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=temperature, top_p=0.95, max_tokens=256,
                                     stop=["<|im_end|>", "<|endoftext|>"])
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s", flush=True)

    os.environ["ALFWORLD_DATA"] = os.path.expanduser("~/.cache/alfworld")
    ragen_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "RAGEN")
    alfworld_cfg = os.path.join(ragen_root, "ragen", "env", "alfworld", "alfworld_config.yaml")
    env_config = AlfredEnvConfig(config_file=alfworld_cfg, eval_dataset="eval_in_distribution")
    env = AlfredTXTEnv(config=env_config, mode="val")
    print(f"ALFWorld env initialized, {env.num_games} games available", flush=True)

    ep_results = []
    eval_start = time.time()
    for ep_i in range(num_episodes):
        ep_start = time.time()
        steps, task_type = run_episode(llm, sampling_params, tokenizer, env, episode_idx=ep_i, max_steps=max_steps)
        ep_time = time.time() - ep_start
        total_steps = len(steps)
        valid_actions = sum(1 for s in steps if s["action_is_valid"])
        format_ok_count = sum(1 for s in steps if s["format_ok"])
        success = any(s.get("success", False) for s in steps)
        result = {"episode": ep_i, "task_type": task_type, "total_steps": total_steps,
                  "valid_actions": valid_actions, "format_ok": format_ok_count,
                  "success": success, "time": round(ep_time, 1)}
        ep_results.append(result)
        status = "SUCCESS" if success else "FAIL"
        print(f"  ep {ep_i:>2}: {status} | steps={total_steps} valid={valid_actions}/{total_steps} fmt={format_ok_count}/{total_steps} type={task_type} ({ep_time:.1f}s)", flush=True)

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time
    n = len(ep_results)
    success_rate = sum(1 for r in ep_results if r["success"]) / n
    avg_valid = np.mean([r["valid_actions"]/max(r["total_steps"],1) for r in ep_results])
    avg_format = np.mean([r["format_ok"]/max(r["total_steps"],1) for r in ep_results])
    avg_steps = np.mean([r["total_steps"] for r in ep_results])

    print()
    print("=" * 60)
    print(f"RESULTS ({n} episodes)")
    print("=" * 60)
    print(f"  success_rate        : {success_rate:.1%} ({sum(1 for r in ep_results if r['success'])}/{n})")
    print(f"  action_is_valid rate: {avg_valid:.1%}")
    print(f"  format_ok rate      : {avg_format:.1%}")
    print(f"  avg_steps/episode   : {avg_steps:.1f}")
    print(f"  eval time           : {eval_time:.1f}s")
    print("=" * 60)

    summary = {"model_path": model_path, "num_episodes": n, "temperature": temperature,
               "max_steps": max_steps, "success_rate": round(success_rate, 4),
               "action_is_valid_rate": round(avg_valid, 4), "format_ok_rate": round(avg_format, 4),
               "avg_steps": round(avg_steps, 1), "eval_time_seconds": round(eval_time, 1),
               "total_time_seconds": round(total_time, 1), "episodes": ep_results}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {output_path}")
    env.close()

if __name__ == "__main__":
    main()
