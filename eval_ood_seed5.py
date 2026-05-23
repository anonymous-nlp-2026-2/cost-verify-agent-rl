"""
Standalone eval: exp029 seed1 checkpoint on 134 OOD tasks
Per-task-type breakdown for paper analysis
"""
import os
import json
import re
import time
from collections import defaultdict

os.environ["ALFWORLD_DATA"] = "/data/alfworld"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TMPDIR"] = "/data/tmp"

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

CHECKPOINT = "./checkpoints/qwen_seed5_step10_hf"
MAX_TURNS = 50
TEMPERATURE = 0.4

SYSTEM_PROMPT = """You're a helpful assistant. You are an expert agent in the ALFRED Embodied Environment.
Complete household tasks by navigating and interacting with objects.

Before each action, you MUST first produce a self-guidance assessment in your reasoning:
[Assessment: positive/neutral/negative] - evaluate your current progress toward the task goal.
[Reasoning: one sentence analyzing what has been accomplished and what remains.]
[Suggestion: the best next action from admissible actions.]

Your reasoning MUST be enclosed within <think> </think> tags.
After the closing </think> tag, wrap your chosen action in <answer> </answer> tags.
The action must be a precise ALFWorld command from the admissible actions list (e.g. go to desk 1, pick up pen 2, open drawer 1).
Do NOT output natural language descriptions or anything after the </answer> tag.

Example: <think>[Assessment: neutral] I need to find a pen. The desk is a likely location. [Suggestion: go to desk 1]</think><answer>go to desk 1</answer>"""

TASK_TYPES = [
    "pick_and_place_simple",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "look_at_obj_in_light",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
]

def get_task_type(game_file_path):
    parts = game_file_path.split("/")
    for part in parts:
        for tt in TASK_TYPES:
            if part.startswith(tt):
                return tt
    return "unknown"

def extract_action(response):
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip().split("\n")[-1].strip()

def get_ood_game_files():
    import alfworld.agents.environment.alfred_tw_env as ate
    from ragen.env.alfworld.utils import load_config
    config_path = "./RAGEN/ragen/env/alfworld/alfworld_config.yaml"
    config = load_config(config_path)
    raw_env = ate.AlfredTWEnv(config=config, train_eval="eval_out_of_distribution")
    game_files = list(raw_env.game_files)
    return game_files

def run_episode(llm, tokenizer, sampling_params, game_file, episode_idx):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    from ragen.env.alfworld.utils import load_config

    config_path = "./RAGEN/ragen/env/alfworld/alfworld_config.yaml"
    config = load_config(config_path)
    max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    wrappers = [AlfredDemangler(), AlfredInfos()]
    env_id = textworld.gym.register_game(
        game_file,
        request_infos=request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=max_steps,
        wrappers=wrappers,
    )
    env = textworld.gym.make(env_id)
    obs, info = env.reset()

    observation = obs[0]
    admissible = info["admissible_commands"][0]
    obs_text = f"{observation}\n\nAdmissible actions: [{', '.join(admissible)}]"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_text},
    ]

    won = False
    num_actions = 0
    valid_actions = 0

    for turn in range(MAX_TURNS):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        action = extract_action(response)
        action_is_valid = action in admissible

        obs_list, _, dones, infos = env.step([action])
        observation = obs_list[0]
        admissible = infos["admissible_commands"][0]
        done = dones[0]
        episode_won = infos["won"][0]
        num_actions += 1
        if action_is_valid:
            valid_actions += 1

        if done or episode_won:
            won = episode_won
            break

        obs_text = f"{observation}\n\nAdmissible actions: [{', '.join(admissible)}]"
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": obs_text})

    env.close()
    return {
        "episode_idx": episode_idx,
        "game_file": game_file,
        "task_type": get_task_type(game_file),
        "success": bool(won),
        "num_actions": num_actions,
        "valid_actions": valid_actions,
        "valid_action_rate": valid_actions / max(num_actions, 1),
    }

def main():
    import sys
    sys.path.insert(0, "./RAGEN")

    print("Loading game files...")
    game_files = get_ood_game_files()
    print(f"Found {len(game_files)} OOD game files")

    print(f"Loading model from {CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    llm = LLM(
        model=CHECKPOINT,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        max_model_len=20480,
        enforce_eager=True,
        enable_chunked_prefill=True,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print("Starting evaluation...")
    results = []
    for i, gf in enumerate(game_files):
        tt = get_task_type(gf)
        t0 = time.time()
        result = run_episode(llm, tokenizer, sampling_params, gf, i)
        elapsed = time.time() - t0
        results.append(result)
        status = "OK" if result["success"] else "FAIL"
        print(f"[{i+1}/{len(game_files)}] {status} {tt} ({result['num_actions']} actions, {elapsed:.1f}s)")

    # Aggregate by task type
    type_stats = defaultdict(lambda: {"count": 0, "success": 0, "total_actions": 0, "total_valid": 0})
    for r in results:
        tt = r["task_type"]
        type_stats[tt]["count"] += 1
        type_stats[tt]["success"] += int(r["success"])
        type_stats[tt]["total_actions"] += r["num_actions"]
        type_stats[tt]["total_valid"] += r["valid_actions"]

    print("\n" + "="*80)
    print("Per-Task-Type Breakdown (seed5 step10, 134 OOD)")
    print("="*80)
    print(f"{'Task Type':<45} {'Count':>5} {'Success':>7} {'Rate':>8} {'Avg Act':>7}")
    print("-"*80)

    total_count = 0
    total_success = 0
    for tt in TASK_TYPES:
        if tt not in type_stats:
            continue
        s = type_stats[tt]
        rate = s["success"] / s["count"] if s["count"] > 0 else 0
        avg_act = s["total_actions"] / s["count"] if s["count"] > 0 else 0
        print(f"{tt:<45} {s['count']:>5} {s['success']:>7} {rate:>7.1%} {avg_act:>7.1f}")
        total_count += s["count"]
        total_success += s["success"]

    overall_rate = total_success / total_count if total_count > 0 else 0
    print("-"*80)
    print(f"{'Total':<45} {total_count:>5} {total_success:>7} {overall_rate:>7.1%}")

    output = {
        "experiment": "seed5_step10",
        "checkpoint": CHECKPOINT,
        "eval_split": "eval_out_of_distribution",
        "total_tasks": total_count,
        "total_success": total_success,
        "overall_success_rate": overall_rate,
        "per_task_type": {},
        "per_episode": results,
    }
    for tt in TASK_TYPES:
        if tt not in type_stats:
            continue
        s = type_stats[tt]
        output["per_task_type"][tt] = {
            "count": s["count"],
            "success": s["success"],
            "success_rate": s["success"] / s["count"] if s["count"] > 0 else 0,
            "avg_actions": s["total_actions"] / s["count"] if s["count"] > 0 else 0,
        }

    save_path = "/data/tmp/seed5_step10_ood_eval.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()
