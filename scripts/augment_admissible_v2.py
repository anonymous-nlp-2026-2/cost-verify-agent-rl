#!/usr/bin/env python3
"""Re-collect expert trajectories WITH admissible actions for exp019b.

Re-runs HandCodedTWAgent on all train games, recording admissible_commands
at each step. Outputs SFT data with admissible actions in RAGEN format:
  "{obs}\n\nAdmissible actions: [{action1}, {action2}, ...]"

Output: /data/expert_sft_v3/expert_sft_{train,val}.jsonl
"""

import os
import json
import random
import re
import argparse
from collections import Counter

os.environ["ALFWORLD_DATA"] = "/data/alfworld"

import textworld
from alfworld.agents.expert import HandCodedTWAgent, HandCodedAgentTimeout, HandCodedAgentFailed
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos


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

SUPPORTED_TASK_TYPES = {
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
}

ALL_KNOWN_TASK_TYPES = SUPPORTED_TASK_TYPES | {"pick_and_place_with_movable_recep"}

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

VALID_ACTION_PREFIXES = (
    "go to ", "open ", "close ", "take ", "put ", "use ",
    "look", "inventory", "examine ", "clean ", "heat ", "cool ",
    "move ", "slice ",
)


def find_game_files(data_dir, split="train"):
    split_dir = os.path.join(data_dir, split)
    game_files = []
    skipped = Counter()
    for task_dir in sorted(os.listdir(split_dir)):
        task_path = os.path.join(split_dir, task_dir)
        if not os.path.isdir(task_path):
            continue
        task_type = None
        for tt in ALL_KNOWN_TASK_TYPES:
            if task_dir.startswith(tt + "-"):
                task_type = tt
                break
        if task_type is None or task_type not in SUPPORTED_TASK_TYPES:
            skipped[task_type or "unknown"] += 1
            continue
        for trial_dir in sorted(os.listdir(task_path)):
            trial_path = os.path.join(task_path, trial_dir)
            if not os.path.isdir(trial_path):
                continue
            game_file = os.path.join(trial_path, "game.tw-pddl")
            traj_file = os.path.join(trial_path, "traj_data.json")
            if os.path.exists(game_file) and os.path.exists(traj_file):
                game_files.append(game_file)
    if skipped:
        print(f"Skipped task types: {dict(skipped)}")
    return game_files


def clean_obs(obs):
    if "-= Welcome to TextWorld, ALFRED! =-" in obs:
        obs = obs.split("-= Welcome to TextWorld, ALFRED! =-")[-1]
    return obs.strip()


def format_observation(obs, admissible_actions):
    actions_str = ", ".join(admissible_actions)
    return f"{obs}\n\nAdmissible actions: [{actions_str}]"


def infer_assessment(prev_obs, action):
    p = prev_obs.lower()
    a = (action or "").lower()
    if "you pick up" in p or "you take" in p:
        return "positive", "Successfully picked up the object."
    if "you put" in p or "you move" in p:
        return "positive", "Successfully placed the object."
    if "nothing happens" in p:
        return "negative", "Action had no effect. Need a different approach."
    if "you open" in p:
        return "neutral", "Opened the receptacle to check contents."
    if "you clean" in p:
        return "positive", "Successfully cleaned the object."
    if "you heat" in p:
        return "positive", "Successfully heated the object."
    if "you cool" in p:
        return "positive", "Successfully cooled the object."
    if "you use" in p:
        return "positive", "Used the lamp to examine the object."
    if "closed" in p:
        return "neutral", "Found a closed receptacle. Should open it."
    if "you see nothing" in p:
        return "negative", "Nothing useful here. Searching elsewhere."
    if "you see" in p:
        return "neutral", "Observing the area for the target object."
    if "go to" in a:
        return "neutral", "Moving to check another location."
    if "look" in a:
        return "neutral", "Looking around to assess the situation."
    return "neutral", "Continuing to work on the task."


def trajectory_to_sft_with_admissible(trajectory):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    init_obs = clean_obs(trajectory[0]["obs"])
    init_admissible = trajectory[0].get("admissible", [])
    messages.append({"role": "user", "content": format_observation(init_obs, init_admissible)})

    for i in range(1, len(trajectory)):
        entry = trajectory[i]
        action = entry["action"]
        obs = clean_obs(entry["obs"])
        admissible = entry.get("admissible", [])
        is_last = (i == len(trajectory) - 1)
        prev_obs = clean_obs(trajectory[i - 1]["obs"])

        if is_last:
            assessment, reasoning = "positive", "Task completed successfully."
        else:
            assessment, reasoning = infer_assessment(prev_obs, action)

        assistant_content = (
            f"<think>[Assessment: {assessment}] "
            f"[Reasoning: {reasoning}] "
            f"[Suggestion: {action}]</think>"
            f"<answer>{action}</answer>"
        )
        messages.append({"role": "assistant", "content": assistant_content})
        if not is_last:
            messages.append({"role": "user", "content": format_observation(obs, admissible)})

    return {"messages": messages}


def collect_single_game(env, agent, game_file, max_steps=150):
    try:
        env.load(game_file)
        state = env.reset()
    except Exception as e:
        return None, f"load_error: {e}"

    try:
        agent.reset(game=game_file)
        agent.observe(state.feedback)
    except Exception as e:
        return None, f"agent_init_error: {e}"

    done = False
    prev_action = ""
    initial_admissible = list(state.admissible_commands) if hasattr(state, 'admissible_commands') else []
    trajectory = [{"obs": state.feedback, "action": None, "admissible": initial_admissible}]

    for step in range(max_steps):
        try:
            action = agent.act(state, 0, done, prev_action)
        except (HandCodedAgentTimeout, HandCodedAgentFailed):
            return None, "agent_failed"
        except Exception as e:
            return None, f"act_error: {e}"

        try:
            state, reward, done = env.step(action)
        except Exception as e:
            return None, f"step_error: {e}"

        prev_action = action
        step_admissible = list(state.admissible_commands) if hasattr(state, 'admissible_commands') else []
        trajectory.append({"obs": state.feedback, "action": action, "admissible": step_admissible})

        if done:
            break

    won = state.get("won", False) if hasattr(state, "get") else getattr(state, "won", False)
    if won:
        return trajectory, "ok"
    return None, "not_won"


def is_valid_action(action):
    a = action.strip().lower()
    return any(a.startswith(p) or a == p.strip() for p in VALID_ACTION_PREFIXES)


def validate_example(ex, min_turns=2, max_turns=40):
    if not isinstance(ex, dict) or "messages" not in ex:
        return False, "no_messages"
    msgs = ex["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return False, "no_system"
    if len(msgs) < 3 or msgs[1]["role"] != "user":
        return False, "no_user_task"

    assistant_turns = 0
    for m in msgs[2:]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            assistant_turns += 1
            ans = ANSWER_RE.search(content)
            if not ans:
                return False, "missing_answer_tag"
            if not THINK_RE.search(content):
                return False, "missing_think_tag"
            action = ans.group(1).strip()
            if not is_valid_action(action):
                return False, "invalid_action"
        elif role == "user":
            continue
        else:
            return False, f"bad_role:{role}"

    if assistant_turns < min_turns:
        return False, "too_short"
    if assistant_turns > max_turns:
        return False, "too_long"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/alfworld/json_2.1.1")
    parser.add_argument("--output-dir", default="/data/expert_sft_v3")
    parser.add_argument("--max-games", type=int, default=0, help="0 = all")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    game_files = find_game_files(args.data_dir, args.split)
    print(f"Found {len(game_files)} game files in {args.split} split")
    import sys
    sys.stdout.flush()

    if args.max_games > 0:
        game_files = game_files[:args.max_games]
        print(f"Limited to {args.max_games} games")

    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)

    def make_demangler(env):
        return AlfredDemangler(env, shuffle=False)

    print("Starting textworld env...", flush=True)
    env = textworld.start(
        game_files[0],
        request_infos=request_infos,
        wrappers=[make_demangler, AlfredInfos],
    )
    print("Env started. Creating agent...", flush=True)
    agent = HandCodedTWAgent(max_steps=args.max_steps)
    print("Agent created. Starting collection...", flush=True)

    all_sft = []
    stats = Counter()
    move_action_count = 0
    put_action_count = 0
    admissible_move_count = 0
    admissible_put_count = 0

    for idx, game_file in enumerate(game_files):
        trajectory, status = collect_single_game(env, agent, game_file, args.max_steps)

        if status == "ok" and trajectory:
            sft_data = trajectory_to_sft_with_admissible(trajectory)
            ok, reason = validate_example(sft_data)
            if ok:
                all_sft.append(sft_data)
                for step in trajectory:
                    if step["action"]:
                        if step["action"].startswith("move "):
                            move_action_count += 1
                        if step["action"].startswith("put "):
                            put_action_count += 1
                    for ac in step.get("admissible", []):
                        if ac.startswith("move "):
                            admissible_move_count += 1
                        if ac.startswith("put "):
                            admissible_put_count += 1
            else:
                stats[f"validate_{reason}"] += 1

        stats[status] += 1

        if (idx + 1) % 100 == 0 or idx == 0 or idx == len(game_files) - 1:
            print(
                f"[{idx + 1}/{len(game_files)}] "
                f"valid_sft={len(all_sft)} | stats={dict(stats)} | "
                f"move_actions={move_action_count} put_actions={put_action_count} | "
                f"admissible_move={admissible_move_count} admissible_put={admissible_put_count}"
            )

    env.close()

    print(f"\nTotal valid SFT examples: {len(all_sft)}")

    if not all_sft:
        print("ERROR: No valid examples collected!")
        return

    random.Random(args.seed).shuffle(all_sft)
    n_val = max(1, int(len(all_sft) * args.val_frac))
    val, train = all_sft[:n_val], all_sft[n_val:]

    train_path = os.path.join(args.output_dir, "expert_sft_train.jsonl")
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote train: {train_path} ({len(train)} examples)")

    val_path = os.path.join(args.output_dir, "expert_sft_val.jsonl")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote val: {val_path} ({len(val)} examples)")

    # Sample output
    print("\n=== Sample entry (first user message) ===")
    sample = train[0]
    for m in sample["messages"][:3]:
        content = m["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"[{m['role']}] {content}")
    print()

    # Check admissible actions in output
    admissible_found = 0
    move_preserved = 0
    for ex in train[:50]:
        for m in ex["messages"]:
            if m["role"] == "user" and "Admissible actions:" in m["content"]:
                admissible_found += 1
                break
        for m in ex["messages"]:
            if m["role"] == "assistant" and "move " in m["content"]:
                move_preserved += 1
                break
    print(f"Verification (first 50 train): admissible_found={admissible_found}/50, move_preserved={move_preserved}")


if __name__ == "__main__":
    main()
