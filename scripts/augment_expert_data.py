#!/usr/bin/env python3
"""Re-collect expert trajectories WITH admissible actions.

Replays the expert agent on completed game files and captures
admissible_commands at each step. Outputs SFT format with admissible
actions appended to observations, matching RAGEN's _format_observation():

    {obs}\n\nAdmissible actions: [{action1}, {action2}, ...]
"""

import os
import sys
import json
import random
import traceback
import argparse
from collections import Counter

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


def clean_obs(obs):
    if "-= Welcome to TextWorld, ALFRED! =-" in obs:
        obs = obs.split("-= Welcome to TextWorld, ALFRED! =-")[-1]
    return obs.strip()


def format_obs_with_admissible(obs, admissible_actions):
    if admissible_actions:
        actions_str = ", ".join(admissible_actions)
        return f"{obs}\n\nAdmissible actions: [{actions_str}]"
    return obs


def infer_assessment(prev_obs, action):
    p = prev_obs.lower()
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
    return "neutral", "Proceeding with the next step."


def collect_with_admissible(env, agent, game_file, max_steps=150):
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

    admissible = list(getattr(state, 'admissible_commands', []))
    trajectory = [{"obs": state.feedback, "action": None, "admissible": admissible}]
    done = False
    prev_action = ""

    for step in range(max_steps):
        try:
            action = agent.act(state, 0, done, prev_action)
        except (HandCodedAgentTimeout, HandCodedAgentFailed):
            return None, "agent_failed"
        except Exception:
            return None, "act_error"

        try:
            state, reward, done = env.step(action)
        except Exception:
            return None, "step_error"

        admissible = list(getattr(state, 'admissible_commands', []))
        trajectory.append({"obs": state.feedback, "action": action, "admissible": admissible})
        prev_action = action

        if done:
            break

    won = state.get("won", False) if hasattr(state, "get") else getattr(state, "won", False)
    if won:
        return trajectory, "ok"
    return None, "not_won"


def trajectory_to_sft(trajectory):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    init_obs = clean_obs(trajectory[0]["obs"])
    init_obs = format_obs_with_admissible(init_obs, trajectory[0]["admissible"])
    messages.append({"role": "user", "content": init_obs})

    for i in range(1, len(trajectory)):
        action = trajectory[i]["action"]
        prev_obs = trajectory[i - 1]["obs"]
        assessment, reasoning = infer_assessment(prev_obs, action)
        assistant_content = (
            f"<think>[Assessment: {assessment}] "
            f"[Reasoning: {reasoning}] "
            f"[Suggestion: {action}]</think>"
            f"<answer>{action}</answer>"
        )
        messages.append({"role": "assistant", "content": assistant_content})
        if i < len(trajectory) - 1:
            obs = trajectory[i]["obs"]
            obs = format_obs_with_admissible(obs, trajectory[i]["admissible"])
            messages.append({"role": "user", "content": obs})

    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-file", default="/data/expert_trajectories/progress.json")
    parser.add_argument("--output-dir", default="/data/expert_sft_v3/")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--max-games", type=int, default=0, help="0 = all")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading progress from {args.progress_file}...")
    sys.stdout.flush()
    with open(args.progress_file) as f:
        progress = json.load(f)
    game_files = sorted(progress["completed"])
    print(f"Loaded {len(game_files)} completed game files")
    sys.stdout.flush()

    if args.max_games > 0:
        game_files = game_files[:args.max_games]
        print(f"Limited to {args.max_games} games")
        sys.stdout.flush()

    request_infos = textworld.EnvInfos(
        won=True,
        admissible_commands=True,
        facts=True,
    )

    def make_demangler(env):
        return AlfredDemangler(env, shuffle=False)

    print(f"Initializing TextWorld env with first game: {game_files[0][:80]}...")
    sys.stdout.flush()

    try:
        env = textworld.start(
            game_files[0],
            request_infos=request_infos,
            wrappers=[make_demangler, AlfredInfos],
        )
        print("TextWorld env initialized successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"FATAL: Failed to initialize TextWorld env: {e}")
        traceback.print_exc()
        sys.exit(1)

    agent = HandCodedTWAgent(max_steps=args.max_steps)
    print("Expert agent initialized")
    sys.stdout.flush()

    os.makedirs(args.output_dir, exist_ok=True)

    # Stream output directly to file to avoid memory accumulation
    train_path = os.path.join(args.output_dir, "expert_sft_all.jsonl")
    out_f = open(train_path, "w")
    
    stats = Counter()
    success_count = 0
    admissible_sum = 0
    admissible_n = 0

    for idx, game_file in enumerate(game_files):
        try:
            trajectory, status = collect_with_admissible(env, agent, game_file, args.max_steps)
        except Exception as e:
            print(f"EXCEPTION at game {idx}: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            stats["exception"] += 1
            continue

        if status == "ok" and trajectory:
            sft_data = trajectory_to_sft(trajectory)
            out_f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")
            out_f.flush()
            success_count += 1
            for step in trajectory:
                if step["admissible"]:
                    admissible_sum += len(step["admissible"])
                    admissible_n += 1

        stats[status] += 1

        if (idx + 1) % args.log_interval == 0 or idx == len(game_files) - 1:
            print(f"[{idx + 1}/{len(game_files)}] success={success_count} | stats={dict(stats)}")
            sys.stdout.flush()

    out_f.close()
    env.close()

    if success_count == 0:
        print("ERROR: No successful trajectories collected!")
        sys.exit(1)

    # Read all, shuffle, split
    print(f"\nShuffling and splitting {success_count} trajectories...")
    sys.stdout.flush()
    
    with open(train_path) as f:
        all_sft = [json.loads(line) for line in f]

    random.Random(args.seed).shuffle(all_sft)

    n_val = max(1, int(len(all_sft) * args.val_frac))
    val, train = all_sft[:n_val], all_sft[n_val:]

    final_train_path = os.path.join(args.output_dir, "expert_sft_train.jsonl")
    with open(final_train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    val_path = os.path.join(args.output_dir, "expert_sft_val.jsonl")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    os.remove(train_path)  # cleanup temp file

    print(f"\n=== Final Summary ===")
    print(f"Total game files: {len(game_files)}")
    print(f"Successful: {success_count}")
    print(f"Train: {len(train)}, Val: {len(val)}")
    print(f"Stats: {dict(stats)}")
    if admissible_n > 0:
        print(f"Avg admissible actions per step: {admissible_sum/admissible_n:.1f}")
    print(f"\nOutput:")
    print(f"  Train: {final_train_path}")
    print(f"  Val: {val_path}")

    # Action verb check
    move_count = 0
    put_count = 0
    total_actions = 0
    for ex in all_sft[:100]:
        for m in ex["messages"]:
            if m["role"] == "assistant" and "<answer>" in m["content"]:
                total_actions += 1
                s = m["content"]
                ans = s[s.index("<answer>")+8:s.index("</answer>")].strip()
                if ans.startswith("move"):
                    move_count += 1
                if ans.startswith("put"):
                    put_count += 1
    print(f"\n=== Action check (first 100 trajectories) ===")
    print(f"  Total actions: {total_actions}")
    print(f"  'move' actions: {move_count}")
    print(f"  'put' actions: {put_count}")

    # Sample
    if all_sft:
        sample = all_sft[0]
        print(f"\n=== Sample (first 4 messages) ===")
        for m in sample["messages"][:4]:
            print(f"--- {m['role']} ---")
            print(m["content"][:400])
            print()
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
