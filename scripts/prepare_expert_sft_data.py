#!/usr/bin/env python3
"""Validate and filter expert trajectories for SFT training.

Input  : /data/expert_trajectories/expert_trajectories.jsonl
Output : /data/expert_sft/expert_sft_data.jsonl

The expert collector already emits the same {"messages": [...]} format used
by exp008_sft (Qwen3 conversation, identical system prompt), so this script
only performs validation, filtering, and statistics — no structural
conversion.
"""

import argparse
import json
import os
import re
import random
from collections import Counter

VALID_ACTION_PREFIXES = (
    "go to ", "open ", "close ", "take ", "put ", "use ",
    "look", "inventory", "examine ", "clean ", "heat ", "cool ",
    "move ", "slice ",
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def is_valid_action(action: str) -> bool:
    a = action.strip().lower()
    return any(a.startswith(p) or a == p.strip() for p in VALID_ACTION_PREFIXES)


def validate_example(ex, min_turns: int, max_turns: int):
    """Return (ok, reason, n_assistant_turns, actions)."""
    if not isinstance(ex, dict) or "messages" not in ex:
        return False, "no_messages", 0, []
    msgs = ex["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return False, "no_system", 0, []
    if len(msgs) < 3 or msgs[1]["role"] != "user":
        return False, "no_user_task", 0, []

    actions = []
    assistant_turns = 0
    for m in msgs[2:]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            assistant_turns += 1
            ans = ANSWER_RE.search(content)
            if not ans:
                return False, "missing_answer_tag", assistant_turns, actions
            if not THINK_RE.search(content):
                return False, "missing_think_tag", assistant_turns, actions
            action = ans.group(1).strip()
            if not is_valid_action(action):
                return False, "invalid_action", assistant_turns, actions
            actions.append(action)
        elif role == "user":
            continue
        else:
            return False, f"bad_role:{role}", assistant_turns, actions

    if assistant_turns < min_turns:
        return False, "too_short", assistant_turns, actions
    if assistant_turns > max_turns:
        return False, "too_long", assistant_turns, actions
    return True, "ok", assistant_turns, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/data/expert_trajectories/expert_trajectories.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/expert_sft/",
    )
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction held out for val (0 = no split).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print stats, do not write output.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    accepted, rejected = [], []
    reasons = Counter()
    turn_counts = []
    action_first = Counter()
    task_descriptions = Counter()

    with open(args.input) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                rejected.append((line_no, f"json_error: {e}"))
                reasons["json_error"] += 1
                continue
            ok, reason, n_turns, actions = validate_example(
                ex, args.min_turns, args.max_turns
            )
            if ok:
                accepted.append(ex)
                turn_counts.append(n_turns)
                for a in actions:
                    action_first[a.split()[0].lower()] += 1
                user_task = ex["messages"][1]["content"]
                m = re.search(r"Your task is to:\s*(.+)", user_task)
                if m:
                    task_descriptions[m.group(1).strip().rstrip(".")[:60]] += 1
            else:
                rejected.append((line_no, reason))
                reasons[reason] += 1

    print(f"=== Expert SFT Data Quality Report ===")
    print(f"Input file       : {args.input}")
    print(f"Total lines      : {len(accepted) + len(rejected)}")
    print(f"Accepted         : {len(accepted)}")
    print(f"Rejected         : {len(rejected)}")
    if reasons:
        print("Rejection reasons:")
        for r, c in reasons.most_common():
            print(f"  {r}: {c}")

    if turn_counts:
        turn_counts.sort()
        n = len(turn_counts)
        print(f"\n=== Trajectory length (assistant turns) ===")
        print(f"  min/median/max : {turn_counts[0]} / "
              f"{turn_counts[n // 2]} / {turn_counts[-1]}")
        print(f"  mean           : {sum(turn_counts) / n:.1f}")
        print(f"  p90 / p95      : {turn_counts[int(n * 0.9)]} / "
              f"{turn_counts[int(n * 0.95)]}")

    if action_first:
        print(f"\n=== Action verb distribution ===")
        for verb, c in action_first.most_common():
            print(f"  {verb:<10}: {c}")

    if task_descriptions:
        print(f"\n=== Top 10 task types ===")
        for t, c in task_descriptions.most_common(10):
            print(f"  [{c:>4}] {t}")

    if args.dry_run:
        print("\n[dry-run] No output written.")
        return

    if not accepted:
        raise SystemExit("No accepted examples. Refusing to write empty output.")

    os.makedirs(args.output_dir, exist_ok=True)

    random.Random(args.seed).shuffle(accepted)
    if args.val_frac > 0:
        n_val = max(1, int(len(accepted) * args.val_frac))
        val, train = accepted[:n_val], accepted[n_val:]
    else:
        train, val = accepted, []

    train_path = os.path.join(args.output_dir, "expert_sft_train.jsonl")
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nWrote train: {train_path} ({len(train)} examples)")

    if val:
        val_path = os.path.join(args.output_dir, "expert_sft_val.jsonl")
        with open(val_path, "w") as f:
            for ex in val:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote val  : {val_path} ({len(val)} examples)")


if __name__ == "__main__":
    main()
