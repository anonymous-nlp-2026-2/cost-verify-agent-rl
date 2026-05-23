#!/usr/bin/env python3
"""Convert u-10bei/sft_alfworld_trajectory_dataset to RAGEN format."""

import json
import os
import re
from collections import Counter
from datasets import load_dataset

# Bug 2 fix: prepend "You're a helpful assistant. " to match ctx_manager.py:_build_system_content
RAGEN_SYSTEM_PROMPT = """You're a helpful assistant. You are an expert agent in the ALFRED Embodied Environment.
Complete household tasks by navigating and interacting with objects.

Before each action, you MUST first produce a self-guidance assessment:
[Assessment: positive/neutral/negative] - evaluate your current progress.
[Reasoning: one sentence analyzing what has been accomplished and what remains.]
[Suggestion: the best next action from admissible actions.]

Your reasoning with self-guidance MUST be enclosed within <think> </think> tags.
Then choose an admissible action and present it within <answer>...</answer> tags."""

VALID_ACTION_PREFIXES = [
    "go to ", "open ", "close ", "take ", "put ", "use ",
    "look", "inventory", "examine ", "clean ", "heat ", "cool ",
]


def is_valid_action(action):
    a = action.strip().lower()
    return any(a.startswith(p) or a == p.strip() for p in VALID_ACTION_PREFIXES)


def infer_assessment(think_text, is_last):
    if is_last:
        return "positive"
    lower = think_text.lower()
    if any(w in lower for w in ["found it", "i found", "i can see", "there's"]):
        return "positive"
    if any(w in lower for w in ["not here", "no luck", "didn't find", "don't see"]):
        return "negative"
    return "neutral"


def convert_example(example, max_turns=30):
    msgs = example["messages"]
    converted = [{"role": "system", "content": RAGEN_SYSTEM_PROMPT}]

    if len(msgs) < 2 or msgs[1]["role"] != "user":
        return None, "no_user_msg", []

    converted.append({"role": "user", "content": msgs[1]["content"]})

    i = 2
    turn_count = 0
    actions = []

    while i + 1 < len(msgs) and turn_count < max_turns:
        if msgs[i]["role"] != "assistant" or msgs[i + 1]["role"] != "tool":
            break

        think_text = msgs[i]["content"]
        tool_text = msgs[i + 1]["content"]

        if think_text.startswith("Think: "):
            think_text = think_text[7:]

        # Bug 1 fix: extract action directly from tool_calls instead of regex on env response
        tool_calls = msgs[i].get("tool_calls")
        if not tool_calls:
            return None, "no_tool_calls", []
        try:
            action = json.loads(tool_calls[0]["function"]["arguments"])["action"]
        except (KeyError, json.JSONDecodeError, IndexError):
            return None, "tool_calls_parse_fail", []

        if not is_valid_action(action):
            return None, "invalid_action", []

        actions.append(action)
        is_last = (i + 2 >= len(msgs)) or (i + 2 < len(msgs) and msgs[i + 2]["role"] != "assistant")

        assessment = infer_assessment(think_text, is_last)
        assistant_content = (
            f"<think>[Assessment: {assessment}] "
            f"[Reasoning: {think_text}] "
            f"[Suggestion: {action}]</think>"
            f"<answer>{action}</answer>"
        )
        converted.append({"role": "assistant", "content": assistant_content})

        # Add tool response as next user message if more turns follow
        if i + 2 < len(msgs) and msgs[i + 2]["role"] == "assistant":
            converted.append({"role": "user", "content": tool_text})

        turn_count += 1
        i += 2

    if turn_count == 0:
        return None, "no_turns", []

    return {"messages": converted}, "ok", actions


def main():
    os.environ["HF_HOME"] = "/data/.hf_cache"
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    ds = load_dataset("u-10bei/sft_alfworld_trajectory_dataset", split="train")
    print(f"Loaded {len(ds)} examples")

    results = []
    filtered = Counter()
    action_types = Counter()
    turn_counts = []
    task_type_counts = Counter()

    for ex in ds:
        converted, status, actions = convert_example(ex, max_turns=30)
        if status != "ok":
            filtered[status] += 1
            continue
        results.append(converted)
        for a in actions:
            action_types[a.split()[0]] += 1
        turn_counts.append(len(actions))
        meta = ex.get("metadata", {})
        task_type_counts[meta.get("task_type", "unknown")] += 1

    output_path = "/data/sft_data/alfworld_sft_ragen.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n=== Statistics ===")
    print(f"Total: {len(ds)}")
    print(f"Converted: {len(results)}")
    print(f"Filtered: {sum(filtered.values())}")
    for reason, count in filtered.most_common():
        print(f"  {reason}: {count}")
    print(f"\nBy task type:")
    for t, c in task_type_counts.most_common():
        print(f"  {t}: {c}")
    print(f"\nAvg turns: {sum(turn_counts)/len(turn_counts):.1f}")
    print(f"Max turns: {max(turn_counts)}")
    print(f"Min turns: {min(turn_counts)}")
    print(f"\nAction type distribution:")
    for a, c in action_types.most_common():
        print(f"  {a}: {c}")
    print(f"\nSaved to {output_path}")

    # Show one converted example
    print(f"\n=== Sample converted example ===")
    sample = results[0]["messages"]
    for j, msg in enumerate(sample[:6]):
        print(f"[{j}][{msg['role']}]: {msg['content'][:200]}")


if __name__ == "__main__":
    main()
