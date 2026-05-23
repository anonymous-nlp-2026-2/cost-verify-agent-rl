#!/usr/bin/env python3
"""Fix expert SFT data: replace 'move X to Y' with 'put X in/on Y'."""

import json
import os
import re
import sys
from collections import Counter


def fix_move_action(text: str) -> str:
    """Replace 'move OBJ to RECEP' with 'put OBJ in/on RECEP'."""
    return re.sub(r'\bmove (\S+ \d+) to (\S+ \d+)\b', r'put \1 in/on \2', text)


def fix_message(content: str) -> tuple[str, dict]:
    """Fix a single assistant message. Returns (fixed_content, stats)."""
    stats = {"answer_fixed": 0, "think_fixed": 0}
    
    # Fix <answer> content
    def fix_answer(m):
        old = m.group(1)
        new = fix_move_action(old)
        if old != new:
            stats["answer_fixed"] += 1
        return f"<answer>{new}</answer>"
    
    content = re.sub(r'<answer>(.*?)</answer>', fix_answer, content, flags=re.DOTALL)
    
    # Fix <think> content (Suggestion field)
    def fix_think(m):
        old = m.group(1)
        new = fix_move_action(old)
        if old != new:
            stats["think_fixed"] += 1
        return f"<think>{new}</think>"
    
    content = re.sub(r'<think>(.*?)</think>', fix_think, content, flags=re.DOTALL)
    
    return content, stats


def validate_format(content: str) -> dict:
    """Validate a single assistant message matches RAGEN format."""
    has_think = bool(re.search(r'<think>.*?</think>', content, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', content, re.DOTALL))
    full_match = bool(re.search(r'<think>.*?</think>\s*<answer>.*?</answer>', content, re.DOTALL))
    
    action = ""
    m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if m:
        action = m.group(1).strip()
    
    return {
        "has_think": has_think,
        "has_answer": has_answer,
        "full_match": full_match,
        "action": action,
        "has_move": "move " in action and action.startswith("move"),
    }


def process_file(input_path: str, output_path: str) -> dict:
    """Process a single JSONL file."""
    with open(input_path) as f:
        entries = [json.loads(line) for line in f]
    
    total_stats = {
        "total_entries": len(entries),
        "total_assistant_msgs": 0,
        "answer_fixed": 0,
        "think_fixed": 0,
        "pre_format_ok": 0,
        "post_format_ok": 0,
        "post_has_move": 0,
        "action_verbs_before": Counter(),
        "action_verbs_after": Counter(),
    }
    
    fixed_entries = []
    for entry in entries:
        fixed_msgs = []
        for msg in entry["messages"]:
            if msg["role"] == "assistant":
                total_stats["total_assistant_msgs"] += 1
                
                # Pre-fix validation
                pre = validate_format(msg["content"])
                if pre["full_match"]:
                    total_stats["pre_format_ok"] += 1
                if pre["action"]:
                    verb = pre["action"].split()[0]
                    total_stats["action_verbs_before"][verb] += 1
                
                # Fix
                fixed_content, stats = fix_message(msg["content"])
                total_stats["answer_fixed"] += stats["answer_fixed"]
                total_stats["think_fixed"] += stats["think_fixed"]
                
                # Post-fix validation
                post = validate_format(fixed_content)
                if post["full_match"]:
                    total_stats["post_format_ok"] += 1
                if post["has_move"]:
                    total_stats["post_has_move"] += 1
                if post["action"]:
                    verb = post["action"].split()[0]
                    total_stats["action_verbs_after"][verb] += 1
                
                fixed_msgs.append({"role": "assistant", "content": fixed_content})
            else:
                fixed_msgs.append(msg)
        
        fixed_entries.append({"messages": fixed_msgs})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in fixed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return total_stats


def main():
    base_in = "/data/expert_sft"
    base_out = "/data/expert_sft_v2"
    
    for split in ["expert_sft_train.jsonl", "expert_sft_val.jsonl"]:
        input_path = os.path.join(base_in, split)
        output_path = os.path.join(base_out, split)
        
        if not os.path.exists(input_path):
            print(f"SKIP {split}: not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {split}")
        print(f"{'='*60}")
        
        stats = process_file(input_path, output_path)
        
        n = stats["total_assistant_msgs"]
        print(f"  Entries: {stats['total_entries']}")
        print(f"  Assistant messages: {n}")
        print(f"  Answer tags fixed (move→put): {stats['answer_fixed']}")
        print(f"  Think tags fixed (move→put): {stats['think_fixed']}")
        print(f"  Format OK before: {stats['pre_format_ok']}/{n} = {stats['pre_format_ok']/n:.1%}")
        print(f"  Format OK after:  {stats['post_format_ok']}/{n} = {stats['post_format_ok']/n:.1%}")
        print(f"  Remaining 'move' actions: {stats['post_has_move']}")
        print()
        print(f"  Action verbs BEFORE fix:")
        for verb, count in stats["action_verbs_before"].most_common():
            print(f"    {verb}: {count}")
        print(f"  Action verbs AFTER fix:")
        for verb, count in stats["action_verbs_after"].most_common():
            print(f"    {verb}: {count}")
        
        print(f"\n  Output: {output_path}")
    
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
